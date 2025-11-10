import numpy as np
import time
import os
import pyautogui
import threading
import easyocr
from dotenv import load_dotenv
from Actions import Actions
from OCR import get_tower_health_values
from inference_sdk import InferenceHTTPClient
from typing import List, Dict, Any, Union, Optional, Tuple
import tempfile
import traceback

# Load environment variables from .env file
load_dotenv()

MAX_ENEMIES = 10
MAX_ALLIES = 10

SPELL_CARDS = ["Fireball", "Zap", "Arrows", "Tornado", "Rocket", "Lightning", "Freeze"]

class ClashRoyaleEnv:
    def __init__(self):
        self.actions = Actions()
        self.rf_model = self.setup_roboflow()
        self.card_model = self.setup_card_roboflow()
        
        self.state_size = 1 + 2 * (MAX_ALLIES + MAX_ENEMIES) + 12
        
        self.num_cards = 4
        self.grid_width = 18
        self.grid_height = 28

        self.screenshot_path = os.path.join(os.path.dirname(__file__), 'screenshots', "current.png")
        self.available_actions = self.get_available_actions()
        self.action_size = len(self.available_actions)
        self.current_cards = []

        self.game_over_flag = None
        self._endgame_thread = None
        self._endgame_thread_stop = threading.Event()

        self.prev_elixir = None
        self.prev_enemy_presence = None
        self.prev_enemy_princess_towers = None
        self.match_over_detected = False

        self.ocr_reader = easyocr.Reader(['en'], gpu=True)
        
        self.center_xcoord = 231
        
        self.first_enemy_tower_check: int = 0
        self.second_enemy_tower_check: int = 0
        self.first_ally_tower_check: int = 0
        self.second_ally_tower_check: int = 0

        self.max_tower_health: List[Optional[int]] = [None]*6
        self.tower_health_values: List[Optional[int]] = [None]*6

        self._capture_lock = threading.Lock()
        self._init_thread: Optional[threading.Thread] = None

    def setup_roboflow(self):
        api_key = os.getenv('ROBOFLOW_API_KEY')
        if not api_key:
            raise ValueError("ROBOFLOW_API_KEY environment variable is not set. Please check your .env file.")
        
        client = InferenceHTTPClient(
            api_url="http://localhost:9001",
            api_key=api_key
        )
        return client

    def setup_card_roboflow(self):
        api_key = os.getenv('ROBOFLOW_API_KEY')
        if not api_key:
            raise ValueError("ROBOFLOW_API_KEY environment variable is not set. Please check your .env file.")
        
        client = InferenceHTTPClient(
            api_url="http://localhost:9001",
            api_key=api_key
        )
        return client

    def reset(self):
        if self._endgame_thread and self._endgame_thread.is_alive():
            self._endgame_thread_stop.set()
            self._endgame_thread.join(timeout=2)
        
        if self._init_thread and self._init_thread.is_alive():
            self._init_thread.join(timeout=2)

        # Instead, just wait for the new game to load after clicking "Play Again"
        time.sleep(3)
        self.game_over_flag = None

        self._endgame_thread_stop.clear()
        self._endgame_thread = threading.Thread(target=self._endgame_watcher, daemon=True)
        self._endgame_thread.start()

        self.prev_elixir = None
        self.prev_enemy_presence = None
        self.match_over_detected = False
        self.max_tower_health = [None]*6

        self._init_thread = threading.Thread(target=self._initialize_max_tower_health, daemon=True)
        self._init_thread.start()
        
        initial_state = self._get_state()
        self.prev_enemy_princess_towers = self._count_enemy_princess_towers(self.current_predictions)

        self.first_enemy_tower_check = 0
        self.second_enemy_tower_check = 0
        self.first_ally_tower_check = 0
        self.second_ally_tower_check = 0
        
        return initial_state
        
    def _initialize_max_tower_health(self, interval=1.0, timeout = 150):
        start_time = time.time()
        while True:
            with self._capture_lock:
                self.actions.capture_tower_health()
                current_values = get_tower_health_values(self.ocr_reader)

            for i, val in enumerate(current_values):
                if val is not None:
                    if self.max_tower_health[i] is None:
                        self.max_tower_health[i] = val
            
            #Stop checking after 2.5 minutes
            if time.time() - start_time > timeout:
                print("Max tower health initialization timed out.")
                break
            
            # Check if all 6 have been initialized
            if all(v is not None for v in self.max_tower_health):
                break
            
            # Sleep a bit before next OCR cycle
            time.sleep(interval)
        
    def normalize_health(self):
        normal_health_val_and_flag = []

        #if all health and max health values are defined
        all_values_present = all(i is not None for i in self.max_tower_health) and \
                             all(i is not None for i in self.tower_health_values)

        if all_values_present:
            health = np.array(self.tower_health_values)
            max_health = np.array(self.max_tower_health)
            normal_health_values = list(health / max_health)
            
            #when reliability flag = 1 that means the value was read from the ocr
            reliability_flags = np.ones_like(normal_health_values)
            
            stacked = np.stack((normal_health_values, reliability_flags))
            normal_health_val_and_flag = stacked.flatten('F').tolist()
        #if one or more of the values is undefined
        else:
            for i in range(6):
                #if i in max_health or tower_health is None
                max_health_is_none = self.max_tower_health[i] is None
                current_health_is_none = self.tower_health_values[i] is None

                if max_health_is_none or current_health_is_none:
                    #-1 means undefined
                    normal_health_val_and_flag.append(-1) 
                    #when the normal health value is -1 the reliability is 0 because the -1 is neant to be a place holder
                    normal_health_val_and_flag.append(0)
                else:
                    new_normal = self.tower_health_values[i] / self.max_tower_health[i]
                    normal_health_val_and_flag.append(new_normal)
                    normal_health_val_and_flag.append(1)
                    
        return normal_health_val_and_flag

    def close(self):
        self._endgame_thread_stop.set()
        if self._endgame_thread:
            self._endgame_thread.join(timeout=2)
        if self._init_thread and self._init_thread.is_alive():
            self._init_thread.join(timeout=2)

    def step(self, action_index):
        # --- THIS IS THE CORRECTED SECTION ---
        #game over check
        if self.game_over_flag:
            done = True
            final_state = self._get_state()
            reward = self._compute_reward(final_state)
            result = self.game_over_flag
            if result == "victory":
                reward += 100
                print("Victory detected - ending episode")
            elif result == "defeat":
                reward -= 100
                print("Defeat detected - ending episode")
            
            self.match_over_detected = False
            return final_state, reward, done
        # --- END OF CORRECTION ---
            
        # Check for match over (mid-game)
        if not self.match_over_detected and hasattr(self.actions, "detect_match_over") and self.actions.detect_match_over():
            print("Match over detected (matchover.png), forcing no-op until next game.")
            self.match_over_detected = True

        # If match over, only allow no-op action (last action in list)
        if self.match_over_detected:
            action_index = len(self.available_actions) - 1

        self.current_cards = self.detect_cards_in_hand()
        print("\nCurrent cards in hand:", self.current_cards)

        # If all cards are "Unknown", click at (1611, 831) and return no-op
        if all(card == "Unknown" for card in self.current_cards):
            pyautogui.moveTo(1611, 831, duration=0.2)
            pyautogui.click()
            # Return current state, zero reward, not done
            next_state = self._get_state()
            return next_state, 0, False

        action = self.available_actions[action_index]
        card_index, x_frac, y_frac = action
        print(f"Action selected: card_index={card_index}, x_frac={x_frac:.2f}, y_frac={y_frac:.2f}")

        card_was_played = False
        card_name = ""
        x, y = 0, 0

        if card_index != -1 and card_index < len(self.current_cards):
            card_was_played = True
            card_name = self.current_cards[card_index]
            print(f"Attempting to play {card_name}")
            x = int(x_frac * self.actions.WIDTH) + self.actions.TOP_LEFT_X
            y = int(y_frac * self.actions.HEIGHT) + self.actions.TOP_LEFT_Y
            self.actions.card_play(x, y, card_index)
            # You can reduce this if needed
            time.sleep(1)
        
        next_state = self._get_state()
        reward = self._compute_reward(next_state)
        
        # --- Spell penalty logic ---
        if card_was_played and card_name in SPELL_CARDS:
            enemy_positions = []
            for i in range(1 + 2 * MAX_ALLIES, 1 + 2 * MAX_ALLIES + 2 * MAX_ENEMIES, 2):
                ex = next_state[i]
                ey = next_state[i + 1]
                if ex != 0.0 or ey != 0.0:
                    ex_px = int(ex * self.actions.WIDTH) + self.actions.TOP_LEFT_X
                    ey_px = int(ey * self.actions.HEIGHT) + self.actions.TOP_LEFT_Y
                    enemy_positions.append((ex_px, ey_px))

            radius = 100
            found_enemy = any((abs(ex - x) ** 2 + abs(ey - y) ** 2) ** 0.5 < radius for ex, ey in enemy_positions)
            
            if not found_enemy:
                # Penalize for wasting spell
                reward -= 5

        # --- Princess tower reward logic ---
        current_enemy_princess_towers = self._count_enemy_princess_towers(self.current_predictions)
        if self.prev_enemy_princess_towers is not None:
            if current_enemy_princess_towers < self.prev_enemy_princess_towers:
                reward += 20
        self.prev_enemy_princess_towers = current_enemy_princess_towers

        done = False
        return next_state, reward, done

    def _get_state(self):
        elixir = self.actions.count_elixir()
        
        tmp_path = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                tmp_path = tmp.name

            with self._capture_lock:
                self.actions.capture_tower_health()
                self.tower_health_values = get_tower_health_values(self.ocr_reader)
                self.actions.capture_area(tmp_path)

            workspace_name = os.getenv('WORKSPACE_TROOP_DETECTION')
            if not workspace_name:
                raise ValueError("WORKSPACE_TROOP_DETECTION environment variable is not set. Please check your .env file.")
            
            results = self.rf_model.run_workflow(
                workspace_name=workspace_name,
                workflow_id="detect-count-and-visualize",
                images={"image": tmp_path}
            )
        finally:
            if tmp_path and os.path.exists(tmp_path):
                os.remove(tmp_path)
        
        if isinstance(results, list) and results:
            first = results[0]
            if isinstance(first, dict) and "predictions" in first:
                self.current_predictions = first["predictions"]
            else:
                self.current_predictions = []
        else:
            self.current_predictions = []
        
        health_state = self.normalize_health()
        health_state = self._correct_destroyed_towers(health_state, self.current_predictions)

        print("Predictions:", self.current_predictions)
        if not self.current_predictions:
            print("WARNING: No predictions found in results")
            zero_state_components = [elixir / 10.0] + \
                                    [0.0] * (2 * (MAX_ALLIES + MAX_ENEMIES)) + \
                                    health_state
            zero_state = np.array(zero_state_components, dtype=np.float32)
            # Still include health state
            return zero_state

        print("RAW predictions:", self.current_predictions)
        print("Detected classes:", [repr(p.get("class", "")) for p in self.current_predictions if isinstance(p, dict)])

        TOWER_CLASSES = {
            "ally king tower",
            "ally princess tower",
            "enemy king tower",
            "enemy princess tower"
        }

        def normalize_class(cls):
            return cls.strip().lower() if isinstance(cls, str) else ""

        allies = []
        for p in self.current_predictions:
            if isinstance(p, dict) and \
               normalize_class(p.get("class", "")) not in TOWER_CLASSES and \
               normalize_class(p.get("class", "")).startswith("ally") and \
               "x" in p and "y" in p:
                allies.append((p["x"], p["y"]))

        enemies = []
        for p in self.current_predictions:
            if isinstance(p, dict) and \
               normalize_class(p.get("class", "")) not in TOWER_CLASSES and \
               normalize_class(p.get("class", "")).startswith("enemy") and \
               "x" in p and "y" in p:
                enemies.append((p["x"], p["y"]))

        print("Allies:", allies)
        print("Enemies:", enemies)

        def pad_units(units, max_units):
            normalized_units = []
            for x, y in units:
                norm_x = x / self.actions.WIDTH
                norm_y = y / self.actions.HEIGHT
                normalized_units.append((norm_x, norm_y))
            
            padding_needed = max_units - len(normalized_units)
            padding = [(0.0, 0.0)] * padding_needed
            
            return normalized_units + padding

        ally_positions = pad_units(allies, MAX_ALLIES)
        ally_flat = [coord for pos in ally_positions for coord in pos]
        
        enemy_positions = pad_units(enemies, MAX_ENEMIES)
        enemy_flat = [coord for pos in enemy_positions for coord in pos]

        state_components = [elixir / 10.0] + ally_flat + enemy_flat + health_state
        state = np.array(state_components, dtype=np.float32)
        return state

    def _compute_reward(self, state):
        if state is None:
            return 0

        elixir = state[0] * 10

        # Only y coords so it does not bias left/right side
        enemy_positions = state[1 + 2 * MAX_ALLIES:-12]
        enemy_presence = sum(enemy_positions[1::2])
        
        reward = -enemy_presence * 0.5

        # Elixir efficiency: reward for spending elixir if it reduces enemy presence
        if self.prev_elixir is not None and self.prev_enemy_presence is not None:
            elixir_spent = self.prev_elixir - elixir
            enemy_reduced = self.prev_enemy_presence - enemy_presence
            if elixir_spent > 0 and enemy_reduced > 0:
                # tune this factor
                reward += 2 * min(elixir_spent, enemy_reduced)

        self.prev_elixir = elixir
        self.prev_enemy_presence = enemy_presence

        return reward

    def detect_cards_in_hand(self):
        try:
            with self._capture_lock:
                card_paths = self.actions.capture_individual_cards()
            
            print("\nTesting individual card predictions:")

            cards = []
            workspace_name = os.getenv('WORKSPACE_CARD_DETECTION')
            if not workspace_name:
                raise ValueError("WORKSPACE_CARD_DETECTION environment variable is not set. Please check your .env file.")
            
            for card_path in card_paths:
                results = self.card_model.run_workflow(
                    workspace_name=workspace_name,
                    workflow_id="custom-workflow",
                    images={"image": card_path}
                )
                
                # Fix: parse nested structure
                predictions = []
                if isinstance(results, list) and results:
                    preds_dict = results[0].get("predictions", {})
                    if isinstance(preds_dict, dict):
                        predictions = preds_dict.get("predictions", [])
                
                if predictions:
                    card_name = predictions[0]["class"]
                    print(f"Detected card: {card_name}")
                    cards.append(card_name)
                else:
                    print("No card detected.")
                    cards.append("Unknown")
            return cards
        except Exception as e:
            print(f"Error in detect_cards_in_hand: {e}")
            return []

    def get_available_actions(self):
        actions = []
        for card in range(self.num_cards):
            for x in range(self.grid_width):
                for y in range(self.grid_height):
                    action = [
                        card, 
                        x / (self.grid_width - 1), 
                        y / (self.grid_height - 1)
                    ]
                    actions.append(action)
        
        # No-op action
        actions.append([-1, 0, 0])
        return actions

    def _endgame_watcher(self):
        while not self._endgame_thread_stop.is_set():
            result = self.actions.detect_game_end()
            if result:
                self.game_over_flag = result
                break
            # Sleep a bit to avoid hammering the CPU
            time.sleep(0.5)

    def _count_enemy_princess_towers(self, predictions: List[Dict]) -> int:
        count = 0
        for p in predictions:
            if isinstance(p, dict) and p.get("class") == "enemy princess tower":
                count += 1
        return count
        
    def _count_ally_princess_towers(self, predictions: List[Dict]) -> int:
        count = 0
        for p in predictions:
            if isinstance(p, dict) and p.get("class") == "ally princess tower":
                count += 1
        return count
    
    def _get_all_predictions(self):
        self.actions.capture_area(self.screenshot_path)
        workspace_name = os.getenv('WORKSPACE_TROOP_DETECTION')
        if not workspace_name:
            raise ValueError("WORKSPACE_TROOP_DETECTION environment variable is not set. Please check your .env file.")
            
        results = self.rf_model.run_workflow(
            workspace_name=workspace_name,
            workflow_id="detect-count-and-visualize",
            images={"image": self.screenshot_path}
        )
        
        predictions = []
        if isinstance(results, dict) and "predictions" in results:
            predictions = results["predictions"]
        elif isinstance(results, list) and results:
            first = results[0]
            if isinstance(first, dict) and "predictions" in first:
                predictions = first["predictions"]
        return predictions
        
    def _get_tower_data(self, predictions: List[Dict], class_name: str) -> Tuple[int, Optional[float]]:
        tower_xcoords = []
        for p in predictions:
            # We need the x coordinate for distinguishing left/right tower
            if isinstance(p, dict) and p.get("class") == class_name and 'x' in p:
                tower_xcoords.append(p["x"])
        
        count = len(tower_xcoords)
        x_coord = None
        # If there is exactly one tower, return its x coordinate
        if count == 1:
            x_coord = tower_xcoords[0] 
            
        return count, x_coord
        
    def _correct_destroyed_towers(self, health_state: List[float], predictions: List[Dict]) -> List[float]:
        e_count, e_xcoord = self._get_tower_data(predictions, "enemy princess tower")
        a_count, a_xcoord = self._get_tower_data(predictions, "ally princess tower")
        
        #---- Enemy Correction ----
        if e_count == 1:
            self.first_enemy_tower_check += 1
            self.second_enemy_tower_check = 0
        elif e_count == 0:
            self.second_enemy_tower_check += 1
            self.first_enemy_tower_check = 0
        else:
            self.first_enemy_tower_check = 0
            self.second_enemy_tower_check = 0
        
        if self.first_enemy_tower_check >= 2 and e_xcoord is not None:
            if e_xcoord > self.center_xcoord:
                health_state[0] = 0 
                health_state[1] = 1
            elif e_xcoord < self.center_xcoord:
                health_state[4] = 0
                health_state[5] = 1
        elif self.second_enemy_tower_check >=2:
            health_state[0] = 0 
            health_state[1] = 1
            health_state[4] = 0
            health_state[5] = 1
        
        #---- Ally Correction ----
        if a_count == 1:
            self.first_ally_tower_check += 1
            self.second_ally_tower_check = 0
        elif a_count == 0:
            self.second_ally_tower_check += 1
            self.first_ally_tower_check = 0
        else:
            self.first_ally_tower_check = 0
            self.second_ally_tower_check = 0
        
        if self.first_ally_tower_check >= 2 and a_xcoord is not None:
            if a_xcoord > self.center_xcoord:
                health_state[6] = 0 
                health_state[7] = 1
            elif a_xcoord < self.center_xcoord:
                health_state[10] = 0
                health_state[11] = 1
        elif self.second_ally_tower_check >=2:
            health_state[6] = 0 
            health_state[7] = 1
            health_state[10] = 0
            health_state[11] = 1
            
        return health_state
