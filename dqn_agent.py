import os
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
from typing import List

# --- Network Definition ---

class DQN(nn.Module):
    def __init__(self, input_dim: int, output_dim: int):
        super(DQN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )

    def forward(self, x):
        return self.net(x)

class DQNAgent:
    def __init__(self, state_size: int, action_size: int):
        self.state_size = state_size
        self.action_size = action_size
        
        self.model = DQN(self.state_size, self.action_size)
        self.target_model = DQN(self.state_size, self.action_size)
        
        self.update_target_model()
        
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.criterion = nn.MSELoss()
        self.memory = deque(maxlen=10000)
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.997

    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def remember(self, s, a, r, s2, done):
        self.memory.append((s, a, r, s2, done))

    def act(self, state: List[float]) -> int:
        if random.random() < self.epsilon:
            return random.randrange(self.action_size)
        
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            q_values = self.model(state_tensor)
        return q_values.argmax().item()

    def replay(self, batch_size: int):
        if len(self.memory) < batch_size:
            return
            
        batch = random.sample(self.memory, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(next_states)
        dones = torch.FloatTensor(dones)

        current_q_values = self.model(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        next_q_values = self.target_model(next_states).max(1)[0]
        
        target_q_values = rewards + self.gamma * next_q_values * (1 - dones)

        loss = self.criterion(current_q_values, target_q_values.detach())

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load(self, filename: str):
        path = filename
        if not os.path.isabs(filename):
            path = os.path.join("models", filename)
            
        try:
            self.model.load_state_dict(torch.load(path))
            self.update_target_model()
            self.model.eval()
            
            self.epsilon = 0.1 
            print(f"Loaded model weights from {path}. Epsilon set to 0.1 for continued training.")
            
        except RuntimeError as e:
            if "size mismatch for net.0.weight" in str(e):
                print(f"WARNING: Skipping model loading due to state size mismatch.")
                print(f"The environment state changed. Starting new training run.")
            else:
                raise e
        except FileNotFoundError:
            print(f"Model file not found at {path}. Starting new training run.")
        except Exception as e:
            print(f"An unexpected error occurred during model loading: {e}")