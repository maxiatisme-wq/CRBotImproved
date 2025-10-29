import os
import time

def process_image_for_health(image_path, reader):
    try:
        # 1. Run OCR (specifying only digits as allowed characters)
        results = reader.readtext(
            image_path, 
            allowlist='0123456789', 
            detail=0,
            paragraph=False
        )
        
        # 2. Extract and Clean Value
        # EasyOCR returns a list of strings; convert to int.
        # If no digits are found, default to None.
        clean_text = "".join(results)
        if clean_text:
            value = int(clean_text)
        else:
            value = None
        
        # 3. Determine the tower name from the path
        name = os.path.basename(image_path).replace(".png", "")
        
        return (name, value)
        
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return (os.path.basename(image_path), None) # Return None on failure

def get_tower_health_values(reader):
    # Define where the images are saved.
    BASE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'ocr_captures') 
    
    # Defines the names of the files that hold the tower health.
    TOWER_ORDER = {
        "l_enemyprincess": 0,
        "enemy_king": 1,
        "r_enemyprincess": 2,
        "l_allyprincess": 3,
        "ally_king": 4,
        "r_allyprincess": 5,
    }
    
    # Creates the list of full file paths to process
    image_paths = [
        os.path.join(BASE_DIR, name + ".png")
        for name in TOWER_ORDER.keys()
    ]
    
    results_tuples = []
    for path in image_paths:
        # Run sequentially
        result = process_image_for_health(path, reader) 
        results_tuples.append(result)
        
    # final output list
    final_output = [None] * len(TOWER_ORDER)
    
    # 2. Populate the final list based on the predefined order
    for name, value in results_tuples:
        if name in TOWER_ORDER:
            index = TOWER_ORDER[name]
            final_output[index] = value
            
    return final_output
