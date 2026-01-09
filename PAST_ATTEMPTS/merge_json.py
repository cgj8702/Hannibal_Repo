import json
import glob
import os
from tqdm import tqdm

INPUT_FOLDER = os.path.join(os.getcwd(), "HANNIBAL_CHUNKS_OUTPUT")
OUTPUT_FILE = os.path.join(os.getcwd(), "Hannibal_Complete_Memories.json")

def merge_files():
    # Get all json files
    files = glob.glob(os.path.join(INPUT_FOLDER, "*_memory.json"))
    
    # Sort them ensuring 1x01 comes before 1x02 etc.
    # The naming convention "Hannibal_1x01_Aperitif" sorts correctly alphabetically.
    files.sort()
    
    master_list = []
    total_scenes = 0
    
    print(f"Found {len(files)} episode files. Merging...")
    
    for f_path in tqdm(files):
        try:
            with open(f_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, list):
                    master_list.extend(data)
                    total_scenes += len(data)
                else:
                    print(f"Warning: {os.path.basename(f_path)} is not a list. Skipping.")
        except Exception as e:
            print(f"Error reading {os.path.basename(f_path)}: {e}")
            
    print(f"\nMerge complete.")
    print(f"Total Episodes: {len(files)}")
    print(f"Total Scenes: {total_scenes}")
    
    # Write to single file
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(master_list, f, indent=4)
        
    print(f"Saved to: {OUTPUT_FILE}")

if __name__ == "__main__":
    merge_files()
