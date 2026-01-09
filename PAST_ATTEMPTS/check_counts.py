import json
import glob
import os

try:
    files = glob.glob(os.path.join("HANNIBAL_CHUNKS_OUTPUT", "*_memory.json"))
    print(f"Total JSON Files Found: {len(files)}")
    
    memories_path = "Hannibal_Complete_Memories.json"
    if os.path.exists(memories_path):
        with open(memories_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            print(f"Total Scenes in Merged File: {len(data)}")
    else:
        print("Hannibal_Complete_Memories.json NOT FOUND.")
except Exception as e:
    print(f"Error: {e}")
