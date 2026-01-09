import json
import glob
import os
import re

files = [os.path.join("HANNIBAL_CHUNKS_OUTPUT", "Hannibal_1x01_Aperitif_memory.json")]

for f in files:
    try:
        with open(f, "r", encoding="utf-8") as file:
            data = json.load(file)
            scenes_with_numbers = []
            for scene in data:
                heading = scene.get("heading", "")
                match = re.search(r"^([A-Z]*\d+[A-Z]*)\s+", heading)
                if match:
                    digits = re.findall(r"\d+", match.group(1))
                    if digits:
                        num = int(digits[0])
                        scenes_with_numbers.append(num)
            
            scenes_with_numbers.sort()
            print(f"File: {os.path.basename(f)}")
            print(f"Found {len(scenes_with_numbers)} numbered scenes. Max number: {scenes_with_numbers[-1]}")
            
            prev = scenes_with_numbers[0]
            print(f"Starting at Scene {prev}")
            for curr in scenes_with_numbers[1:]:
                if curr > prev + 1:
                    print(f"MISSING: Scenes {prev + 1} to {curr - 1} (Gap of {curr - prev - 1})")
                prev = curr

    except Exception as e:
        print(f"Error {f}: {e}")
