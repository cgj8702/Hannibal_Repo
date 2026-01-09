import json
import glob
import os

files = glob.glob(os.path.join("HANNIBAL_CHUNKS_OUTPUT", "*_memory.json"))
files.sort()

print(f"Checking {len(files)} files...")
total = 0
for f in files:
    try:
        with open(f, "r", encoding="utf-8") as file:
            data = json.load(file)
            print(f"{os.path.basename(f)}: {len(data)} scenes")
            total += len(data)
    except Exception as e:
        print(f"{os.path.basename(f)}: ERROR {e}")

print(f"SUM OF ALL FILES: {total}")
