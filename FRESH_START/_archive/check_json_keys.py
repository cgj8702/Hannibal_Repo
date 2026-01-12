
import json

with open("output_v2/predictions.jsonl", "r", encoding="utf-8") as f:
    line = f.readline()
    data = json.loads(line)
    print("Keys found:", list(data.keys()))
    
    if "response" in data:
        print("Response keys:", list(data["response"].keys()))
    if "prediction" in data:
        print("Prediction keys:", list(data["prediction"].keys()))
