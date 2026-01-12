
import json

with open('results_downloaded/predictions.jsonl', 'r') as f:
    line = f.readline()
    data = json.loads(line)
    print("Top keys:", list(data.keys()))
    if 'instance' in data:
        print("Instance keys:", list(data['instance'].keys()))
        if 'request' in data['instance']:
             print("Instance->Request keys:", list(data['instance']['request'].keys()))
    elif 'request' in data:
        print("Request keys:", list(data['request'].keys()))
