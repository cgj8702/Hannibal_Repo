
import json

def debug_mismatch():
    # 1. Read first request
    req_text = ""
    with open('batch_requests.jsonl', 'r', encoding='utf-8') as f:
        line = f.readline()
        data = json.loads(line)
        req_text = data['request']['contents'][0]['parts'][0]['text']
        print(f"REQUEST TEXT LEN: {len(req_text)}")
        print(f"REQUEST START: {repr(req_text[:50])}")

    # 2. Read first prediction
    pred_text = ""
    with open('results_downloaded/predictions.jsonl', 'r', encoding='utf-8') as f:
        line = f.readline()
        data = json.loads(line)
        
        # Try finding the input in the prediction
        found_text = "NOT FOUND"
        
        # Check standard path
        if 'instance' in data and 'request' in data['instance']:
             found_text = data['instance']['request']['contents'][0]['parts'][0]['text']
        
        # Check direct path
        elif 'request' in data and 'contents' in data['request']:
             found_text = data['request']['contents'][0]['parts'][0]['text']
             
        # Check user's other path
        elif 'instance' in data and 'contents' in data['instance']:
             found_text = data['instance']['contents'][0]['parts'][0]['text']
             
        pred_text = found_text
        print(f"PREDICTION INPUT LEN: {len(pred_text)}")
        print(f"PREDICTION INPUT START: {repr(pred_text[:50])}")

    # 3. Compare
    if req_text == pred_text:
        print("MATCH!")
    else:
        print("MISMATCH!")
        # Find first difference
        for i, (c1, c2) in enumerate(zip(req_text, pred_text)):
            if c1 != c2:
                print(f"Difference at index {i}: Req='{repr(c1)}' vs Pred='{repr(c2)}'")
                break

if __name__ == "__main__":
    debug_mismatch()
