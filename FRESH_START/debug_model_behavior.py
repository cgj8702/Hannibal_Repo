
import json

def analyze_edits():
    diff_count = 0
    total_count = 0
    echo_count = 0
    
    with open('results_downloaded/predictions.jsonl', 'r', encoding='utf-8') as f:
        for line in f:
            total_count += 1
            data = json.loads(line)
            
            # Extract Input
            input_text = ""
            if 'request' in data and 'contents' in data['request']:
                input_text = data['request']['contents'][0]['parts'][0]['text']
            elif 'instance' in data:
                if 'request' in data['instance']:
                    input_text = data['instance']['request']['contents'][0]['parts'][0]['text']
                elif 'contents' in data['instance']:
                    input_text = data['instance']['contents'][0]['parts'][0]['text']
            
            # Extract Output
            output_text = input_text # Default
            if 'prediction' in data:
                 candidates = data['prediction'].get('candidates', [])
                 if candidates:
                    output_text = candidates[0]['content']['parts'][0]['text']
            
            # Normalize for comparison (ignore whitespace changes which might just be formatting)
            norm_input = "".join(input_text.split())
            norm_output = "".join(output_text.split())
            
            if norm_input == norm_output:
                echo_count += 1
            else:
                diff_count += 1
                # Print first diff for sanity check
                if diff_count == 1:
                    print("--- First Content Change Detected ---")
                    print(f"Input len: {len(input_text)}, Output len: {len(output_text)}")
                    print(f"Sample Input: {repr(input_text[:100])}...")
                    print(f"Sample Output: {repr(output_text[:100])}...")
    
    print("-" * 30)
    print(f"Total Chunks: {total_count}")
    print(f"Chunks with Actual Text Changes: {diff_count}")
    print(f"Chunks that are Identical (Echoes): {echo_count}")
    print(f"Echo Rate: {echo_count/total_count*100:.1f}%")

if __name__ == "__main__":
    analyze_edits()
