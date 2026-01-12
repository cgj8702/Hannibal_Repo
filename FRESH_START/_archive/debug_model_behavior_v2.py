
import json
import glob
import os

def analyze_edits():
    diff_count = 0
    total_count = 0
    echo_count = 0
    
    files = glob.glob(os.path.join('results_downloaded', '*.jsonl'))
    print(f"Analyzing {len(files)} files: {[os.path.basename(f) for f in files]}")

    for file_path in files:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                total_count += 1
                try:
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
                    
                    # Normalize for comparison
                    norm_input = "".join(input_text.split())
                    norm_output = "".join(output_text.split())
                    
                    if norm_input == norm_output:
                        echo_count += 1
                    else:
                        diff_count += 1
                        # Print first few diffs
                        if diff_count <= 3:
                            print(f"\n--- Content Change Detected ({diff_count}) ---")
                            print(f"File: {os.path.basename(file_path)}")
                            # Simplistic diff preview
                            print(f"Input Len: {len(input_text)} -> Output Len: {len(output_text)}")
                
                except Exception as e:
                    print(f"Error parsing line in {file_path}: {e}")
    
    print("-" * 30)
    print(f"Total Chunks: {total_count}")
    print(f"Chunks with Actual Text Changes: {diff_count}")
    print(f"Chunks that are Identical (Echoes): {echo_count}")
    if total_count > 0:
        print(f"Echo Rate: {echo_count/total_count*100:.1f}%")
        print(f"Edit Rate: {diff_count/total_count*100:.1f}%")

if __name__ == "__main__":
    analyze_edits()
