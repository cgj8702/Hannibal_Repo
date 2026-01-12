
import json
import glob
import os

OUTPUT_DIR = "results_downloaded"
FINAL_OUTPUT_FILE = "Hannibal_Proofread_Complete.txt"
PREAMBLE = (
    "Proofread the following text for grammar and spelling errors. "
    "Output ONLY the corrected text. Do not add markdown formatting, "
    "introductions, or explanations. Maintain all original line breaks.\n\n"
)

def parallel_stitch():
    print("--- Stitching Parallel Worker Outputs ---")
    
    # 1. Gather all segments
    segments = []
    worker_files = glob.glob(os.path.join(OUTPUT_DIR, "worker_*_predictions.jsonl"))
    print(f"Found {len(worker_files)} worker files.")
    
    for w_file in worker_files:
        print(f"Reading {w_file}...")
        with open(w_file, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    data = json.loads(line)
                    
                    # Extract Index
                    idx = data.get('original_index')
                    if idx is None:
                        # Fallback if I messed up and some lines don't have it (shouldn't happen)
                        print("Warning: valid JSON line missing 'original_index'. Skipping.")
                        continue
                        
                    # Extract Content (Prediction or Fallback)
                    # Note: My worker script writes 'prediction' even on failure (fallback echo),
                    # but let's be robust.
                    text = ""
                    if 'prediction' in data and 'candidates' in data['prediction']:
                        text = data['prediction']['candidates'][0]['content']['parts'][0]['text']
                    else:
                        print(f"Warning: Chunk {idx} has no prediction. Using request/echo.")
                        text = data['request']['contents'][0]['parts'][0]['text'].replace(PREAMBLE, "")
                        
                    segments.append((idx, text))
                    
                except Exception as e:
                    print(f"Error parsing line in {w_file}: {e}")

    # 2. Sort by Index
    print(f"Total segments collected: {len(segments)}")
    segments.sort(key=lambda x: x[0])
    
    # 3. Check for Gaps
    if not segments:
        print("Error: No segments found!")
        return

    max_idx = segments[-1][0]
    expected_count = max_idx + 1
    
    # Create valid map
    valid_indices = {s[0] for s in segments}
    missing = [i for i in range(expected_count) if i not in valid_indices]
    
    if missing:
        print(f"WARNING: Missing {len(missing)} chunks! Indices: {missing[:10]}...")
        # We should probably fill gaps with original text if we want a complete file.
        # But this script only reads outputs. 
        # Ideally, we read 'batch_requests.jsonl' to fill gaps.
        # Let's add that robustness.
        fill_gaps = True
    else:
        print("Perfect! No missing chunks.")
        fill_gaps = False

    # 4. Fill Gaps (Robustness)
    final_ordered_text = []
    if fill_gaps:
        print("Attempting to fill gaps from source...")
        try:
            with open("batch_requests.jsonl", 'r', encoding='utf-8') as f:
                all_originals = [json.loads(line)['request']['contents'][0]['parts'][0]['text'].replace(PREAMBLE, "") for line in f]
            
            for i in range(len(all_originals)):
                # Find in results
                found = next((text for idx, text in segments if idx == i), None)
                if found:
                    final_ordered_text.append(found)
                else:
                    print(f"Filling gap {i} with original text.")
                    final_ordered_text.append(all_originals[i])
                    
        except Exception as e:
            print(f"Could not read source for filling gaps: {e}")
            # Fallback: Just join what we have? No, that shifts text. 
            print("Aborting fill. Result will be fragmentary.")
            return
            
    else:
        final_ordered_text = [s[1] for s in segments]

    # 5. Write
    print(f"Writing to {FINAL_OUTPUT_FILE}...")
    with open(FINAL_OUTPUT_FILE, 'w', encoding='utf-8') as f:
        f.write("".join(final_ordered_text))
        
    print("Stitching Complete.")

if __name__ == "__main__":
    parallel_stitch()
