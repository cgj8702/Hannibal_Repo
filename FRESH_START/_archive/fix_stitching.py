
import json
import os
import glob

BATCH_JSONL_FILE = "batch_requests.jsonl"
LOCAL_RESULTS_DIR = "results_downloaded"
FINAL_OUTPUT_FILE = "Hannibal_Proofread_Complete.txt"

PREAMBLE = (
    "Proofread the following text for grammar and spelling errors. "
    "Output ONLY the corrected text. Do not add markdown formatting, "
    "introductions, or explanations. Maintain all original line breaks.\n\n"
)

def normalize_key(text):
    return "".join(text.split())

def fix_stitching():
    print("--- Restitching with Preamble Removal ---")
    
    # 1. Load Original Texts (Clean)
    original_texts = []
    print(f"Reading {BATCH_JSONL_FILE}...")
    with open(BATCH_JSONL_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            req = json.loads(line)
            full_prompt = req['request']['contents'][0]['parts'][0]['text']
            
            # Strip Preamble to get just the screenplay text
            if full_prompt.startswith(PREAMBLE):
                clean_text = full_prompt.replace(PREAMBLE, "", 1)
            else:
                # Fallback if logic changes, though it shouldn't
                clean_text = full_prompt
                
            original_texts.append(clean_text)
            
    print(f"Loaded {len(original_texts)} original chunks.")

    # 2. Build Map from Results
    correction_map = {}
    result_files = glob.glob(os.path.join(LOCAL_RESULTS_DIR, "*.jsonl"))
    print(f"Found {len(result_files)} result files.")
    
    for res_file in result_files:
        with open(res_file, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    data = json.loads(line)
                    
                    # Extract Input (Request)
                    input_text = ""
                    if 'request' in data and 'contents' in data['request']:
                        input_text = data['request']['contents'][0]['parts'][0]['text']
                    elif 'instance' in data: # Handle potential instance format
                         if 'request' in data['instance']:
                              input_text = data['instance']['request']['contents'][0]['parts'][0]['text']
                         elif 'contents' in data['instance']:
                              input_text = data['instance']['contents'][0]['parts'][0]['text']
                    
                    # Strip Preamble from Input to use as Key
                    if input_text.startswith(PREAMBLE):
                        clean_input = input_text.replace(PREAMBLE, "", 1)
                    else:
                        clean_input = input_text

                    # Extract Output (Prediction)
                    output_text = ""
                    if 'prediction' in data and 'candidates' in data['prediction']:
                         output_text = data['prediction']['candidates'][0]['content']['parts'][0]['text']
                    else:
                        # Fallback to input if no prediction (echo behavior)
                        output_text = clean_input 

                    if clean_input:
                        correction_map[normalize_key(clean_input)] = output_text
                        
                except Exception as e:
                    print(f"Skipping line due to error: {e}")

    # 3. Stitch
    print("Stitching...")
    with open(FINAL_OUTPUT_FILE, 'w', encoding='utf-8') as outfile:
        missing_count = 0
        match_count = 0
        
        for chunk in original_texts:
            key = normalize_key(chunk)
            
            if key in correction_map:
                # Found proofread version
                outfile.write(correction_map[key])
                match_count += 1
            else:
                # Missing: Write Original Clean Text (NOT Prompt)
                outfile.write(chunk)
                missing_count += 1
                
    print(f"Done. Matches: {match_count}, Missing/Fallback: {missing_count}")
    print(f"Final File: {FINAL_OUTPUT_FILE}")

if __name__ == "__main__":
    fix_stitching()
