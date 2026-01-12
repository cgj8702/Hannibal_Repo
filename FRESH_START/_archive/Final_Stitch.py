
import json
import glob
import os

# --- CONFIGURATION ---
ORIGINAL_BATCH_FILE = "batch_requests.jsonl" 
RESULTS_FOLDER = "results_downloaded"       
FINAL_OUTPUT_FILE = "Hannibal_Proofread_Complete.txt"

def stitch_results():
    print("Loading original requests to establish order...")
    original_prompts = []
    
    # 1. Read the original order of chunks
    with open(ORIGINAL_BATCH_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            req = json.loads(line)
            # Extract the text chunk we sent
            prompt_text = req['request']['contents'][0]['parts'][0]['text']
            original_prompts.append(prompt_text)

    print(f"Expect to stitch {len(original_prompts)} chunks.")

    # 2. Load all prediction results into a lookup map
    # Maps: Original_Chunk_Text -> Corrected_Output_Text
    correction_map = {}
    
    result_files = glob.glob(os.path.join(RESULTS_FOLDER, "*.jsonl"))
    print(f"Found {len(result_files)} result files. Processing...")

    # Helper for robust matching
    def normalize_key(text):
        return "".join(text.split())

    for res_file in result_files:
        with open(res_file, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    data = json.loads(line)
                    
                    # The output JSONL contains the input 'instance' (request) and the 'prediction'
                    # We use the input text to identify which chunk this is
                    # Vertex Batch sometimes nests the request in 'instance' -> 'request'
                    # OR just 'instance' if it was sent that way.
                    # Based on my previous code, I assumed data['instance']['request']...
                    
                    input_text = ""
                    # Check for direct 'request' key (common in some batch outputs)
                    if 'request' in data and 'contents' in data['request']:
                        input_text = data['request']['contents'][0]['parts'][0]['text']
                    # Check for 'instance' wrapper
                    elif 'instance' in data: 
                        if 'request' in data['instance']:
                             input_text = data['instance']['request']['contents'][0]['parts'][0]['text']
                        elif 'contents' in data['instance']:
                             input_text = data['instance']['contents'][0]['parts'][0]['text']
                    
                    # Extract the model's output
                    response_text = input_text # Default to original
                    
                    if 'prediction' in data:
                        candidates = data['prediction'].get('candidates', [])
                        if candidates:
                            response_text = candidates[0]['content']['parts'][0]['text']
                    else:
                        print(f"Skipping a line with no prediction: {data.get('status', 'Unknown')}")
                    
                    if input_text:
                        # Use normalized key
                        correction_map[normalize_key(input_text)] = response_text
                    
                except Exception as e:
                    print(f"Error parsing line in {res_file}: {e}")

    # 3. Reconstruct in order
    print("Reconstructing file...")
    missing_chunks = 0
    
    with open(FINAL_OUTPUT_FILE, 'w', encoding='utf-8') as outfile:
        for i, original_chunk in enumerate(original_prompts):
            # Normalize lookup key
            key = normalize_key(original_chunk)
            
            if key in correction_map:
                outfile.write(correction_map[key])
            else:
                print(f"Warning: Chunk {i+1} was missing from results. Using original text.")
                outfile.write(original_chunk) # Fallback
                missing_chunks += 1
                
            # Add a newline between chunks if needed, though original text usually has them?
            # The prompt had "Maintain all original line breaks."
            # The chunker just did text[i:i+CHUNK_SIZE].
            # So simply concatenating them is correct. 

    print("------------------------------------------------")
    if missing_chunks == 0:
        print(f"SUCCESS! Full file reconstructed at: {FINAL_OUTPUT_FILE}")
    else:
        print(f"Done, but {missing_chunks} chunks were missing/failed and reverted to original.")

if __name__ == "__main__":
    stitch_results()
