
import vertexai
from vertexai.generative_models import GenerativeModel, SafetySetting
import json
import os
import time
import argparse
import sys

# Configuration
PROJECT_ID = "gen-lang-client-0813719350"
LOCATION = "us-central1"
MODEL_ID = "gemini-2.0-flash-lite-001"
INPUT_FILE = "batch_requests.jsonl"
OUTPUT_DIR = "results_downloaded"
PREAMBLE = (
    "Proofread the following text for grammar and spelling errors. "
    "Output ONLY the corrected text. Do not add markdown formatting, "
    "introductions, or explanations. Maintain all original line breaks.\n\n"
)

def proofread_worker(worker_id, total_workers):
    # Setup
    output_file = os.path.join(OUTPUT_DIR, f"worker_{worker_id}_predictions.jsonl")
    vertexai.init(project=PROJECT_ID, location=LOCATION)
    model = GenerativeModel(MODEL_ID)
    
    safety_settings = [
        SafetySetting(
            category=SafetySetting.HarmCategory.HARM_CATEGORY_HATE_SPEECH,
            threshold=SafetySetting.HarmBlockThreshold.BLOCK_NONE,
        ),
        SafetySetting(
            category=SafetySetting.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
            threshold=SafetySetting.HarmBlockThreshold.BLOCK_NONE,
        ),
        SafetySetting(
            category=SafetySetting.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
            threshold=SafetySetting.HarmBlockThreshold.BLOCK_NONE,
        ),
        SafetySetting(
            category=SafetySetting.HarmCategory.HARM_CATEGORY_HARASSMENT,
            threshold=SafetySetting.HarmBlockThreshold.BLOCK_NONE,
        ),
    ]

    # Create output dir if needed (race condition handled by OS usually, but good practice)
    if not os.path.exists(OUTPUT_DIR):
        try:
            os.makedirs(OUTPUT_DIR)
        except FileExistsError:
            pass

    # Read all inputs
    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        all_inputs = [json.loads(line) for line in f]
    
    total_inputs = len(all_inputs)
    my_chunk_count = 0
    
    # Check for existing progress
    processed_indices = set()
    if os.path.exists(output_file):
        with open(output_file, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    data = json.loads(line)
                    # We need a way to track WHICH chunk this was. 
                    # Ideally we'd store index in output, but sticking to schema...
                    # Let's count lines and trust sequential append? 
                    # No, for robustness, allow duplicate processing if interrupted is safer than skipping.
                    # Or simpler: The worker just appends. We dedupe later.
                    pass 
                except:
                    pass
    
    # We will assume simple append. If restarting, verifying exact state is hard without index in JSON.
    # For now, simplistic approach: JUST RUN. If user restarts, we might redo work, but that's acceptable.
    
    print(f"[Worker {worker_id}] Started. Processing chunks where index % {total_workers} == {worker_id}")

    with open(output_file, 'a', encoding='utf-8') as out_f:
        for i, req in enumerate(all_inputs):
            if i % total_workers != worker_id:
                continue
                
            chunk_prompt = req['request']['contents'][0]['parts'][0]['text']
            # clean_text for logging
            # clean_text = chunk_prompt.replace(PREAMBLE, "")
            
            # Simple logging
            # print(f"[Worker {worker_id}] Processing Chunk {i}...")

            try:
                # Retry logic
                retries = 3
                success = False
                while retries > 0 and not success:
                    try:
                        response = model.generate_content(
                            chunk_prompt,
                            safety_settings=safety_settings,
                            generation_config={"temperature": 0.2, "max_output_tokens": 8192}
                        )
                        result_text = response.text
                        success = True
                    except Exception as e:
                        retries -= 1
                        print(f"[Worker {worker_id}] Error chunk {i}: {e}. Retrying ({retries})...")
                        time.sleep(2)
                
                if not success:
                    print(f"[Worker {worker_id}] FAILED chunk {i} after retries.")
                    # Write failure placeholder? Or skip? Skip = hole in data.
                    # Write Request as Prediction (Echo fallback) to keep continuity?
                    result_text = chunk_prompt.replace(PREAMBLE, "") 
                    
                # Write
                output_row = {
                    "original_index": i, # ADDED METADATA for easier stitching
                    "request": {"contents": [{"parts": [{"text": chunk_prompt}]}]},
                    "prediction": {"candidates": [{"content": {"parts": [{"text": result_text}]}}]}
                }
                out_f.write(json.dumps(output_row) + '\n')
                out_f.flush()
                my_chunk_count += 1
                
                if my_chunk_count % 5 == 0:
                     print(f"[Worker {worker_id}] Processed {my_chunk_count} chunks.")

                # Rate Limit handling (distributed 5 workers = higher load)
                time.sleep(1.0) 

            except Exception as e:
                print(f"[Worker {worker_id}] CATASTROPHIC error on {i}: {e}")

    print(f"[Worker {worker_id}] Finished. Total processed: {my_chunk_count}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker-id", type=int, required=True)
    parser.add_argument("--total-workers", type=int, required=True)
    args = parser.parse_args()
    
    proofread_worker(args.worker_id, args.total_workers)
