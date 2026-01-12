
import vertexai
from vertexai.generative_models import GenerativeModel, SafetySetting
import json
import os
import time

PROJECT_ID = "gen-lang-client-0813719350"
LOCATION = "us-central1"
MODEL_ID = "gemini-2.0-flash-lite-001"
INPUT_FILE = "batch_requests.jsonl"
OUTPUT_DIR = "results_downloaded"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "serial_predictions.jsonl")

# Preamble to strip for cleaner logging/results if needed
PREAMBLE = (
    "Proofread the following text for grammar and spelling errors. "
    "Output ONLY the corrected text. Do not add markdown formatting, "
    "introductions, or explanations. Maintain all original line breaks.\n\n"
)

def serial_proofread():
    # 1. Setup
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

    # 2. Check Resume Status
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        
    start_index = 0
    if os.path.exists(OUTPUT_FILE):
        with open(OUTPUT_FILE, 'r', encoding='utf-8') as f:
            start_index = sum(1 for _ in f)
        print(f"Resuming from chunk {start_index}...")
    
    # 3. Read Inputs
    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        inputs = [json.loads(line) for line in f]
    
    total = len(inputs)
    print(f"Total Chunks to Process: {total}")
    
    # 4. Processing Loop
    with open(OUTPUT_FILE, 'a', encoding='utf-8') as out_f:
        for i, req in enumerate(inputs):
            if i < start_index:
                continue
                
            chunk_prompt = req['request']['contents'][0]['parts'][0]['text']
            # Reconstruct clean text for logging/verification
            clean_text = chunk_prompt.replace(PREAMBLE, "")
            
            print(f"[{i+1}/{total}] Processing chunk ({len(clean_text)} chars)... ", end="")
            
            try:
                response = model.generate_content(
                    chunk_prompt,
                    safety_settings=safety_settings,
                    generation_config={"temperature": 0.2, "max_output_tokens": 8192}
                )
                
                result_text = response.text
                
                # Basic validation
                if result_text == clean_text:
                    status = "ECHOED"
                elif " " not in result_text and len(result_text) < 100:
                     status = "SUSPICIOUS_SHORT"
                else:
                    status = "EDITED"
                
                print(f"{status}")
                
                # Write in compatible JSONL format for stitching
                output_row = {
                    "request": {
                        "contents": [{"parts": [{"text": chunk_prompt}]}]
                    },
                    "prediction": {
                        "candidates": [{"content": {"parts": [{"text": result_text}]}}]
                    }
                }
                out_f.write(json.dumps(output_row) + '\n')
                out_f.flush() # Ensure save
                
                # Rate Limit Throttling (conservative)
                time.sleep(1.0) 
                
            except Exception as e:
                print(f"ERROR: {e}")
                # Save invalid row to keep alignment or retry? 
                # Better to crash/stop for manual resume than skip silently
                time.sleep(10) # Backoff
                
    print("Serial Proofreading Complete!")

if __name__ == "__main__":
    serial_proofread()
