
import vertexai
from vertexai.generative_models import GenerativeModel, SafetySetting
import json
import os
import time

PROJECT_ID = "gen-lang-client-0813719350"
LOCATION = "us-central1"
# switching to a Pro model if possible for quality, or Flash for speed/context
# Search suggested gemini-2.5-pro might have 64k output. 
# But let's use gemini-1.5-pro-002 or gemini-2.0-flash-exp which definitely exist in my known timeline.
# Actually, let's stick to gemini-2.0-flash-lite-001 as we verified it works, but INCREASE chunks.
# Wait, flash-lite has 8k limit. 
# Search said 1.5 Pro has 8k-32k. 
# Using gemini-2.0-flash-lite-001 with optimized chunk size
MODEL_ID = "gemini-2.0-flash-lite-001" 
CHUNK_SIZE = 20000  # 5k tokens output < 8k limit
INPUT_TEXT_FILE = "Hannibal_All_Screenplays.txt"
OUTPUT_FILE = "results_downloaded/fast_track_predictions.jsonl"

def fast_track_proofread():
    vertexai.init(project=PROJECT_ID, location=LOCATION)
    # Trying 1.5 Pro for higher output limit and quality
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

    # 1. Chunking
    print(f"--- Chunking {INPUT_TEXT_FILE} to {CHUNK_SIZE} chars ---")
    if not os.path.exists(INPUT_TEXT_FILE):
        print("Input file not found!")
        return

    with open(INPUT_TEXT_FILE, 'r', encoding='utf-8') as f:
        text = f.read()
    
    chunks = [text[i:i+CHUNK_SIZE] for i in range(0, len(text), CHUNK_SIZE)]
    total_chunks = len(chunks)
    print(f"Total Chunks: {total_chunks}")

    # 2. Resume Logic
    start_index = 0
    if os.path.exists(OUTPUT_FILE):
        with open(OUTPUT_FILE, 'r', encoding='utf-8') as f:
            start_index = sum(1 for _ in f)
        print(f"Resuming from chunk {start_index}...")

    # 3. Processing
    with open(OUTPUT_FILE, 'a', encoding='utf-8') as out_f:
        for i, chunk in enumerate(chunks):
            if i < start_index:
                continue

            print(f"[{i+1}/{total_chunks}] Processing... ", end="")
            
            prompt = (
                "Proofread the following text for grammar and spelling errors. "
                "Output ONLY the corrected text. Do not add markdown formatting, "
                "introductions, or explanations. Maintain all original line breaks.\n\n"
                f"{chunk}"
            )
            
            try:
                response = model.generate_content(
                    prompt,
                    safety_settings=safety_settings,
                    generation_config={"temperature": 0.2, "max_output_tokens": 8192} 
                )
                
                # If output is truncated, verify.
                result_text = response.text
                
                # Check ratio
                ratio = len(result_text) / len(chunk)
                status = "OK"
                if ratio < 0.8: status = "TRUNCATED?"
                if ratio > 1.2: status = "HALLUCINATED?"
                if result_text == chunk: status = "ECHOED"
                
                print(f"{status} (Len: {len(chunk)} -> {len(result_text)})")
                
                # Write
                output_row = {
                    "input": chunk, # Storing raw input for easier stitching logic later if needed
                    "prediction": result_text
                }
                out_f.write(json.dumps(output_row) + '\n')
                out_f.flush()
                
                time.sleep(2) # Modest rate limit for Pro model
                
            except Exception as e:
                print(f"ERROR: {e}")
                time.sleep(10)

if __name__ == "__main__":
    fast_track_proofread()
