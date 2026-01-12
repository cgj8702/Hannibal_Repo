
import json
import os
import glob
import time
from google.cloud import aiplatform
from google.cloud import storage
import subprocess
import shutil

# --- CONFIGURATION ---
PROJECT_ID = "gen-lang-client-0813719350"
LOCATION = "us-central1"
BUCKET_NAME = "hannibal-screenplays"
INPUT_TEXT_FILE = "Hannibal_All_Screenplays.txt"
BATCH_JSONL_FILE = "batch_requests.jsonl"
OUTPUT_PREFIX = "results/" 
LOCAL_RESULTS_DIR = "results_downloaded"
FINAL_OUTPUT_FILE = "Hannibal_Proofread_Complete.txt"
CHUNK_SIZE = 4000 # Reduced to 4000 for quality + safety buffer
MODEL_VERSION = "gemini-2.0-flash-lite-001"

# --- PART 1: CHUNKER ---
def create_batch_file():
    print("--- 1. Chunking Screenplay ---")
    if not os.path.exists(INPUT_TEXT_FILE):
        print(f"Error: {INPUT_TEXT_FILE} not found.")
        return False

    with open(INPUT_TEXT_FILE, 'r', encoding='utf-8') as f:
        text = f.read()

    print(f"Total file size: {len(text)} characters.")
    
    with open(BATCH_JSONL_FILE, 'w', encoding='utf-8') as out_f:
        for i in range(0, len(text), CHUNK_SIZE):
            chunk = text[i:i + CHUNK_SIZE]
            prompt = (
                "Proofread the following text for grammar and spelling errors. "
                "Output ONLY the corrected text. Do not add markdown formatting, "
                "introductions, or explanations. Maintain all original line breaks.\n\n"
                f"{chunk}"
            )
            request_row = {
                "request": {
                    "contents": [{"role": "user", "parts": [{"text": prompt}]}],
                    "generation_config": {"temperature": 0.2, "max_output_tokens": 8192},
                    "safety_settings": [
                        {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
                        {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
                        {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
                        {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"}
                    ]
                }
            }
            out_f.write(json.dumps(request_row) + '\n')
            
    print(f"Created {BATCH_JSONL_FILE} with {len(text)//CHUNK_SIZE} chunks.")
    return True

# --- PART 2: UPLOAD (Implicit Requirement) ---
def upload_file_to_gcs():
    print("--- 2. Uploading to GCS ---")
    storage_client = storage.Client()
    bucket = storage_client.bucket(BUCKET_NAME)
    blob = bucket.blob(BATCH_JSONL_FILE) # Same name in bucket

    print(f"Uploading {BATCH_JSONL_FILE}...")
    blob.upload_from_filename(BATCH_JSONL_FILE)
    print(f"Uploaded to gs://{BUCKET_NAME}/{BATCH_JSONL_FILE}")
    return True

# --- PART 3: SUBMIT ---
def submit_batch_job():
    print("--- 3. Submitting Batch Job ---")
    aiplatform.init(project=PROJECT_ID, location=LOCATION)
    
    input_uri = f"gs://{BUCKET_NAME}/{BATCH_JSONL_FILE}"
    output_uri_prefix = f"gs://{BUCKET_NAME}/{OUTPUT_PREFIX}"
    
    job = aiplatform.BatchPredictionJob.create(
        job_display_name=f"hannibal-proofread-{MODEL_VERSION}",
        model_name=f"publishers/google/models/{MODEL_VERSION}",
        instances_format="jsonl",
        predictions_format="jsonl",
        gcs_source=input_uri,
        gcs_destination_prefix=output_uri_prefix,
    )
    print(f"Job submitted: {job.resource_name}")
    return job

# --- PART 4: MONITOR & DOWNLOAD ---
def monitor_and_download(initial_job):
    print("--- 4. Monitoring Job ---")
    
    while True:
        # Re-fetch job to get status
        job = aiplatform.BatchPredictionJob(initial_job.resource_name)
        state = job.state.name
        
        print(f"Job State: {state}...", end="\r")
        
        if state == "JOB_STATE_SUCCEEDED":
            print(f"\nJob SUCCEEDED!")
            return True
        elif state in ["JOB_STATE_FAILED", "JOB_STATE_CANCELLED"]:
            print(f"\nJob FAILED: {job.error}")
            return False
            
        time.sleep(60)

def download_results():
    print("--- 5. Downloading Results ---")
    if not os.path.exists(LOCAL_RESULTS_DIR):
        os.makedirs(LOCAL_RESULTS_DIR)

    storage_client = storage.Client()
    bucket = storage_client.bucket(BUCKET_NAME)
    blobs = bucket.list_blobs(prefix=OUTPUT_PREFIX)
    
    count = 0
    for blob in blobs:
        if blob.name.endswith(".jsonl"):
            local_path = os.path.join(LOCAL_RESULTS_DIR, os.path.basename(blob.name))
            blob.download_to_filename(local_path)
            count += 1
            
    print(f"Downloaded {count} result files.")
    return count > 0

# --- PART 5: RESTITCH ---
def stitch_results():
    print("--- 6. Restitching Output ---")
    
    # Reload original prompts for ordering
    original_prompts = []
    with open(BATCH_JSONL_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            req = json.loads(line)
            original_prompts.append(req['request']['contents'][0]['parts'][0]['text'])

    # Map inputs to outputs
    correction_map = {}
    result_files = glob.glob(os.path.join(LOCAL_RESULTS_DIR, "*.jsonl"))
    
    # Helper for robust matching
    def normalize_key(text):
        return "".join(text.split())

    for res_file in result_files:
        with open(res_file, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    data = json.loads(line)
                    # Extract input and output
                    input_text = ""
                    if 'request' in data and 'contents' in data['request']:
                        input_text = data['request']['contents'][0]['parts'][0]['text']
                    elif 'instance' in data:
                        if 'request' in data['instance']:
                             input_text = data['instance']['request']['contents'][0]['parts'][0]['text']
                        elif 'contents' in data['instance']:
                             input_text = data['instance']['contents'][0]['parts'][0]['text']

                    if 'prediction' in data:
                        resp = data['prediction']['candidates'][0]['content']['parts'][0]['text']
                    else:
                        resp = input_text # Fallback
                        
                    if input_text:
                        correction_map[normalize_key(input_text)] = resp
                except:
                    pass

    # Reconstruct
    with open(FINAL_OUTPUT_FILE, 'w', encoding='utf-8') as outfile:
        missing = 0
        for chunk in original_prompts:
            key = normalize_key(chunk)
            if key in correction_map:
                outfile.write(correction_map[key])
            else:
                outfile.write(chunk)
                missing += 1
    
    print(f"Done! Final file: {FINAL_OUTPUT_FILE}")
    if missing > 0:
        print(f"Warning: {missing} chunks were missing and reverted to original.")

# --- MASTER FLOW ---
if __name__ == "__main__":
    if create_batch_file():
        if upload_file_to_gcs():
            job = submit_batch_job()
            if monitor_and_download(job):
                if download_results():
                    stitch_results()
