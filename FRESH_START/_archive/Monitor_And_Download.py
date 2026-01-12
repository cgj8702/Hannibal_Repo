
import time
import os
import glob
from google.cloud import aiplatform
from google.cloud import storage
import subprocess
import shutil

# --- CONFIGURATION ---
PROJECT_ID = "gen-lang-client-0813719350"
LOCATION = "us-central1"
BUCKET_NAME = "hannibal-screenplays"
OUTPUT_PREFIX = "results/" # The prefix usage in Submit_Batch_Request was "results/"
LOCAL_DOWNLOAD_DIR = "results_downloaded"
RESTITCHER_SCRIPT = "Chunked_Output_Restitcher.py"

def get_latest_job():
    aiplatform.init(project=PROJECT_ID, location=LOCATION)
    jobs = aiplatform.BatchPredictionJob.list(order_by="create_time desc")
    if not jobs:
        return None
    return jobs[0]

def download_blob_pattern(bucket_name, prefix, local_dir):
    """Downloads files matching the prefix from GCS."""
    if not os.path.exists(local_dir):
        os.makedirs(local_dir)
        
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blobs = bucket.list_blobs(prefix=prefix)
    
    count = 0
    for blob in blobs:
        if blob.name.endswith("/"): 
            continue # Skip directories
            
        # We only care about .jsonl files (predictions) or errors
        if not blob.name.endswith(".jsonl"):
            continue

        filename = os.path.basename(blob.name)
        local_path = os.path.join(local_dir, filename)
        
        print(f"Downloading {blob.name} to {local_path}...")
        blob.download_to_filename(local_path)
        count += 1
        
    return count

def monitor_and_download():
    print(f"Checking for latest batch job in project {PROJECT_ID}...")
    job = get_latest_job()
    
    if not job:
        print("No batch jobs found.")
        return

    print(f"Monitoring Job: {job.display_name} ({job.resource_name})")
    print(f"Initial State: {job.state.name}")

    while True:
        # Refresh job state
        # The client object might not auto-refresh, so we re-fetch or use checking methods
        # The job object from .list() is a snapshot. We can call job.state but better to reload?
        # Actually job.state is a property, might need re-fetching execution info.
        # Safest is to get the job again by ID or name, but aiplatform.BatchPredictionJob.get(...) works.
        
        # Re-fetch the job using the constructor which gets the latest state by resource name
        current_job = aiplatform.BatchPredictionJob(job.resource_name)
        state = current_job.state
        
        print(f"Current State: {state.name}", end="\r")
        
        if state.name == "JOB_STATE_SUCCEEDED":
            print(f"\nJob SUCCEEDED! Starting download...")
            
            # Download
            count = download_blob_pattern(BUCKET_NAME, OUTPUT_PREFIX, LOCAL_DOWNLOAD_DIR)
            print(f"Downloaded {count} files.")
            
            if count > 0:
                print("Triggering Restitcher...")
                subprocess.run(["python", RESTITCHER_SCRIPT])
            else:
                print("Warning: No result files found in GCS bucket.")
            
            break
            
        elif state.name in ["JOB_STATE_FAILED", "JOB_STATE_CANCELLED", "JOB_STATE_EXPIRED"]:
            print(f"\nJob FAILED/CANCELLED. State: {state.name}")
            if current_job.error:
                print(f"Error: {current_job.error}")
            break
            
        # Wait
        time.sleep(60)

if __name__ == "__main__":
    monitor_and_download()
