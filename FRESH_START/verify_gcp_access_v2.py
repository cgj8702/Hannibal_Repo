
from google.cloud import storage
from google.cloud import aiplatform
import google.auth
import sys

LOG_FILE = "gcp_verify_log.txt"

def log(message):
    print(message)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(message + "\n")

def verify_access():
    with open(LOG_FILE, "w", encoding="utf-8") as f:
        f.write("--- Verifying Google Cloud Access ---\n")

    # 1. Credentials & Project
    log("Checking Credentials...")
    try:
        credentials, project_id = google.auth.default()
        log(f"Credentials found: {credentials is not None}")
        log(f"Inferred Project ID: {project_id}")
    except Exception as e:
        log(f"Auth credential check failed: {e}")
        return

    # 2. Storage Access
    log("\n--- Testing Storage Access ---")
    try:
        storage_client = storage.Client(credentials=credentials, project=project_id)
        # Just getting the bucket we know exists
        bucket_name = "hannibal-screenplays"
        bucket = storage_client.bucket(bucket_name)
        if bucket.exists():
             log(f"Success: Bucket '{bucket_name}' is accessible.")
        else:
             log(f"Warning: Bucket '{bucket_name}' not found, but access seems okay.")
    except Exception as e:
        log(f"Storage access failed: {e}")

    # 3. Vertex AI Access
    log("\n--- Testing Vertex AI Access ---")
    try:
        aiplatform.init(project=project_id, location="us-central1")
        # Try to list jobs to verify API access
        jobs = aiplatform.BatchPredictionJob.list(limit=1)
        log("Success: Connected to Vertex AI.")
        log(f"Found {len(jobs)} existing jobs.")
    except Exception as e:
        log(f"Vertex AI access failed: {e}")

if __name__ == "__main__":
    verify_access()
