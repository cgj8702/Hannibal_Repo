
import os
import glob
from google.cloud import storage

# Configuration
BUCKET_NAME = "hannibal-screenplays"  # Using the one suggested in walkthrough
SOURCE_DIR = "Final_Proofread_Screenplays"
DESTINATION_FOLDER = "data"

def upload_files():
    # Initialize client (will search for credentials in env vars or default locations)
    # Note: On local machine, user might need to have run 'gcloud auth application-default login'
    try:
        storage_client = storage.Client()
    except Exception as e:
        print(f"Error initializing Storage Client: {e}")
        print("Please ensure you have authenticated using 'gcloud auth application-default login' or set GOOGLE_APPLICATION_CREDENTIALS.")
        return

    try:
        bucket = storage_client.bucket(BUCKET_NAME)
        # Check if bucket exists
        if not bucket.exists():
            print(f"Bucket {BUCKET_NAME} does not exist. Please create it or update the script.")
            return
    except Exception as e:
         print(f"Error accessing bucket: {e}")
         return

    files = glob.glob(os.path.join(SOURCE_DIR, "*.txt"))
    if not files:
        print(f"No files found in {SOURCE_DIR}.")
        return

    print(f"Found {len(files)} files to upload to gs://{BUCKET_NAME}/{DESTINATION_FOLDER}/...")

    for file_path in files:
        filename = os.path.basename(file_path)
        blob_name = f"{DESTINATION_FOLDER}/{filename}"
        blob = bucket.blob(blob_name)
        
        print(f"Uploading {filename}...")
        blob.upload_from_filename(file_path)
    
    print("Upload complete!")

if __name__ == "__main__":
    upload_files()
