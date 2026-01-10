
from google.cloud import storage
import os

# Configuration
BUCKET_NAME = "hannibal-screenplays"
SOURCE_FILE = "batch_request_chunked.jsonl"
DESTINATION_BLOB_NAME = "batch_request_chunked.jsonl"

def upload_blob():
    """Uploads a file to the bucket."""
    storage_client = storage.Client()
    bucket = storage_client.bucket(BUCKET_NAME)
    blob = bucket.blob(DESTINATION_BLOB_NAME)

    print(f"Uploading {SOURCE_FILE} to {BUCKET_NAME}/{DESTINATION_BLOB_NAME}...")
    blob.upload_from_filename(SOURCE_FILE)

    print(f"File {SOURCE_FILE} uploaded to {DESTINATION_BLOB_NAME}.")

if __name__ == "__main__":
    upload_blob()
