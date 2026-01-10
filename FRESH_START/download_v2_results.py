
from google.cloud import storage
import os
import glob

BUCKET_NAME = "hannibal-screenplays"
PREFIX = "output_v2_chunked/"
LOCAL_DIR = "output_v2_chunked"

def download_results():
    if not os.path.exists(LOCAL_DIR):
        os.makedirs(LOCAL_DIR)

    storage_client = storage.Client()
    bucket = storage_client.bucket(BUCKET_NAME)
    blobs = bucket.list_blobs(prefix=PREFIX)

    count = 0
    for blob in blobs:
        if blob.name.endswith(".jsonl"):
            # Construct local path, removing prefix directory structure if needed
            filename = os.path.basename(blob.name)
            local_path = os.path.join(LOCAL_DIR, filename)
            
            print(f"Downloading {blob.name} to {local_path}...")
            blob.download_to_filename(local_path)
            count += 1

    print(f"Downloaded {count} files.")

if __name__ == "__main__":
    download_results()
