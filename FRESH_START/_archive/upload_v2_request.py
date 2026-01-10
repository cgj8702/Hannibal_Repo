
from google.cloud import storage
import os

BUCKET_NAME = "hannibal-screenplays"
SOURCE_FILE = "batch_request_v2.jsonl"
DESTINATION_BLOB_NAME = "batch_request_v2.jsonl"

def upload_blob(bucket_name, source_file_name, destination_blob_name):
    """Uploads a file to the bucket."""
    # The ID of your GCS bucket
    # bucket_name = "your-bucket-name"
    # The path to your file to upload
    # source_file_name = "local/path/to/file"
    # The ID of your GCS object
    # destination_blob_name = "storage-object-name"

    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)

    blob.upload_from_filename(source_file_name)

    print(
        f"File {source_file_name} uploaded to {destination_blob_name}."
    )

if __name__ == "__main__":
    if os.path.exists(SOURCE_FILE):
        print(f"Uploading {SOURCE_FILE} to gs://{BUCKET_NAME}/{DESTINATION_BLOB_NAME}...")
        try:
            upload_blob(BUCKET_NAME, SOURCE_FILE, DESTINATION_BLOB_NAME)
            print("Upload Successful.")
        except Exception as e:
            print(f"Upload Failed: {e}")
    else:
        print(f"File {SOURCE_FILE} not found.")
