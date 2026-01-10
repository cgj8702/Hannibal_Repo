
from google.cloud import storage
from google.cloud import aiplatform
import google.auth

def verify_access():
    print("--- Verifying Google Cloud Access ---")
    
    # 1. Credentials & Project
    try:
        credentials, project_id = google.auth.default()
        print(f"Credentials found: {credentials != None}")
        print(f"Inferred Project ID: {project_id}")
    except Exception as e:
        print(f"Auth credential check failed: {e}")
        return

    # 2. Storage Access
    print("\n--- Testing Storage Access ---")
    try:
        storage_client = storage.Client(credentials=credentials, project=project_id)
        buckets = list(storage_client.list_buckets(max_results=5))
        print("Successfully listed buckets:")
        for b in buckets:
            print(f" - {b.name}")
    except Exception as e:
        print(f"Storage access failed: {e}")

    # 3. Vertex AI Access
    print("\n--- Testing Vertex AI Access ---")
    try:
        aiplatform.init(project=project_id, location="us-central1")
        # Try to list jobs to verify API access
        jobs = aiplatform.BatchPredictionJob.list(limit=1)
        print("Successfully connected to Vertex AI.")
        print(f"Examples of existing jobs: {[j.display_name for j in jobs]}")
    except Exception as e:
        print(f"Vertex AI access failed: {e}")
        print("Note: You may need to enable the Vertex AI API or set the correct quota project.")

if __name__ == "__main__":
    verify_access()
