
from google.cloud import aiplatform
import time

# Configuration
PROJECT_ID = "hannibal-chatbot-447514" # Derived from previous context or user's project
# Wait, I don't know the project ID for sure. 
# I will try to get it from environment or let the user fill it.
# Actually, the user has a GCS bucket "hannibal-screenplays". I can assume the project is linked.
# I will use a placeholder or try to infer. 
# The `gcloud config list` output would have been useful.
# But `google.auth.default()` usually picks it up.
# Let's try to pass None and see if it picks up from auth.

REGION = "us-central1"
JOB_DISPLAY_NAME = "hannibal-proofread-v2"
MODEL_NAME = "publishers/google/models/gemini-2.0-flash-lite-001"
GCS_SOURCE = "gs://hannibal-screenplays/batch_request_v2.jsonl"
GCS_DESTINATION = "gs://hannibal-screenplays/output_v2/"

def submit_job():
    aiplatform.init(location=REGION)

    job = aiplatform.BatchPredictionJob.create(
        job_display_name=JOB_DISPLAY_NAME,
        model_name=MODEL_NAME,
        instances_format="jsonl",
        predictions_format="jsonl",
        gcs_source=GCS_SOURCE,
        gcs_destination_prefix=GCS_DESTINATION,
    )

    print(f"Job submitted. Resource name: {job.resource_name}")
    print(f"State: {job.state}")
    return job

if __name__ == "__main__":
    try:
        submit_job()
    except Exception as e:
        print(f"Failed to submit job: {e}")
        # Setup fallback instructions
        print("\nIf this failed due to Project ID, please run:")
        print(f"gcloud ai batch-predictions create --display-name={JOB_DISPLAY_NAME} --model={MODEL_NAME} --gcs-source={GCS_SOURCE} --gcs-destination-prefix={GCS_DESTINATION} --region={REGION}")
