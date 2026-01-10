
from google.cloud import aiplatform
import time

# Configuration
import google.auth

# Configuration
# Try to get project from environment or auth
try:
    _, project_id = google.auth.default()
    PROJECT_ID = project_id
    print(f"Detected Project ID: {PROJECT_ID}")
except:
    PROJECT_ID = "hannibal-chatbot-447514" # Fallback
    print(f"Using Fallback Project ID: {PROJECT_ID}")

REGION = "us-central1"
JOB_DISPLAY_NAME = "hannibal-proofread-v2"
MODEL_NAME = "publishers/google/models/gemini-2.0-flash-lite-001"
GCS_SOURCE = "gs://hannibal-screenplays/batch_request_v2.jsonl"
GCS_DESTINATION = "gs://hannibal-screenplays/output_v2/"

def submit_job():
    aiplatform.init(project=PROJECT_ID, location=REGION)

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
