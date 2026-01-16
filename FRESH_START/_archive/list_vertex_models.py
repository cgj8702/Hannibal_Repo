import google.auth
from google.cloud import aiplatform

def list_foundational_models():
    project = "gen-lang-client-0813719350"
    location = "us-central1"
    
    aiplatform.init(project=project, location=location)
    
    # This might list publisher models if available
    from google.cloud.aiplatform_v1 import (
        ModelServiceClient,
        ListPublisherModelsRequest,
    )
    
    client = ModelServiceClient(
        client_options={"api_endpoint": f"{location}-aiplatform.googleapis.com"}
    )
    
    parent = f"locations/{location}/publishers/google"
    request = ListPublisherModelsRequest(parent=parent)
    
    print(f"--- Models in {parent} ---")
    try:
        results = client.list_publisher_models(request=request)
        for model in results:
            if "gemini" in model.name.lower():
                print(model.name)
    except Exception as e:
        print(f"Error listing models: {e}")

if __name__ == "__main__":
    list_foundational_models()
