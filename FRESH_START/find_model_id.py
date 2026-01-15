import os
from langchain_google_genai import ChatGoogleGenerativeAI, HarmCategory, HarmBlockThreshold

def test_model_version(model_id):
    project = "gen-lang-client-0813719350"
    location = "us-central1"
    
    print(f"--- Testing {model_id} ---")
    try:
        llm = ChatGoogleGenerativeAI(
            model=model_id,
            vertexai=True,
            project=project,
            location=location,
            safety_settings={
                HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.OFF,
                HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.OFF,
                HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.OFF,
                HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.OFF,
            }
        )
        # Dummy call to verify access
        res = llm.invoke("Hello")
        print(f"SUCCESS: {model_id} is accessible.")
        return True
    except Exception as e:
        print(f"FAILURE for {model_id}: {e}")
        return False

if __name__ == "__main__":
    versions = [
        "gemini-2.0-flash-001",
        "gemini-2.0-flash-lite-001",
        "gemini-1.5-flash-002",
    ]
    for v in versions:
        if test_model_version(v):
            print(f"\nFinal Recommendation: Use '{v}'")
            break
