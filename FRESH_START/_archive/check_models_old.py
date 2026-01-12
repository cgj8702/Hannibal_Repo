
import vertexai
from vertexai.generative_models import GenerativeModel

PROJECT_ID = "gen-lang-client-0813719350"
LOCATION = "us-central1"

vertexai.init(project=PROJECT_ID, location=LOCATION)

# Try to verify model existence by instantiating simple generation
models_to_test = [
    "gemini-1.5-pro-preview-0409",
    "gemini-1.5-pro-001",
    "gemini-1.5-pro",
    "gemini-1.0-pro-001",
    "gemini-2.0-flash-exp",
    "gemini-2.0-flash-lite-001"
]

print("Testing Model Availability:")
for m_id in models_to_test:
    try:
        model = GenerativeModel(m_id)
        # Quick ping (dry run)
        print(f"  {m_id}: ", end="")
        response = model.generate_content("Hi", generation_config={"max_output_tokens": 10})
        print("AVAILABLE")
    except Exception as e:
        print(f"FAILED ({e})")
