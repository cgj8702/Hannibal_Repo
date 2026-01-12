
import vertexai
from vertexai.generative_models import GenerativeModel, SafetySetting

PROJECT_ID = "gen-lang-client-0813719350"
LOCATION = "us-central1"
MODEL_ID = "gemini-2.0-flash-lite-001"

def test_8k_chunk():
    vertexai.init(project=PROJECT_ID, location=LOCATION)
    
    model = GenerativeModel(MODEL_ID)
    
    # 1. 8k Chunk with typo injected
    # This text is approx 8000 chars (grabbed from view_file output) using simple string rep here
    text_chunk = """
    EXT. HOME - LIVING ROOM - NIGHT
    Arterial spray splashes a wall near a blood-soaked carpet.
    Through the windows we see DOZENS OF OFICERRS and as many
    POLICE CARS. A CRIME-SCENE PHOTOGRAPHER takes pictures.
    """ + ("\nThis is filler text to reach 8000 characters. " * 200)

    print(f"Testing Chunk Size: {len(text_chunk)} characters")
    
    prompt = (
        "Proofread the following text for grammar and spelling errors. "
        "Output ONLY the corrected text. Do not add markdown formatting, "
        "introductions, or explanations. Maintain all original line breaks.\n\n"
        f"{text_chunk}"
    )

    print(f"--- Sending 8k Request to {MODEL_ID} ---")
    
    try:
        response = model.generate_content(
            prompt,
            generation_config={"temperature": 0.2, "max_output_tokens": 8192}
        )
        
        corrected = response.text
        if "OFICERRS" in corrected:
             print("\nRESULT: FAILED (Echoed typo 'OFICERRS')")
        else:
             print("\nRESULT: SUCCESS (Fixed 'OFICERRS' -> 'OFFICERS')")

    except Exception as e:
        print(f"\nERROR: {e}")

if __name__ == "__main__":
    test_8k_chunk()
