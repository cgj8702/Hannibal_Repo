
import vertexai
from vertexai.generative_models import GenerativeModel, SafetySetting

PROJECT_ID = "gen-lang-client-0813719350"
LOCATION = "us-central1"
MODEL_ID = "gemini-2.0-flash-lite-001"

def test_model():
    vertexai.init(project=PROJECT_ID, location=LOCATION)
    
    model = GenerativeModel(MODEL_ID)
    
    # 1. Test Text with obvious typos
    bad_text = (
        "EXT. FOREST - NGHT\n\n"
        "Hannibal walk through the woods. He is hungary. "
        "The sky is bleck and the stars is shinning bright.\n"
        "He hold a knife in his hand."
    )
    
    prompt = (
        "Proofread the following text for grammar and spelling errors. "
        "Output ONLY the corrected text. Do not add markdown formatting, "
        "introductions, or explanations. Maintain all original line breaks.\n\n"
        f"{bad_text}"
    )

    print(f"--- Sending Test Request to {MODEL_ID} ---")
    print(f"Input:\n{bad_text}\n")
    
    try:
        # Disable safety filters to ensure it doesn't block "knife" or "woods" context
        safety_settings = [
            SafetySetting(
                category=SafetySetting.HarmCategory.HARM_CATEGORY_HATE_SPEECH,
                threshold=SafetySetting.HarmBlockThreshold.BLOCK_NONE,
            ),
            SafetySetting(
                category=SafetySetting.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
                threshold=SafetySetting.HarmBlockThreshold.BLOCK_NONE,
            ),
            SafetySetting(
                category=SafetySetting.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
                threshold=SafetySetting.HarmBlockThreshold.BLOCK_NONE,
            ),
            SafetySetting(
                category=SafetySetting.HarmCategory.HARM_CATEGORY_HARASSMENT,
                threshold=SafetySetting.HarmBlockThreshold.BLOCK_NONE,
            ),
        ]

        response = model.generate_content(
            prompt,
            safety_settings=safety_settings,
            generation_config={"temperature": 0.2, "max_output_tokens": 1024}
        )
        
        print(f"--- Response ---")
        print(response.text)
        
        if response.text.strip() == bad_text.strip():
            print("\nRESULT: FAILURE (ECHOED)")
        else:
            print("\nRESULT: SUCCESS (EDITED)")
            
    except Exception as e:
        print(f"\nERROR: {e}")

if __name__ == "__main__":
    test_model()
