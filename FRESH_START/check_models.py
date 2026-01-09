import google.generativeai as genai
import os

api_key = os.environ.get("GOOGLE_API_KEY")
if not api_key:
    with open("models_list_py.txt", "w", encoding="utf-8") as f:
        f.write("GOOGLE_API_KEY not found.")
else:
    genai.configure(api_key=api_key)
    try:
        with open("models_list_py.txt", "w", encoding="utf-8") as f:
            f.write("Listing available models:\n")
            for m in genai.list_models():
                if 'generateContent' in m.supported_generation_methods:
                    f.write(f"{m.name}\n")
    except Exception as e:
        with open("models_list_py.txt", "w", encoding="utf-8") as f:
            f.write(f"Error listing models: {e}")
