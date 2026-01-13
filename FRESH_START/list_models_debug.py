import google.generativeai as genai
import os

API_KEY = os.environ.get("GOOGLE_API_KEY")

if not API_KEY:
    print("Error: GOOGLE_API_KEY not set.")
    exit()

genai.configure(api_key=API_KEY)

print("Listing available models...")
try:
    with open("models_list_utf8.txt", "w", encoding="utf-8") as f:
        for m in genai.list_models():
            line = f"Model: {m.name} | SupportedGenerationMethods: {m.supported_generation_methods}\n"
            print(line.strip())
            f.write(line)
    print("\nList saved to models_list_utf8.txt")
except Exception as e:
    print(f"Error: {e}")
