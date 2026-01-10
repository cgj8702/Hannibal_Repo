
import os
import json
import glob
# Removed explicit SDK imports to generate raw JSONL for the batch API directly

# Configuration
INPUT_DIR = "ParsedScreenplays"
OUTPUT_FILE = "batch_request_v2.jsonl"
MODEL_ID = "gemini-2.0-flash-lite-preview-02-05" # Using the preview model often yields better instruction following
# Or stick to the one we used: "gemini-2.0-flash-lite-001"
# Let's use the stable lite version to avoid surprises, or actually, the user used "gemini-2.0-flash-lite-001" before.
# Let's check what was used. The previous file said "model": "publishers/google/models/gemini-2.0-flash-lite-001".
MODEL_ID = "gemini-2.0-flash-lite-001"

# Schema Definition
response_schema = {
    "type": "object",
    "properties": {
        "corrected_segments": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "scene_header": {"type": "string", "description": "The scene header if present in this segment"},
                    "original_text": {"type": "string", "description": "The exact original text of the segment"},
                    "corrected_text": {"type": "string", "description": "The fully corrected text of the segment, preserving screenplay format"},
                    "changes_made": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of specific corrections made (e.g., 'Removed line number', 'Fixed typo')"
                    }
                },
                "required": ["original_text", "corrected_text"]
            }
        }
    }
}

def create_batch_request():
    files = glob.glob(os.path.join(INPUT_DIR, "*.txt"))
    
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as out_f:
        for file_path in files:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            filename = os.path.basename(file_path)
            
            # Construct the prompt
            prompt = f"""
            You are a professional screenplay proofreader. Your task is to proofread the following text, which is a raw extraction from a PDF screenplay.
            
            The raw text has artifacts like:
            - Page numbers (e.g., "1.")
            - Headers (e.g., "HANNIBAL - PROD. #101...")
            - "CONTINUED:" markers
            - "OMITTED" lines
            
            Your GOAL:
            1. Remove all these artifacts.
            2. Fix any obvious OCR errors or formatting issues.
            3. Return the clean, reconstructed screenplay content in the JSON format specified.
            4. Do NOT summarize or analyze. Just give me the text.
            
            Screenplay Content:
            {content}
            """
            
            # Construct JSONL Entry
            # Note: The structure for Vertex AI Batch is specific.
            # We must use "contents" for the prompt and "generation_config" for schema.
            
            request_entry = {
                "request": {
                    "contents": [
                        {"role": "user", "parts": [{"text": prompt}]}
                    ],
                    "generationConfig": {
                        "responseMimeType": "application/json",
                        "responseSchema": response_schema
                    }
                }
            }
            
            out_f.write(json.dumps(request_entry) + "\n")
            
    print(f"Generated {OUTPUT_FILE} with {len(files)} requests.")

if __name__ == "__main__":
    create_batch_request()
