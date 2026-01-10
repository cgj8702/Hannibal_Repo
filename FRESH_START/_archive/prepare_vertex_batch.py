
import json
import os
import glob
import re

# === CONFIGURATION ===
INPUT_DIR = "ParsedScreenplays"
OUTPUT_FILE = "batch_requests.jsonl"
MODEL_URI = "publishers/google/models/gemini-1.5-flash-002" # Fast and cheap model
# =====================

def split_into_chunks(text, max_chars=14000): # Larger chunks for Gemini 1.5 Flash (1M context)
    # Split by scene headers to respect context
    scene_pattern = re.compile(r'(=== \d+ .*? ===)')
    parts = scene_pattern.split(text)
    chunks = []
    current_chunk = ""
    if parts:
         current_chunk = parts[0]
         for i in range(1, len(parts), 2):
             header = parts[i]
             content = parts[i+1] if i+1 < len(parts) else ""
             full_scene = header + content
             if len(current_chunk) + len(full_scene) > max_chars:
                 chunks.append(current_chunk)
                 current_chunk = full_scene
             else:
                 current_chunk += full_scene
         if current_chunk:
             chunks.append(current_chunk)
    else:
        chunks = [text[i:i+max_chars] for i in range(0, len(text), max_chars)]
    return chunks

def create_batch_request():
    files = sorted(glob.glob(os.path.join(INPUT_DIR, "*.txt")))
    print(f"Found {len(files)} screenplays.")
    
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as outfile:
        request_count = 0
        for file_path in files:
            filename = os.path.basename(file_path)
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Recalculate chunks for larger context model
            chunks = split_into_chunks(content)
            
            for i, chunk in enumerate(chunks):
                prompt = (
                    "You are a professional proofreader. Proofread the following screenplay text segment, "
                    "identifying typos, formatting inconsistencies, or logical errors. "
                    "For each error you find:\n"
                    "1. Cite the Episode (from filename) and Scene (from headers).\n"
                    "2. Explain the error.\n"
                    "3. Provide the corrected text for that section.\n\n"
                    "After listing the errors, provide the full validated and corrected version of the text segment. "
                    "Do not leave out any part of the original text in your corrected version.\n\n"
                    f"Filename: {filename} (Chunk {i+1})\n\n"
                    "Screenplay Content:\n"
                    f"{chunk}"
                )
                
                # Vertex AI Batch Request Format
                request_body = {
                    "request": {
                        "contents": [
                            {
                                "role": "user",
                                "parts": [{"text": prompt}]
                            }
                        ],
                        "generationConfig": {
                            "temperature": 0.2, # Lower temp for more accurate proofreading
                            "maxOutputTokens": 8192,
                        }
                    }
                }
                
                outfile.write(json.dumps(request_body) + "\n")
                request_count += 1
                
    print(f"Successfully created {OUTPUT_FILE} with {request_count} requests!")
    print("Next Step: Upload this file to a Google Cloud Storage Bucket.")

if __name__ == "__main__":
    create_batch_request()
