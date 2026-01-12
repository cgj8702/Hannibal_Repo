
import os
import json
import glob
import re

# Configuration
INPUT_DIR = "ParsedScreenplays"
OUTPUT_FILE = "batch_request_chunked.jsonl"
MODEL_ID = "publishers/google/models/gemini-2.0-flash-lite-001"
MAX_WORDS_PER_CHUNK = 2500

def chunk_screenplay(text):
    """
    Splits screenplay into chunks based on Scene Headers.
    Accumulates scenes until MAX_WORDS_PER_CHUNK is reached.
    """
    # Split by Scene Headers (capturing the split pattern)
    # Look for INT. EXT. or EST. at start of line
    # We use a lookahead to keep the delimiter
    scenes = re.split(r'(?=\n[ \t]*(?:INT\.|EXT\.|EST\.|I/E\.))', text)
    
    chunks = []
    current_chunk = []
    current_word_count = 0
    
    for scene in scenes:
        scene_word_count = len(scene.split())
        
        # If adding this scene exceeds max (and we have content), push chunk
        if current_chunk and (current_word_count + scene_word_count > MAX_WORDS_PER_CHUNK):
            chunks.append("".join(current_chunk))
            current_chunk = []
            current_word_count = 0
            
        current_chunk.append(scene)
        current_word_count += scene_word_count
        
    # Push final chunk
    if current_chunk:
        chunks.append("".join(current_chunk))
        
    return chunks

def create_batch_request():
    files = sorted(glob.glob(os.path.join(INPUT_DIR, "*.txt")))
    
    if not files:
        print("No files found!")
        return

    print(f"Found {len(files)} screenplays. Generating chunked batch request...")
    
    request_count = 0
    total_screenplays_processed = 0
    
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as out_f:
        for file_path in files:
            filename = os.path.basename(file_path)
            
            with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
                content = f.read()
            
            # Create Chunks
            chunks = chunk_screenplay(content)
            total_chunks = len(chunks)
            if total_chunks > 0:
                total_screenplays_processed += 1
                
            print(f"  {filename}: {total_chunks} chunks")
            
            for i, chunk_text in enumerate(chunks):
                chunk_index = i + 1
                
                # Metadata Header embedded in the Prompt
                metadata_header = f"PROOFREAD_METADATA: filename='{filename}', chunk={chunk_index}, total={total_chunks}\n\n"
                
                # Prompt Construction
                full_prompt_text = f"""
{metadata_header}
You are a professional screenplay proofreader. Your task is to proofread the following text, which is a raw extraction from a PDF screenplay.

The raw text has artifacts like:
- Page numbers (e.g., "1.")
- Headers (e.g., "HANNIBAL - PROD. #101...")
- "CONTINUED:" markers
- "OMITTED" lines

Your GOAL:
1. Remove all artifacts (page numbers, headers, CONTINUEDs, OMITTEDs).
2. Fix formatting issues (e.g., disjointed lines).
3. Return the CLEAN, corrected text in a specific JSON format.
4. Do NOT remove scene headers or dialogue. Keep the content intact.

Input Screenplay Chunk:
{chunk_text}
"""
                
                # Request Entry
                request_entry = {
                    "request": {
                        "contents": [
                            {
                                "role": "user",
                                "parts": [{"text": full_prompt_text}]
                            }
                        ],
                        "generationConfig": {
                            "response_mime_type": "application/json",
                            "response_schema": {
                                "type": "object",
                                "properties": {
                                    "corrected_segments": {
                                        "type": "array",
                                        "items": {
                                            "type": "object",
                                            "properties": {
                                                "scene_header": {"type": "string"},
                                                "original_text": {"type": "string"},
                                                "corrected_text": {"type": "string"},
                                                "changes_made": {
                                                    "type": "array",
                                                    "items": {"type": "string"}
                                                }
                                            },
                                            "required": ["original_text", "corrected_text"]
                                        }
                                    },
                                    "full_reconstructed_text": {"type": "string"}
                                }
                            }
                        }
                    }
                }
                
                out_f.write(json.dumps(request_entry) + "\n")
                request_count += 1
                
    print(f"Successfully generated {OUTPUT_FILE} with {request_count} requests from {total_screenplays_processed} files.")

if __name__ == "__main__":
    create_batch_request()
