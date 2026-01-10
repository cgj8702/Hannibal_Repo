
import json
import os
import glob
import re
from collections import defaultdict

# Configuration
INPUT_PATTERN = "output_v2_chunked/*.jsonl"
OUTPUT_DIR = "Final_Proofread_Screenplays_V2"

def clean_markdown_and_parse(raw_text, filename, chunk_index):
    # Clean formatting (strip markdown code blocks)
    if raw_text.strip().startswith("```"):
        raw_text = re.sub(r"^```(?:json)?\n", "", raw_text.strip())
        raw_text = re.sub(r"\n```$", "", raw_text)
    
    raw_text = raw_text.strip()
    
    try:
        model_output = json.loads(raw_text)
        
        # Check structure
        if 'full_reconstructed_text' in model_output:
            return model_output['full_reconstructed_text']
        elif 'corrected_segments' in model_output:
            full_text = ""
            for seg in model_output['corrected_segments']:
                full_text += seg.get('corrected_text', '') + "\n\n"
            return full_text
        else:
            return str(model_output) # Fallback
            
    except json.JSONDecodeError as e:
        print(f"Warning: JSON decode failed for {filename} (Chunk {chunk_index}). Using raw snippet.")
        return raw_text

def process_v2_results():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        
    result_files = glob.glob(INPUT_PATTERN)
    if not result_files:
        print("No result files found in output_v2_chunked/")
        return

    print(f"Processing {len(result_files)} result files...")
    
    # Storage: filename -> {chunk_index -> content}
    screenplay_parts = defaultdict(dict)
    screenplay_totals = {} # filename -> total_chunks expected
    
    processed_count = 0
    
    for file_path in result_files:
        print(f"Reading {file_path}...")
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    entry = json.loads(line)
                    processed_count += 1
                    
                    # 1. Extract Metadata from Request
                    filename = "Unknown.txt"
                    chunk_index = -1
                    total_chunks = -1
                    
                    if 'request' in entry and 'contents' in entry['request']:
                        try:
                            prompt_text = entry['request']['contents'][0]['parts'][0]['text']
                            # Look for PROOFREAD_METADATA: filename='Hannibal_1x01.txt', chunk=1, total=5
                            meta_match = re.search(r"PROOFREAD_METADATA: filename='(.*?)', chunk=(\d+), total=(\d+)", prompt_text)
                            if meta_match:
                                filename = meta_match.group(1)
                                chunk_index = int(meta_match.group(2))
                                total_chunks = int(meta_match.group(3))
                                
                                screenplay_totals[filename] = total_chunks
                        except Exception:
                            pass
                    
                    if chunk_index == -1:
                        print("Skipping entry: Could not extract metadata.")
                        continue

                    # 2. Extract Response Content
                    content = ""
                    if 'response' in entry:
                        candidates = entry['response'].get('candidates', [])
                        if candidates:
                            raw_text = candidates[0]['content']['parts'][0]['text']
                            content = clean_markdown_and_parse(raw_text, filename, chunk_index)
                            
                    # Store
                    screenplay_parts[filename][chunk_index] = content
                    
                except Exception as e:
                    print(f"Error processing line: {e}")

    # 3. Reassemble and Save
    print(f"Reassembling {len(screenplay_parts)} screenplays...")
    
    for filename, parts in screenplay_parts.items():
        total_expected = screenplay_totals.get(filename, len(parts))
        
        # Check completeness
        missing_chunks = []
        full_text = ""
        
        for i in range(1, total_expected + 1):
            if i in parts:
                full_text += parts[i] + "\n\n" # Add some spacing between chunks
            else:
                missing_chunks.append(i)
                full_text += f"\n[MISSING CHUNK {i}]\n"
        
        if missing_chunks:
            print(f"Warning: {filename} missing chunks {missing_chunks}")
        
        # Determine Output Filename (Use original filename but prepend Proofread)
        # If filename was "Hannibal_1x01...txt", make it "Proofread_Hannibal_1x01...txt"
        out_name = f"Proofread_V2_{filename}"
        out_path = os.path.join(OUTPUT_DIR, out_name)
        
        with open(out_path, 'w', encoding='utf-8') as out_f:
            out_f.write(full_text)
            
        print(f"Saved {out_name} ({len(parts)}/{total_expected} chunks)")

if __name__ == "__main__":
    process_v2_results()
