
import json
import os
import re
import glob

# === CONFIGURATION ===
# Find the latest predictions file
files = glob.glob("prediction-model-*_predictions.jsonl")
if not files:
    raise FileNotFoundError("No prediction file found!")
INPUT_FILE = max(files, key=os.path.getctime) # Use the newest one
OUTPUT_DIR = "Final_Proofread_Screenplays"
# =====================

def extract_screenplay_content(full_response):
    """
    Extracts the screenplay text from the model's response.
    The model typically outputs:
    "Here are the errors...
    1. ...
    
    Here is the corrected text:
    ```
    INT. SCENE ...
    ```"
    We want what is inside the ``` blocks.
    """
    # Find all closed code blocks
    code_blocks = []
    last_end = 0
    pattern = re.compile(r'```(.*?)```', re.DOTALL)
    for match in pattern.finditer(full_response):
        code_blocks.append(match.group(1))
        last_end = match.end()
    
    # Check for unclosed code block after the last closed one
    remaining_text = full_response[last_end:]
    start_marker = "```"
    start_idx = remaining_text.find(start_marker)
    
    if start_idx != -1:
        # Found an unclosed block
        content = remaining_text[start_idx + len(start_marker):]
        # Remove language identifier if present
        if content.startswith("text\n"):
             content = content[5:]
        elif content.startswith("\n"):
             content = content[1:]
        
        print(f"Warning: Found unclosed code block at end. Appending it.")
        code_blocks.append(content.strip())
        
    # Join blocks if found, else use full response
    content = "\n".join(code_blocks).strip() if code_blocks else full_response

    # Post-processing: Remove "Analysis" or "Errors" preamble if present
    # Models often output: [Analysis of errors] \n\n **Corrected Screenplay Segment:** \n [Actual Text]
    # We want to keep ONLY [Actual Text]
    
    # Common headers used by the model
    header_patterns = [
        r'\*\*Corrected Screenplay Segment:?\*\*[:\s]*',
        r'\*\*Corrected Text:?\*\*[:\s]*',
        r'\*\*Corrected Text Segment:?\*\*[:\s]*',
        r'\*\*Corrected Segment:?\*\*[:\s]*',
        r'\*\*Corrected Information:?\*\*[:\s]*',
        r'Here is the corrected text[:\s]*',
        r'\*\*Corrected Screenplay:?\*\*[:\s]*'
    ]
    
    # DEBUG: Print content sample to see what we are matching against
    # print(f"DEBUG: Content start: {content[:50]!r}")

    for pattern in header_patterns:
        match = re.search(pattern, content, re.IGNORECASE)
        if match:
            print(f"MATCH FOUND: '{match.group()}' at index {match.start()}. Stripping {match.end()} chars.")
            possible_content = content[match.end():].strip()
            if len(possible_content) > 10: 
                 content = possible_content
                 break 
        # else:
            # print(f"No match for pattern: {pattern}")
    
    return content

def main():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        
    print(f"Processing {INPUT_FILE}...")
    
    # Store chunks temporarily: { "Filename": { 1: "content", 2: "content" } }
    file_chunks = {}
    
    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f):
            try:
                data = json.loads(line)
                
                # 1. Get Metadata from Request
                request_text = data['request']['contents'][0]['parts'][0]['text']
                
                # Extract Filename and Chunk ID
                # Pattern: "Filename: Hannibal_1x01_Aperitif.txt (Chunk 1)"
                match = re.search(r'Filename: (.*?) \(Chunk (\d+)\)', request_text)
                if not match:
                    print(f"Skipping line {line_num}: Could not find filename/chunk info.")
                    continue
                    
                filename = match.group(1)
                chunk_id = int(match.group(2))
                
                # 2. Get Predicted Content from Response
                if 'response' in data and 'candidates' in data['response']:
                    candidate = data['response']['candidates'][0]
                    if 'content' in candidate and 'parts' in candidate['content']:
                        full_output = candidate['content']['parts'][0]['text']
                        
                        # Extract just the screenplay part
                        clean_content = extract_screenplay_content(full_output)
                        
                        if filename not in file_chunks:
                            file_chunks[filename] = {}
                        
                        file_chunks[filename][chunk_id] = clean_content
                    else:
                        print(f"Line {line_num}: No content in response.")
                else:
                    print(f"Line {line_num}: No valid response object.")
                    
            except json.JSONDecodeError:
                print(f"Line {line_num}: Invalid JSON.")
            except Exception as e:
                print(f"Error processing line {line_num}: {e}")

    # 3. Reassemble and Save
    for filename, chunks in file_chunks.items():
        sorted_chunk_ids = sorted(chunks.keys())
        
        # Verify if we have all chunks? (Optional, but good to know)
        # Assuming we just concatenate what we have.
        
        full_text = ""
        for cid in sorted_chunk_ids:
            full_text += chunks[cid] + "\n\n"
            
        output_path = os.path.join(OUTPUT_DIR, f"Proofread_{filename}")
        with open(output_path, 'w', encoding='utf-8') as out:
            out.write(full_text)
            
        print(f"Saved {output_path} ({len(sorted_chunk_ids)} chunks)")

    print("\nProcessing Complete! Ready for Vector Database.")

if __name__ == "__main__":
    main()
