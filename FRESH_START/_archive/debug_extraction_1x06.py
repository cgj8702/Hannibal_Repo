
import json
import glob
import re
import os

# Copying the exact logic from process_batch_results.py
def extract_screenplay_content(full_response):
    # 1. Find all code blocks
    code_block_pattern = re.compile(r'```(.*?)```', re.DOTALL)
    code_blocks = code_block_pattern.findall(full_response)
    
    # Handle unclosed code block at the end
    if not code_blocks:
        start_marker = "```"
        if full_response.count(start_marker) >= 1:
            last_start = full_response.rfind(start_marker)
            # check if there is a closing marker after it
            if full_response.find(start_marker, last_start + 3) == -1:
                content_after = full_response[last_start + 3:]
                code_blocks.append(content_after.strip())

    # Join blocks if found, else use full response
    content = "\n".join(code_blocks).strip() if code_blocks else full_response

    print(f"DEBUG: Content length: {len(content)}")
    print(f"DEBUG: Content sample (middle): {content[len(content)//2 - 100 : len(content)//2 + 100]!r}")

    # Post-processing: Remove "Analysis" or "Errors" preamble if present
    header_patterns = [
        r'\*\*Corrected Screenplay Segment:?\*\*[:\s]*',
        r'\*\*Corrected Text:?\*\*[:\s]*',
        r'\*\*Corrected Text Segment:?\*\*[:\s]*',
        r'\*\*Corrected Segment:?\*\*[:\s]*',
        r'\*\*Corrected Information:?\*\*[:\s]*',
        r'Here is the corrected text[:\s]*',
        r'\*\*Corrected Screenplay:?\*\*[:\s]*'
    ]
    
    for pattern in header_patterns:
        match = re.search(pattern, content, re.IGNORECASE)
        if match:
            print(f"MATCH FOUND: '{match.group()}' at index {match.start()}. Stripping {match.end()} chars.")
            possible_content = content[match.end():].strip()
            if len(possible_content) > 10: 
                 content = possible_content
                 break 
        else:
            print(f"No match for pattern: {pattern}")
    
    return content

# Find the prediction file
prediction_files = glob.glob("prediction-model-*.jsonl")
if not prediction_files:
    print("No prediction file found.")
    exit()

PREDICTION_FILE = prediction_files[0]
print(f"Reading {PREDICTION_FILE}...")

full_text = ""
with open(PREDICTION_FILE, 'r', encoding='utf-8') as f:
    for line in f:
        try:
            entry = json.loads(line)
            # Vertex AI Batch Prediction output format
            if 'instance' in entry and 'content' in entry['instance']:
                input_uri = entry['instance']['content']
                filename = os.path.basename(input_uri)
            else:
                # Fallback or skip
                continue
                
            if "Hannibal_1x06_Entree.txt" in filename:
                print(f"Found 1x06 entry: {filename}")
                
                # Extract prediction
                if 'prediction' in entry:
                    # Usually candidates -> content -> parts -> text
                    candidates = entry['prediction'].get('candidates', [])
                    if candidates:
                        content_obj = candidates[0].get('content', {})
                        parts = content_obj.get('parts', [])
                        if parts:
                            text = parts[0].get('text', '')
                            
                            print("--- Processing Chunk ---")
                            extracted = extract_screenplay_content(text)
                            print(f"--- Extracted Length: {len(extracted)} ---")
                            full_text += extracted + "\n\n"
        except Exception as e:
            print(f"Error parsing line: {e}")

print("Done processing.")
