import pdfplumber
import re

pdf_path = "C:/Users/carly/Documents/Coding/movie_scripts/big_fish.pdf"

# 1. KEYWORD PATTERN (The Classics)
# Matches INT., EXT., I/E, and now INTERCUT
KEYWORD_PATTERN = re.compile(
    r"^\s*(?:[\d]+[A-Z]?\s+)?(?:INT\.|EXT\.|INT\./EXT\.|EXT\./INT\.|I\/E\.|INTERCUT)\s+", 
    re.IGNORECASE
)

# 2. SHOOTING SCRIPT PATTERN (The "Sandwich")
# Matches: Number -> Uppercase Text -> (Optional Number)
# Ex: "53 FURTHER ALONG 53" or "190 YOUNG EDWARD"
# We ensure the text is [A-Z] (uppercase), numbers, or punctuation like . / ' - ( )
SHOOTING_SCRIPT_PATTERN = re.compile(
    r"^\s*\d+[A-Z]?\s+[A-Z0-9 \-\.\/\(\)\'\’]+\s*(?:\d+[A-Z]?)?\s*$"
)

# 3. EXCLUSION PATTERN
# Shooting scripts often put "(CONTINUED)" or "OMITTED" in uppercase; ignore those.
IGNORE_PATTERN = re.compile(r"^\s*(?:\d+[A-Z]?\s+)?(?:\(CONTINUED\)|CONTINUED:|OMITTED)", re.IGNORECASE)

# Helper to strip numbers from the start/end of the line for the final clean text
MARGIN_NUMBER_PATTERN = re.compile(r"(^\s*\d+[A-Z]?\s+)|(\s+\d+[A-Z]?\s*$)", re.MULTILINE)

def is_scene_heading(text):
    clean = text.strip()
    
    # 1. If it's a "CONTINUED" or "OMITTED" line, it's NOT a scene.
    if IGNORE_PATTERN.match(clean):
        return False
        
    # 2. Check for Keywords (INT/EXT/INTERCUT)
    if KEYWORD_PATTERN.match(clean):
        return True
        
    # 3. Check for "Shooting Script" style (Number + Uppercase Text)
    # This catches "53 FURTHER ALONG 53"
    if SHOOTING_SCRIPT_PATTERN.match(clean):
        # Double check: Ensure there is at least one letter so we don't match page numbers like "12."
        if any(c.isalpha() for c in clean):
            return True
            
    return False

def chunk_script_by_scenes(pdf_path):
    chunks = []
    current_chunk = []
    current_scene_title = "START"
    
    with pdfplumber.open(pdf_path) as pdf:
        full_text = ""
        for page in pdf.pages:
            text = page.extract_text(layout=True)
            if text:
                full_text += text + "\n"

    lines = full_text.split('\n')

    for line in lines:
        clean_line = line.strip()
        
        # USE OUR NEW FUNCTION
        if is_scene_heading(clean_line):
            
            # Save previous scene
            if current_chunk:
                chunks.append({
                    "scene_heading": current_scene_title,
                    "content": "\n".join(current_chunk).strip()
                })
                current_chunk = []
            
            # Update Title (and clean the numbers off the title for readability)
            current_scene_title = MARGIN_NUMBER_PATTERN.sub("", clean_line).strip()
            
        else:
            # Add dialogue (cleaning off margin numbers)
            line_without_numbers = MARGIN_NUMBER_PATTERN.sub("", line)
            if line_without_numbers.strip():
                current_chunk.append(line_without_numbers)

    if current_chunk:
        chunks.append({
            "scene_heading": current_scene_title,
            "content": "\n".join(current_chunk).strip()
        })

    return chunks

# Run it
script_chunks = chunk_script_by_scenes(pdf_path)

# Verify
print(f"Total Scenes Found: {len(script_chunks)}")
with open("final_check_v3.txt", "w", encoding="utf-8") as f:
    for chunk in script_chunks:
        f.write(f"=== {chunk['scene_heading']} ===\n")
        f.write(chunk['content'])
        f.write("\n\n")

# Run and Save
script_chunks = chunk_script_by_scenes(pdf_path)

# Verify the count
print(f"Total Scenes Found: {len(script_chunks)}")

# Assuming you already ran the code from before and have the 'script_chunks' variable
def save_chunks_to_text(chunks, output_filename):
    with open(output_filename, "w", encoding="utf-8") as f:
        for i, chunk in enumerate(chunks):
            f.write(f"================ SCENE {i+1} ================\n")
            f.write(f"HEADING: {chunk['scene_heading']}\n")
            f.write("---------------------------------------------\n")
            f.write(chunk['content'])
            f.write("\n\n") # Add extra space between scenes


            # Usage
save_chunks_to_text(script_chunks, "my_script_debug.txt")
print("Saved! Go check 'my_script_debug.txt' to see your scenes.")

import json

def save_chunks_to_json(chunks, output_filename):
    with open(output_filename, "w", encoding="utf-8") as f:
        # indent=4 makes it look pretty and readable to humans too
        json.dump(chunks, f, indent=4)

# Usage
save_chunks_to_json(script_chunks, "my_script_data.json")