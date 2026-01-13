from google import genai
from google.genai import types
import os
import glob
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

# --- CONFIGURATION ---
API_KEY = os.environ.get("GOOGLE_API_KEY")

# Folder names
INPUT_FOLDER = "raw_scripts"
OUTPUT_FOLDER = "clean_scripts"

# --- THE FINAL SYSTEM INSTRUCTION ---
# This is the exact "Scorched Earth" protocol we developed.
SYSTEM_INSTRUCTION = """
<system_role>
You are an expert script editor. You must convert raw screenplay text into a clean, readable narrative format (dialogue + action + locations) inside a text code block.
</system_role>

<critical_formatting_rules>
1. CONTAINER: Output inside a code block. Start with ```text and end with ```.
2. CLEAN TEXT ONLY: Remove all tags. The output should look like a novel, not a database.
3. DIALOGUE QUOTES: Enclose all spoken dialogue in double quotes ("Hello").
4. NO MARKDOWN: No bold, italics, or headers.
5. SPACING: Double-newlines between blocks.
</critical_formatting_rules>

<priority_cleaning_protocols>
1. SANITIZATION (PRIORITY #0):
   - DELETE all tags.
   - DELETE all === number and number === wrappers around headers.
   - DELETE all lines containing "OMITTED" (e.g., "18 OMITTED 18").
   - TRUNCATE headers: If a header ends with "-- ESTABLISHING", delete that suffix.

2. SLUGLINE PURGE (STRICT):
   - KEEP: Lines starting with "INT." or "EXT."
   - EXCEPTION (Objects): Convert "A PENDULUM" or "A RAVEN-FEATHERED STAG" to sentence case and merge with the next line.
   - DELETE: ALL other uppercase structure lines, including:
     * "TEASER", "ACT ONE", "ACT TWO", "THE END"
     * "QUICK POP TO:", "SMASH CUT TO:", "MATCH CUT TO:", "TIME CUT TO:"
     * "CLOSE ON...", "ANGLE ON...", "THE DOOR", "THE MISSING LUNGS", "ON WILL"

3. HEADER RECONSTRUCTION:
   - Screenplays often split headers across lines. MERGE them.
   - Input:
     INT.
     NICHOLS’ HOME - NIGHT
   - Output:
     INT. NICHOLS’ HOME - NIGHT

4. RESIDUE DESTRUCTION:
   - DELETE standalone words: "Establishing", "Establishing.", "Chyron:", "CONTINUOUS", "Omniscient P.O.V.".
   - EXCEPTION: If a line contains location info (e.g., "We are in Duluth"), MOVE it to the Scene Header.
</priority_cleaning_protocols>

<name_normalization>
Always use the FULL NAME in dialogue tags:
- "Hannibal" -> Hannibal Lecter:
- "Jacob" -> Garret Jacob Hobbs:
- "Abigail" -> Abigail Hobbs:
- "Will" -> Will Graham:
- "Jack" -> Jack Crawford:
- "Alana" -> Alana Bloom:
- "Beverly" -> Beverly Katz:
- "Chilton" -> Dr. Frederick Chilton:
</name_normalization>

<narrative_logic>
1. HEADER RELOCATION: If a Scene Header (INT./EXT.) appears AFTER a sound or action that describes the scene, move the Header to the TOP of that sequence.
2. CAMERA REMOVAL: Rewrite "Camera reveals..." or "Camera finds..." as simple action.
   - Input: "Camera reveals Beverly." -> Output: "Beverly is there."
</narrative_logic>

<task>
Clean and reformat the following screenplay text:
</task>
"""

def process_single_file(filepath):
    filename = os.path.basename(filepath)
    output_path = os.path.join(OUTPUT_FOLDER, f"CLEAN_{filename}")
    
    # Skip if already exists (optional, but good for resuming)
    # FORCE OVERWRITE for now to fix bad files
    # if os.path.exists(output_path):
    #     return f"Skipped (Exists): {filename}"

    try:
        with open(filepath, "r", encoding="utf-8") as f:
            raw_text = f.read()

        client = genai.Client(api_key=API_KEY)
        response = client.models.generate_content(
            model="gemini-2.5-pro",
            contents=raw_text,
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_INSTRUCTION
            )
        )

        clean_text = response.text.replace("```text", "").replace("```", "").strip()

        with open(output_path, "w", encoding="utf-8") as f:
            f.write(clean_text)
            
        return f"Completed: {filename}"
        
    except Exception as e:
        return f"ERROR ({filename}): {e}"

def process_scripts_parallel():
    if not API_KEY:
        print("Error: GOOGLE_API_KEY not set.")
        return

    # genai.configure(api_key=API_KEY) # Not needed for new SDK
    
    if not os.path.exists(OUTPUT_FOLDER):
        os.makedirs(OUTPUT_FOLDER)

    files = glob.glob(os.path.join(INPUT_FOLDER, "*.txt"))
    if not files:
        print(f"No .txt files found in '{INPUT_FOLDER}'.")
        return

    print(f"Found {len(files)} scripts. Starting parallel processing...")
    
    # 5 workers is usually safe for rate limits on Pro tier, adjust if needed
    max_workers = 5 
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_file = {executor.submit(process_single_file, f): f for f in files}
        
        # Process as they complete
        for future in as_completed(future_to_file):
            result = future.result()
            print(result)
            # Small sleep to be nice to the API even in parallel
            time.sleep(0.5) 

    print("\nParallel processing complete!")

if __name__ == "__main__":
    process_scripts_parallel()
