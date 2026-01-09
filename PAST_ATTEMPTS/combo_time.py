import re
import pdfplumber
import os
import json
from tqdm import tqdm

# --- CONFIGURATION ---
SCRIPTS_FOLDER = "C:/Users/carly/Documents/Coding/episode_scripts/"
OUTPUT_FOLDER = "C:/Users/carly/Documents/Coding/Hannibal_chunks/"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# User's Title/Unit Regex (Untouched)
UNIT = r"(?=.*\d)[A-Z0-9]+(?:-[A-Z0-9]+)?"

# Uppercase words that are NEVER character names
STRICT_BLACKLIST = [
    "TEASER",
    "ACT ONE",
    "ACT TWO",
    "ACT THREE",
    "ACT FOUR",
    "ACT FIVE",
    "CONTINUED",
    "OMITTED",
    "SHOW",
    "END OF SHOW",
    "END OF TEASER",
    "ACT ",
]

# ==========================================
# 1. THE "NUKE" LIST (Whole Line Killers)
# ==========================================
# If these appear, the line is deleted entirely.
WHOLE_LINE_KILLERS = [
    "CONTINUED",
    "OMITTED",
    "ACT ",
    "TEASER",
    "END OF SHOW",
    "END OF ACT",
    "END OF TEASER",
    "AIR #",
    "PROD #",
]

# ==========================================
# 2. THE "SNIPER" LIST (Partial Removers)
# ==========================================
# Grouped for organization.
# NOTE: Duplicates are removed and sorting happens automatically below.

_raw_sniper_items = [
    # --- Dialogue Tags & Parentheticals ---
    "(CONT'D)",
    "(CONT’D)",
    "(cont'd)",
    "(cont’d)",
    "(V.O.)",
    "(O.S.)",
    "(pre-lap)",
    "(into phone)",
    "(then)",
    "(MORE)",
    "...",
    # --- Transitions ---
    "CUT TO:",
    "BACK TO:",
    "POP TO:",
    "SMASH BACK TO:",
    "QUICK POP TO:",
    "TIME CUT TO:",
    "CUT TO",
    "BACK TO",
    # --- Camera Movements & Directions ---
    "CAMERA POPS WIDE TO REVEAL",
    "as CAMERA PULLS BACK TO REVEAL we are --",
    "CAMERA PULLS BACK TO REVEAL",
    "CAMERA PUSHES IN ON",
    "CAMERA FINDS the car,",
    "CAMERA TRACKS along a",
    "CAMERA comes to REST on",
    "CAMERA MOVES INTO",
    "CAMERA REVEALS",
    "and CAMERA REVEALS",
    "CAMERA FOLLOWS",
    "CAMERA FINDS",
    "CAMERA INCLUDES",
    "CAMERA comes",
    "WIDENING, we see that",
    "WIDENING, we find --",
    "WIDEN TO REVEAL--",
    "CLOSE ON -",
    "CLOSE ON",
    "We POP CLOSE",
    "REVERSING,",
    "REVEALS",
    "PUSH",
    "DOLLY",
    # --- Narrative Bridges ---
    "before we realize we are:",
    "and we...",
    "we --",
    "We are--",
    "we are--",
    "we are --",
    "We are --",
    "We see",
    "we see",
    # --- Specific Production Artifacts ---
    "INCLUDE THE SURROUNDING POLICE LINE",
    "AT THE POLICE LINE",
    "ON THE MONITOR",
    "STOCK FOOTAGE",
    "OMNISCIENT P.O.V.",
    "BLACK",
    # --- Other stupidly specific lines ---
    "on the picture -- and Will suddenly walks into",
    "frame",
    "a HIGH-POWERED MICROPHONE, like a STETHOSCOPE,",
    "held against the wall.",
    "the thin cable of the",
    "HIGH-POWERED MICROPHONE to its source.",
    "EXTREME CLOSE ON - PILLS" "",
    "The DOLLS roll and turn as they funnel from their AUTOMATED",
    "PILL DISPENSER into a PILL BOTTLE.",
    "CLOSE ON - A LABEL",
    # --- Chyron Variations ---
    "A CHYRON tells us",
    "A CHRYON tells us",
    "A CHYRON",
    "A CHRYON",
    # --- Script Header Leaks (Preventing "OFF WILL:" in prose) ---
    "OFF WILL,",
    "OFF Will,",
    "OFF WILL:",
    "OFF JACK,",
    "OFF JACK:",
    "ON WILL,",
    "ON WILL:",
    "ON JACK,",
    "ON JACK:",
]

# --- AUTO-SORTING & DEDUPING ---
# 1. set() removes duplicates
# 2. sorted(key=len, reverse=True) puts Longest phrases first so they are caught before shorter substrings
PARTIAL_REMOVERS = sorted(list(set(_raw_sniper_items)), key=len, reverse=True)


def is_junk_line(text):
    """
    Checks if the line contains a 'Nuke' phrase or regex pattern.
    STRICT CASE SENSITIVITY APPLIED.
    """
    clean = text.strip()
    if not clean:
        return True

    # 1. Check the Nuke List (Case Sensitive)
    nuke_pattern = "|".join(map(re.escape, WHOLE_LINE_KILLERS))
    if re.search(nuke_pattern, clean):
        return True

    # 2. Hannibal Production Footer Regex (Case Sensitive)
    if re.search(r"HANNIBAL\s*-\s*(?:AIR|PROD).*?#\d+", clean):
        return True

    # 3. Episode Titles & Margin Codes
    if re.search(r"^“.*”$|^\"[A-Z].*?[a-z]\"$", clean):
        return True
    if re.match(rf"^\s*{UNIT}\.?\s*$", clean):
        return True

    return False


def clean_specific_words(text):
    """
    Sniper function: Removes specific tags.
    STRICT CASE SENSITIVITY APPLIED + MULTILINE SUPPORT.
    """
    sniper_pattern = "|".join(map(re.escape, PARTIAL_REMOVERS))

    # flags=re.MULTILINE helps if the text chunk has newlines
    cleaned_text = re.sub(sniper_pattern, "", text, flags=re.MULTILINE)

    return re.sub(r" +", " ", cleaned_text).strip()


def get_line_type(raw_line):
    """
    STRICT Gutters for Shooting Scripts:
    - Action/Sluglines: Left (0-15 spaces)
    - Dialogue: Middle (16-32 spaces) + NARROW width
    - Character Names: Deep (33+ spaces) + ALL CAPS
    """
    clean = raw_line.strip()
    # FIX 1: Return a tuple ("EMPTY", "") instead of just string "EMPTY"
    if not clean:
        return "EMPTY", ""

    leading = len(raw_line) - len(raw_line.lstrip())
    line_width = len(clean)

    # 1. CHARACTER NAMES (Deep centered indent)
    if leading >= 32 and clean.isupper() and clean not in STRICT_BLACKLIST:
        # Filter parentheticals that slip into character gutter
        if not clean.startswith("("):
            return "CHARACTER", re.sub(r"\(.*?\)", "", clean).strip()

    # 2. DIALOGUE (Centered column + Narrow line width)
    if 16 <= leading < 32 and line_width < 55:
        # FIX 2: Return tuple here too for ignored parentheticals
        if clean.startswith("(") and clean.endswith(")"):
            return "EMPTY", ""
        return "DIALOGUE", clean

    # 3. SLUGLINES (Left margin keywords)
    if leading < 16 and any(x in clean for x in ["INT.", "EXT.", "I/E", "INT/EXT"]):
        return "SLUGLINE", clean

    # 4. ACTION (Descriptions)
    # If it's at the left OR if it's too wide to be dialogue
    if leading < 16 or line_width >= 55:
        return "ACTION", clean

    return "ACTION", clean


def process_script(pdf_path):
    chunks = []
    current_scene_content = []
    current_scene_title = "START"

    active_speaker = None
    dialogue_buffer = []

    def flush_dialogue():
        """Saves current buffer and clears speaker state."""
        nonlocal active_speaker, dialogue_buffer
        if active_speaker and dialogue_buffer:
            speech = " ".join(dialogue_buffer).strip()
            if speech:
                current_scene_content.append(f'{active_speaker}: "{speech}"')
        active_speaker = None
        dialogue_buffer = []

    tqdm.write(f"Processing: {os.path.basename(pdf_path)}...")

    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages[1:]:
            text = page.extract_text(layout=True)
            if not text:
                continue

            for line in text.split("\n"):
                if "Final Shooting Script" in line or "Collated" in line:
                    continue

                # 1. NUKE CHECK (Strict Case)
                if is_junk_line(line):
                    if "CONTINUED" not in line:
                        flush_dialogue()
                    continue

                l_type, content = get_line_type(line)

                # 2. SCENE BREAK
                if l_type == "SLUGLINE":
                    flush_dialogue()
                    if current_scene_content:
                        chunks.append(
                            {
                                "heading": current_scene_title,
                                "content": "\n".join(current_scene_content).strip(),
                            }
                        )
                        current_scene_content = []
                    current_scene_title = re.sub(
                        rf"^{UNIT}\s+|\s+{UNIT}$", "", content
                    ).strip()

                # 3. SPEAKER BREAK
                elif l_type == "CHARACTER":
                    flush_dialogue()
                    content = clean_specific_words(content)
                    active_speaker = content

                # 4. SPEECH COLLECTION
                elif l_type == "DIALOGUE":
                    dialogue_buffer.append(content)

                # 5. ACTION / NARRATION
                elif l_type == "ACTION":
                    flush_dialogue()
                    content = re.sub(rf"^{UNIT}\s+|\s+{UNIT}$", "", content).strip()
                    content = clean_specific_words(content)

                    # [ORPHAN KILLER REMOVED]

                    if content and content not in STRICT_BLACKLIST:
                        current_scene_content.append(content)

    flush_dialogue()  # Final catch
    if current_scene_content:
        chunks.append(
            {
                "heading": current_scene_title,
                "content": "\n".join(current_scene_content).strip(),
            }
        )

    return chunks


# --- EXECUTION ---
pdf_files = [f for f in os.listdir(SCRIPTS_FOLDER) if f.endswith(".pdf")]
for filename in tqdm(pdf_files, desc="Parsing Chunks"):
    data = process_script(os.path.join(SCRIPTS_FOLDER, filename))
    base = os.path.splitext(filename)[0]

    with open(
        os.path.join(OUTPUT_FOLDER, f"{base}_data.json"), "w", encoding="utf-8"
    ) as f:
        json.dump(data, f, indent=4)

    with open(
        os.path.join(OUTPUT_FOLDER, f"{base}_debug.txt"), "w", encoding="utf-8"
    ) as f:
        for scene in data:
            f.write(f"=== {scene['heading']} ===\n{scene['content']}\n\n")

print(f"\nDone! Case-sensitive cleaning complete. 🦌🍷")
