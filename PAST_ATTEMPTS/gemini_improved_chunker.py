import re
import pdfplumber
import os
import json
from tqdm import tqdm

# --- CONFIGURATION ---
SCRIPTS_FOLDER = "C:/Users/carly/Documents/Coding/episode_scripts/"
OUTPUT_FOLDER = "C:/Users/carly/Documents/Coding/Hannibal_chunks/"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Uppercase words that are NEVER character names or locations
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
    "SCENE",
    "SEQUENCE",
]

# Lines that should be NUKED if they appear on their own
SHOT_HEADERS = {
    "PILLS",
    "A LABEL",
    "GUN",
    "SHOOTS",
    "BLAM",
    "BANG",
    "CLICK",
    "PHONE",
    "DOOR",
    "WINDOW",
    "REVEAL",
    "POP",
    "WIDEN",
    "HAND",
    "FACE",
    "EYES",
    "PENDULUM",
    "SHOVEL",
    "MUSHROOM",
    "BED",
    "OFF",
    "KNOCKS",
    "A GUN UNLOADING",
    "WIDENING",
    "REVERSING",
    "A SHOVEL",
    "PUSHING",
    "PULLING",
    "PANNING",
    "TILTING",
    "TRACKING",
    "THE CLUSTER OF NEST-SHAPED MUSHROOMS",
    "A NEST-SHAPED MUSHROOM",
}

# Words that disqualify a line from being a Header
PROMOTION_BLACKLIST = [
    "HOBBS",
    "GRAHAM",
    "LECTER",
    "CRAWFORD",
    "BLOOM",
    "KATZ",
    "ZELLER",
    "PRICE",
    "LOUNDS",
    "PILLS",
    "LABEL",
    "GUN",
    "SHOOTS",
    "BLAM",
    "BANG",
    "CLICK",
    "PHONE",
    "DOOR",
    "WINDOW",
    "REVEAL",
    "POP",
    "WIDEN",
    "HAND",
    "FACE",
    "EYES",
    "PENDULUM",
    "SHOVEL",
    "MUSHROOM",
    "BED",
    "OFF",
    "KNOCKS",
]

# ==========================================
# 1. THE "NUKE" LIST (Whole Line Killers)
# ==========================================
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
    "We are --",
    "we are --",
    "We are--",
]

# ==========================================
# 2. DYNAMIC NUKE REGEX (Whole Line)
# ==========================================
DYNAMIC_KILLERS = [
    # Catches "OFF WILL" or "ON ELEVATOR DOORS" - must be whole line or trailing
    r"^\s*(?:OFF|ON)\s+[A-Z0-9\s\(\)\'\-]+.*$",
    r"^\s*(?:FWUM|CLICK|BLAM|WHOOSH|CRASH|BANG)[^a-z]*$",  # Pure SFX
    # STRICT CAMERA KILLER: Catches "CAMERA comes", "CAMERA finds", "CAMERA POPS"
    # Matches start of line, "CAMERA", space, any text. Case insensitive flags used in check.
    r"^\s*CAMERA\s+.*$",
    r"^\s*COLD OPEN\s*$",
    r"^\s*FADE (?:IN|OUT|TO):?",
    r"^\s*(?:CUT|SMASH CUT|BACK|MATCH CUT|TIME CUT) TO:?",
    r"^\s*DISSOLVE TO:?",
    # Catches camera verbs at start of line
    r"^\s*(?:WIDENING|REVERSING|PUSHING|PULLING|PANNING|TILTING|TRACKING)\b",
]

# ==========================================
# 3. THE "SNIPER" LIST (Partial Removers)
# ==========================================
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
    # --- Transitions (Mixed Case included to catch residue) ---
    "CUT TO:",
    "CUT TO",
    "BACK TO:",
    "BACK TO",
    "DISSOLVE TO:",
    "OFF Will",
    "OFF Jack",
    "OFF Hannibal",
    "OFF Freddie",
    "OFF Alana",
    "OFF Abigail",
    "OFF Graham",
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
    # --- Chyron variations ---
    "A CHYRON tells us",
    "A CHYRON",
    # --- Technical Suffixes ---
    " - FLASHBACK",
    " - DREAM SEQUENCE",
    " - REALITY",
    " - CONTINUOUS",
    " - DREAM STATE",
    " - MEMORY FLASH",
    # --- Camera/Direction ---
    "POV",
    "REVERSE ANGLE",
    "WIDE ANGLE",
    "CLOSE ON",
    "EXTREME CLOSE UP",
    "EXTREME CLOSE ON",
    "ANGLED ON",
    "HIGH ANGLE",
    "LOW ANGLE",
    "WIDEN TO REVEAL",
    "CAMERA FOLLOWS",
]
PARTIAL_REMOVERS = sorted(list(set(_raw_sniper_items)), key=len, reverse=True)


def is_junk_line(text):
    """Checks if the line contains a 'Nuke' phrase or regex pattern."""
    clean = text.strip()
    if not clean:
        return True

    if any(k in clean for k in WHOLE_LINE_KILLERS):
        return True

    # Check strict shot headers (Handle potential trailing punctuation)
    clean_upper = clean.rstrip(".: ").upper()
    if clean_upper in SHOT_HEADERS:
        return True

    for pattern in DYNAMIC_KILLERS:
        # Ignore case for CAMERA checks to catch "CAMERA comes to REST"
        if re.search(pattern, clean, re.IGNORECASE):
            return True

    if re.search(r"HANNIBAL\s*-\s*(?:AIR|PROD).*?#\d+", clean):
        return True

    if re.search(r"^“.*”$|^\"[A-Z].*?[a-z]\"$", clean):
        return True

    # Standard "Scene Number" margin pattern (e.g. "1A" or "20")
    if re.match(r"^\s*[A-Z]?\d+[A-Z]?\s*$", clean):
        return True

    return False


def clean_specific_words(text):
    """Sniper function: Removes specific tags and cleans residue."""

    # 1. Run the sniper list (Enhanced for whitespace)
    # We replace spaces in the target phrases with \s+ to match across newlines/extra spaces
    for item in PARTIAL_REMOVERS:
        if item in text:  # Fast check
            # Create regex that matches the phrase with flexible whitespace
            pattern = re.escape(item).replace(r"\ ", r"[\s\n]+")
            text = re.sub(pattern, "", text, flags=re.MULTILINE | re.IGNORECASE)

    # 2. Dynamic Sniper for "OFF [Character]" residue
    # FIX: Uses (?:^|\s) to ensure we don't match inside words (e.g. ELD-ON)
    # Only removes if target is UPPERCASE or Numbers
    text = re.sub(r"(?:^|\s)(?:OFF|ON)\s+[A-Z0-9\s\(\)\'\-]+[.:]?\s*$", "", text)

    # 3. Nuke Empty Parentheses
    text = re.sub(r"\(\s*\)", "", text)

    # 4. Clean Leading/Trailing Scene Numbers
    text = re.sub(r"^\s*[A-Z]?\d+[A-Z]?\s+", "", text)
    text = re.sub(r"\s+[A-Z]?\d+[A-Z]?[.]?\s*$", "", text)

    # 5. Clean leading/trailing punctuation (Fixes "- PILLS", ": a TARRY")
    # DO NOT remove periods '.' from the end.
    text = re.sub(r"^\s*[-.:,]+\s*", "", text)
    text = re.sub(r"\s*[-:,]+\s*$", "", text)

    return re.sub(r" +", " ", text).strip()


def clean_scene_heading(text):
    """Safely cleans Scene Headings."""
    remove_list = [
        "RESUMING",
        "CONTINUOUS",
        "ESTABLISHING",
        "SAME TIME",
        "LATER",
        "MOMENTS LATER",
        "DREAM STATE",
        "MEMORY FLASH",
    ]
    pattern = r"\b(" + "|".join(remove_list) + r")\b"
    text = re.sub(pattern, "", text)

    text = re.sub(r"[-–—\s]+[A-Z]?\d+[A-Z]?\s*$", "", text)
    text = re.sub(r"^\s*[A-Z]?\d+[A-Z]?\s+(?=INT|EXT|I/E)", "", text)

    return text.strip(" -.,")


def normalize_text_set(text):
    clean = re.sub(r"[^A-Z0-9\s]", "", text.upper())
    return set(clean.split())


def get_line_type(raw_line):
    clean = raw_line.strip()
    if not clean:
        return "EMPTY", ""

    leading = len(raw_line) - len(raw_line.lstrip())
    line_width = len(clean)

    # 1. CHARACTER NAMES
    if leading >= 32 and clean.isupper() and clean not in STRICT_BLACKLIST:
        if not clean.startswith("("):
            return "CHARACTER", re.sub(r"\(.*?\)", "", clean).strip()

    # 2. DIALOGUE
    if 16 <= leading < 32 and line_width < 55:
        if clean.startswith("(") and clean.endswith(")"):
            return "EMPTY", ""
        return "DIALOGUE", clean

    # 3. SLUGLINES
    if leading < 16 and any(x in clean for x in ["INT.", "EXT.", "I/E", "INT/EXT"]):
        return "SLUGLINE", clean

    # 4. ACTION
    return "ACTION", clean


def process_script(pdf_path):
    chunks = []
    current_scene_content = []
    current_scene_title = "START"
    active_speaker = None
    dialogue_buffer = []

    def flush_dialogue():
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

            lines = text.split("\n")
            for i, line in enumerate(lines):
                if "Final Shooting Script" in line or "Collated" in line:
                    continue

                if is_junk_line(line):
                    if "CONTINUED" not in line:
                        flush_dialogue()
                    continue

                l_type, content = get_line_type(line)

                # --- 1. SCENE HEADING ---
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

                    current_scene_title = clean_scene_heading(content)

                # --- 2. CHARACTER NAME ---
                elif l_type == "CHARACTER":
                    flush_dialogue()
                    content = clean_specific_words(content)
                    active_speaker = content

                # --- 3. DIALOGUE ---
                elif l_type == "DIALOGUE":
                    dialogue_buffer.append(content)

                # --- 4. ACTION / DESCRIPTION ---
                elif l_type == "ACTION":
                    flush_dialogue()
                    content = clean_specific_words(content)
                    if not content:
                        continue

                    # --- ORPHAN MERGE LOGIC ---
                    if content[0].islower() and current_scene_content:
                        current_scene_content[-1] += " " + content
                        continue

                    # --- PROMOTION LOGIC ---
                    if (
                        content.isupper()
                        and len(content) < 80
                        and content not in STRICT_BLACKLIST
                    ):

                        # Guard 1: Props, Sounds, Yelling, Characters
                        is_bad_content = (
                            "!" in content
                            or "?" in content
                            or content.startswith("ON ")
                            or content.startswith("CAMERA")
                            or any(bad in content for bad in PROMOTION_BLACKLIST)
                            or content.rstrip(".:") in SHOT_HEADERS
                        )

                        # Guard 2: Verbs ending in -ING (Smiling, Reversing, Widening)
                        first_word = re.sub(r"[^A-Z]", "", content.split()[0])
                        is_verb = first_word.endswith("ING")

                        if is_bad_content or is_verb:
                            pass
                        elif content.startswith("-") or content.startswith("."):
                            pass
                        else:
                            # 4a. FUZZY REDUNDANCY CHECK
                            h_set = normalize_text_set(current_scene_title)
                            l_set = normalize_text_set(content)

                            intersection = l_set.intersection(h_set)
                            overlap_ratio = (
                                len(intersection) / len(l_set) if l_set else 0
                            )

                            if overlap_ratio >= 0.6:
                                continue

                            # 4b. SUB-LOCATION PROMOTION
                            current_scene_title += f" - {content}"
                            continue

                    current_scene_content.append(content)

    # Final flush
    flush_dialogue()
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

print(f"\nDone! Eldon protected, Camera Directions arrested. 🦌🍷")
