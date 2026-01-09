import re
import os
import json
import pdfplumber
from tqdm import tqdm

# --- CONFIGURATION ---
SCRIPTS_FOLDER = "C:/Users/carly/Documents/Coding/episode_scripts/"
OUTPUT_FOLDER = "C:/Users/carly/Documents/Coding/Hannibal_memories/"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)


class HannibalParser:
    def __init__(self):
        # ==========================================
        # 1. THE "NUKE" LIST (Whole Line Killers)
        # ==========================================
        self.WHOLE_LINE_KILLERS = [
            "CONTINUED",
            "OMITTED",
            "ACT ONE",
            "ACT TWO",
            "ACT THREE",
            "ACT FOUR",
            "ACT FIVE",
            "ACT SIX",
            "TEASER",
            "END OF SHOW",
            "END OF ACT",
            "END OF TEASER",
            "AIR #",
            "PROD #",
            "We are --",
            "we are --",
            "We are--",
            "Collated",
            "FINAL SHOOTING SCRIPT",
            "FULL BLUE",
            "PINK",
            "YELLOW",
            "GREEN",
        ]

        # ==========================================
        # 2. DYNAMIC KILLERS (Regex)
        # ==========================================
        self.DYNAMIC_KILLERS = [
            r"^\s*(?:FWUM|CLICK|BLAM|WHOOSH|CRASH|BANG|SMASH)[^a-z]*$",  # Pure SFX
            r"^\s*COLD OPEN\s*$",
            r"^\s*FADE (?:IN|OUT|TO):?",
            r"^\s*(?:CUT|SMASH CUT|BACK|MATCH CUT|TIME CUT|QUICK POP) TO:?",
            r"^\s*DISSOLVE TO:?",
            r"^\s*(?:WIDENING|REVERSING|PUSHING|PULLING|PANNING|TILTING|TRACKING)\b",
            # NUCLEAR FOOTER KILLER: Matches "HANNIBAL" ... "PROD" broadly to catch dash types
            r"^\s*HANNIBAL.*PROD.*$",
            # Page numbers just floating (e.g. " 1.")
            r"^\s*\d+\.?\s*$",
        ]

        # ==========================================
        # 3. THE "SNIPER" LIST (Partial Removers)
        # ==========================================
        self.PARTIAL_REMOVERS = [
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
            "CUT TO:",
            "CUT TO",
            "BACK TO:",
            "BACK TO",
            "DISSOLVE TO:",
            "QUICK POP TO:",
            "before we realize we are:",
            "and we...",
            "we --",
            "We are--",
            "we are--",
            "we are --",
            "We are --",
            "We see",
            "we see",
            "INCLUDE THE SURROUNDING POLICE LINE",
            "AT THE POLICE LINE",
            "ON THE MONITOR",
            "STOCK FOOTAGE",
            "BLACK",
            "A CHYRON tells us",
            "A CHYRON",
        ]

        # Words that disqualify a line from being a Character or Scene Header
        self.PROMOTION_BLACKLIST = [
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
            "SMASH",
            "QUICK POP TO:",
            "POP",
            "CLOSE ON",
            "SMASH CUT",
            "QUICK POP",
        ]

        self.shot_keywords = [
            "CAMERA",
            "ANGLE",
            "REVEAL",
            "CLOSE UP",
            "WIDE",
            "VIEW",
            "INSERT",
            "IMAGE",
            "P.O.V.",
            "PERSPECTIVE",
            "FOCUS",
            "PAN",
            "TRACK",
            "DOLLY",
        ]

        # Terms that indicate a state of mind or time shift (FOR METADATA)
        self.meta_markers = {
            "DREAM": ["DREAM", "NIGHTMARE", "HALLUCINATION", "SURREAL"],
            "FLASHBACK": ["FLASHBACK", "MEMORY", "PREVIOUSLY"],
            "POV": ["POV", "POINT OF VIEW", "THROUGH THE", "P.O.V."],
            "TIME": [
                "LATER",
                "MOMENTS LATER",
                "CONTINUOUS",
                "SAME TIME",
                "DAY",
                "NIGHT",
                "DAWN",
                "DUSK",
            ],
        }

    def clean_text(self, text):
        """Run the sniper list to clean dialogue tags and artifacts."""
        for item in self.PARTIAL_REMOVERS:
            if item in text:
                pattern = re.escape(item).replace(r"\ ", r"[\s\n]+")
                text = re.sub(pattern, "", text, flags=re.MULTILINE | re.IGNORECASE)

        text = re.sub(r"\(\s*\)", "", text)
        return text.strip()

    def transform_observation(self, text):
        """The Hannibal Special: Turns Camera directions into Observations."""
        # Added apostrophes to the regex class [A-Z0-9\s\-\'’]
        match = re.search(r"^(?:OFF|ON)\s+([A-Z0-9\s\-\'’]+)(.*)", text, re.IGNORECASE)
        if match:
            target = match.group(1).strip().title()
            description = match.group(2).strip().strip(",.-")
            if not description:
                return f"[Observation: Focus is on {target}]"
            return f"[Observation: {target} reacts: {description}]"

        if text.upper().startswith("REVEAL"):
            return "We see " + text[6:].strip()

        return text

    def extract_scene_metadata(self, heading):
        """Extracts rich metadata from the slugline."""
        meta = {
            "is_dream": False,
            "is_flashback": False,
            "pov": None,
            "time_of_day": "UNKNOWN",
            "location": "UNKNOWN",
            "characters_present": [],
        }

        heading_upper = heading.upper()

        # Check Boolean Markers
        for key in self.meta_markers["DREAM"]:
            if key in heading_upper:
                meta["is_dream"] = True
        for key in self.meta_markers["FLASHBACK"]:
            if key in heading_upper:
                meta["is_flashback"] = True
        for key in self.meta_markers["POV"]:
            if key in heading_upper:
                meta["pov"] = "Specific POV"
                break

        # Extract Time of Day
        for time_key in self.meta_markers["TIME"]:
            pattern = r"(?:^|\s|-)" + re.escape(time_key) + r"(?:$|\s)"
            if re.search(pattern, heading_upper):
                meta["time_of_day"] = time_key
                break

        # Extract Location - ROBUST VERSION
        # 1. Find the first occurrence of INT. or EXT. and chop everything before it
        start_match = re.search(r"(?:INT\.|EXT\.|I/E)", heading_upper)
        if start_match:
            loc_clean = heading_upper[start_match.start() :]
        else:
            # Fallback if no INT/EXT found (unlikely for valid headers)
            loc_clean = re.sub(r"^\d+\s+", "", heading_upper)

        # 2. Remove the INT./EXT. prefix itself
        loc_clean = re.sub(r"^(?:INT\.|EXT\.|I/E)\s+", "", loc_clean).strip()

        # 3. Remove Page Numbers at the end
        loc_clean = re.sub(r"\s+\d+\.?$", "", loc_clean)

        # Remove Time
        if meta["time_of_day"] != "UNKNOWN":
            loc_clean = loc_clean.replace(meta["time_of_day"], "")

        # Remove POV markers from Location string
        for pov_marker in self.meta_markers["POV"]:
            loc_clean = loc_clean.replace(pov_marker, "")

        # Clean specific artifacts
        for noise in [
            "- ESTABLISHING",
            "RESUMING",
            "CONTINUOUS",
            "MOMENTS LATER",
            "OMNISCIENT",
            "WILL'S",
            "WILL’S",
        ]:
            loc_clean = loc_clean.replace(noise, "")

        # Cleanup dangling dashes and spaces
        loc_clean = re.sub(r"\s*-\s*", " - ", loc_clean)  # normalize dashes
        loc_clean = re.sub(r"^[-\s]+|[-\s]+$", "", loc_clean)  # strip ends
        loc_clean = re.sub(r"-\s*-\s*", "-", loc_clean)  # double dashes
        loc_clean = re.sub(r"\s+", " ", loc_clean)  # collapse spaces

        meta["location"] = loc_clean.strip()
        return meta

    def get_line_type(self, raw_line):
        """Line detection logic."""
        clean = raw_line.strip()
        if not clean:
            return "EMPTY", ""

        leading = len(raw_line) - len(raw_line.lstrip())
        line_width = len(clean)

        # 1. CHARACTER NAMES
        if leading >= 30 and clean.isupper():
            # STRIP PUNCTUATION before checking blacklist to catch "SMASH!"
            clean_no_punct = clean.strip(".:!?- ")

            if (
                not clean.startswith("(")
                and clean_no_punct not in self.PROMOTION_BLACKLIST
            ):
                char_name = re.sub(r"\(.*?\)", "", clean).strip()
                # Extra check to ensure it's not a SHOT pretending to be a char
                if char_name not in self.shot_keywords:
                    return "CHARACTER", char_name

        # 2. DIALOGUE
        if 14 <= leading < 32 and line_width < 55:
            if clean.startswith("(") and clean.endswith(")"):
                return "PARENTHETICAL", clean

            # ALL CAPS CHECK
            if clean.isupper():
                if any(kw in clean for kw in self.shot_keywords):
                    return "ACTION", clean
                # If it's short and has no punctuation, likely a minor shot (CLOSE ON TABLE)
                if len(clean.split()) < 10 and not any(
                    p in clean for p in [".", "!", "?"]
                ):
                    return "ACTION", clean
                # If it's in blacklist, definitely ACTION
                if clean.strip(".:!?- ") in self.PROMOTION_BLACKLIST:
                    return "ACTION", clean

                return "DIALOGUE", clean

            return "DIALOGUE", clean

        # 3. SLUGLINES
        if leading < 16 and any(x in clean for x in ["INT.", "EXT.", "I/E", "INT/EXT"]):
            return "SLUGLINE", clean

        # 4. ACTION
        return "ACTION", clean

    def is_junk_line(self, text):
        """Checks if the line matches any Killers."""
        clean = text.strip()
        if any(k in clean for k in self.WHOLE_LINE_KILLERS):
            return True
        for pattern in self.DYNAMIC_KILLERS:
            if re.search(pattern, clean, re.IGNORECASE):
                return True
        return False

    def parse_pdf(self, pdf_path):
        chunks = []
        current_scene_content = []
        current_scene_title_text = "START"
        current_scene_meta = self.extract_scene_metadata("START")

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

        tqdm.write(f"Processing: {os.path.basename(pdf_path)}")

        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages[1:]:
                text = page.extract_text(layout=True)
                if not text:
                    continue

                lines = text.split("\n")
                for line in lines:
                    if self.is_junk_line(line):
                        continue

                    l_type, content = self.get_line_type(line)

                    # --- 1. SCENE HEADING ---
                    if l_type == "SLUGLINE":
                        flush_dialogue()
                        if current_scene_content:
                            chunks.append(
                                {
                                    "heading": current_scene_title_text,
                                    "metadata": current_scene_meta,
                                    "content": "\n".join(current_scene_content).strip(),
                                }
                            )
                            current_scene_content = []

                        current_scene_title_text = self.clean_text(content)
                        current_scene_meta = self.extract_scene_metadata(content)

                    # --- 2. CHARACTER NAME ---
                    elif l_type == "CHARACTER":
                        flush_dialogue()
                        active_speaker = self.clean_text(content)
                        if (
                            active_speaker
                            not in current_scene_meta["characters_present"]
                        ):
                            current_scene_meta["characters_present"].append(
                                active_speaker
                            )

                    # --- 3. DIALOGUE ---
                    elif l_type == "DIALOGUE":
                        clean_d = self.clean_text(content)
                        if clean_d:
                            dialogue_buffer.append(clean_d)

                    # --- 4. PARENTHETICAL ---
                    elif l_type == "PARENTHETICAL":
                        clean_p = content.lower().replace("(", "").replace(")", "")
                        if active_speaker:
                            dialogue_buffer.append(f"[{clean_p}]")

                    # --- 5. ACTION ---
                    elif l_type == "ACTION":
                        flush_dialogue()
                        clean_action = self.clean_text(content)

                        is_promotable = (
                            clean_action.isupper()
                            and len(clean_action) < 50
                            and clean_action.strip(".:!?- ")
                            not in self.PROMOTION_BLACKLIST
                            and not any(x in clean_action for x in ["!", "?"])
                        )

                        if is_promotable:
                            if clean_action in current_scene_title_text:
                                continue

                        final_action = self.transform_observation(clean_action)

                        if (
                            final_action
                            and final_action[0].islower()
                            and current_scene_content
                        ):
                            current_scene_content[-1] += " " + final_action
                        elif final_action:
                            current_scene_content.append(final_action)

        flush_dialogue()
        if current_scene_content:
            chunks.append(
                {
                    "heading": current_scene_title_text,
                    "metadata": current_scene_meta,
                    "content": "\n".join(current_scene_content).strip(),
                }
            )

        return chunks


# --- EXECUTION ---
if __name__ == "__main__":
    parser = HannibalParser()
    if os.path.exists(SCRIPTS_FOLDER):
        pdf_files = [f for f in os.listdir(SCRIPTS_FOLDER) if f.endswith(".pdf")]
        for filename in tqdm(pdf_files):
            full_path = os.path.join(SCRIPTS_FOLDER, filename)
            data = parser.parse_pdf(full_path)

            out_name = os.path.splitext(filename)[0] + "_memory.json"
            with open(
                os.path.join(OUTPUT_FOLDER, out_name), "w", encoding="utf-8"
            ) as f:
                json.dump(data, f, indent=4)

            debug_name = os.path.splitext(filename)[0] + "_readable.txt"
            with open(
                os.path.join(OUTPUT_FOLDER, debug_name), "w", encoding="utf-8"
            ) as f:
                for scene in data:
                    f.write(f"\n=== {scene['heading']} ===\n")
                    f.write(f"[META: {scene['metadata']}]\n")
                    f.write(scene["content"] + "\n")
                    f.write("-" * 40 + "\n")
        print("Processing Complete. Memory Palace Constructed.")
    else:
        print(f"Warning: Script folder not found at {SCRIPTS_FOLDER}")
