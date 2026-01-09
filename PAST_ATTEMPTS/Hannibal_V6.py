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
            "END OF EPISODE",
            "THE END",
            "FINAL SHOOTING SCRIPT",
            "FINAL DRAFT",
            "Collated",
            "WHITE REVISIONS",
            "BLUE",
            "PINK",
            "YELLOW",
            "GREEN",
            "GOLDENROD",
            "WHITE",
        ]

        # ==========================================
        # 2. DYNAMIC KILLERS (Regex)
        # ==========================================
        self.DYNAMIC_KILLERS = [
            # Pure SFX and mechanical sounds from S1 and S2
            r"^\s*(?:FWUM|CLICK|BLAM|WHOOSH|CRASH|BANG|SMASH|KLANG|K’CHUK|RING|PLIP|WHAM|P’KEE|PFFT|KLAXON|BUZZ|P'KEET|SHLUCK|PWUM|PWUM|PLIP)[^a-z]*$",
            r"^\s*FADE (?:IN|OUT|TO):?",
            r"^\s*(?:CUT|SMASH CUT|BACK|MATCH CUT|TIME CUT|QUICK POP|SMASH BACK) TO:?",
            r"^\s*DISSOLVE TO:?",
            r"^\s*(?:WIDENING|REVERSING|PUSHING|PULLING|PANNING|TILTING|TRACKING)\b",
            # THE FOOTER KILLER: Enhanced to catch S2 style "HANNIBAL Ep. #201..."
            r"HANNIBAL.*(?:PROD|AIR|Ep\.).*#\d+.*",
            # Catch trailing date/page patterns in S2 footers
            r"\d{2}/\d{2}/\d{2}\s+\d+\.?\s*$",
            # Floating page numbers
            r"^\s*\d+\.\s*$",
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
            "(O.C.)",
            "(pre-lap)",
            "(into phone)",
            "(then)",
            "(MORE)",
            "(stunned)",
            "(whispers)",
            "(calling out)",
            "(intense)",
            "(pointing)",
            "(distorted, dreamlike)",
            "...",
            "we --",
            "We are--",
            "we are--",
            "we are --",
            "We are --",
            "A CHYRON tells us",
            "A CHYRON",
            "RE-USE EP 101",
            "PRESENT",
            "FLASHBACK",
            "OMNISCIENT P.O.V.",
            "WILL’S P.O.V.",
            "WILL'S P.O.V.",
            "GEORGIA’S P.O.V.",
            "LONE MAN’S P.O.V.",
            "ABIGAIL’S P.O.V.",
            "JACK'S P.O.V.",
            "WILL’S FEVERISH P.O.V.",
            "ALANA’S POV",
            "INTERCUT WITH:",
            "WILL IS HALLUCINATING.",
            "STOCK FOOTAGE",
        ]

        # Words that disqualify a line from being a Character name
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
            "GIDEON",
            "CHILTON",
            "LASS",
            "SUTCLIFFE",
            "MADCHEN",
            "BOYLE",
            "BUDISH",
            "SUMMERS",
            "CARRUTHERS",
            "SHELL",
            "STAG",
            "PENDULUM",
            "WINSTON",
            "DOE",
            "PILLS",
            "LABEL",
            "GUN",
            "REVEAL",
            "CAMERA",
            "CLOSE ON",
            "ON WILL",
            "ON HANNIBAL",
            "ON JACK",
            "ON ALANA",
            "A GLASS OF WATER",
            "THE ALARM CLOCK",
            "A PENDULUM",
            "A PICTURE",
            "THE MORGUE",
            "A SEA URCHIN",
            "A BEAUTIFUL PIECE",
            "A TASTEFULLY-ORNATE",
            "A LARYNGOSCOPIC VIEW",
            "THE STRING SECTION",
            "THE MISSING INTESTINES",
            "A KITCHEN CENTRIFUGE",
            "A SNOWY SKY",
            "A HORRIBLE EYE",
            "A PAIR OF EYES",
            "THE BED OVERTURNS",
            "A RAINBOW TROUT",
            "UNDER THE BED",
            "A ROW OF CORN",
            "THE HUMAN MURAL",
            "THE MISSING LEG",
            "A BEAUTIFUL PLATE",
            "A SUDDEN PIERCING",
            "A SLOW SWINGING",
            "THE WENDIGO MAN STAG",
            "A CUT-GLASS PERFUME",
        ]

    def clean_text(self, text):
        """Standardizes text and removes dialogue artifacts."""
        for item in self.PARTIAL_REMOVERS:
            text = text.replace(item, "")
        text = re.sub(r"\(\s*\)", "", text)
        return text.strip()

    def transform_observation(self, text):
        """Converts camera/focus cues into narrative observations."""
        match = re.search(
            r"^(?:OFF|ON)\s+([A-Z0-9\s\-\'’\.!]+)(.*)", text, re.IGNORECASE
        )
        if match:
            target = match.group(1).strip().title()
            desc = match.group(2).strip().strip(",.- ")
            if not desc:
                return f"[Observation: Focus on {target}]"
            return f"[Observation: {target} reacts: {desc}]"

        if text.upper().startswith("REVEAL"):
            return "We see " + text[6:].strip()
        return text

    def extract_scene_metadata(self, heading):
        """Extracts location, time, and mental state from sluglines."""
        meta = {
            "is_dream": False,
            "is_flashback": False,
            "time": "UNKNOWN",
            "location": "UNKNOWN",
        }
        heading_upper = heading.upper()

        if any(
            x in heading_upper
            for x in ["DREAM", "NIGHTMARE", "HALLUCINATION", "DREAMSCAPE", "FEVERISH"]
        ):
            meta["is_dream"] = True
        if any(
            x in heading_upper
            for x in ["FLASHBACK", "MEMORY", "TWO YEARS EARLIER", "2 YEARS AGO"]
        ):
            meta["is_flashback"] = True

        # Extract Time
        times = ["DAY", "NIGHT", "MORNING", "EVENING", "DAWN", "DUSK"]
        for t in times:
            if re.search(rf"\b{t}\b", heading_upper):
                meta["time"] = t
                break

        # Extract Location: Handles Alphanumeric IDs and 'A'/'B' scene suffixes
        loc = re.sub(r"^[A-Z]*\d+[A-Z]*\s+", "", heading_upper)
        loc = re.sub(r"\s+[A-Z]*\d+[A-Z]*\.?$", "", loc)
        loc = re.sub(r"^(?:INT\.|EXT\.|I/E|INT/EXT)\s+", "", loc)

        for noise in [
            "- DAY",
            "- NIGHT",
            "- MORNING",
            "- EVENING",
            "RESUMING",
            "CONTINUOUS",
            "PRESENT",
            "FLASHBACK",
        ]:
            loc = loc.replace(noise, "")

        meta["location"] = loc.strip(" -")
        return meta

    def get_line_type(self, raw_line):
        """Determines line type via indentation and keywords."""
        clean = raw_line.strip()
        if not clean:
            return "EMPTY", ""
        leading = len(raw_line) - len(raw_line.lstrip())

        # 1. CHARACTER NAMES
        if leading >= 25 and clean.isupper() and not clean.startswith("("):
            # Check if it starts with 'A ' or 'THE ' to filter out visual headers
            if not any(clean.startswith(x) for x in ["A ", "THE "]):
                if clean not in self.PROMOTION_BLACKLIST and len(clean.split()) <= 4:
                    return "CHARACTER", re.sub(r"\(.*?\)", "", clean).strip()

        # 2. DIALOGUE & PARENTHETICALS
        if 8 <= leading < 30:
            if clean.startswith("(") and clean.endswith(")"):
                return "PARENTHETICAL", clean
            return "DIALOGUE", clean

        # 3. SLUGLINES (Headers)
        if any(
            x in clean
            for x in [
                "INT.",
                "EXT.",
                "I/E",
                "UNDERWATER",
                "REALITY",
                "IN HIS MIND",
                "IN THE",
            ]
        ):
            return "SLUGLINE", clean

        return "ACTION", clean

    def is_junk_line(self, text):
        clean = text.strip()
        if any(k in clean for k in self.WHOLE_LINE_KILLERS):
            return True
        for pattern in self.DYNAMIC_KILLERS:
            if re.search(pattern, text, re.IGNORECASE):
                return True
        return False

    def parse_pdf(self, pdf_path):
        scenes = []
        current_scene = {"heading": "START", "metadata": {}, "content": []}
        active_speaker = None
        dialogue_buffer = []

        def flush_dialogue():
            nonlocal active_speaker, dialogue_buffer
            if active_speaker and dialogue_buffer:
                speech = " ".join(dialogue_buffer).strip()
                if speech:
                    current_scene["content"].append(f'{active_speaker}: "{speech}"')
            active_speaker = None
            dialogue_buffer = []

        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages[1:]:
                text_content = page.extract_text(layout=True)
                if not text_content:
                    continue

                for line in text_content.split("\n"):
                    if self.is_junk_line(line):
                        continue
                    l_type, content = self.get_line_type(line)

                    if l_type == "SLUGLINE":
                        flush_dialogue()
                        if current_scene["content"]:
                            scenes.append(current_scene)
                        current_scene = {
                            "heading": content.strip(),
                            "metadata": self.extract_scene_metadata(content),
                            "content": [],
                        }

                    elif l_type == "CHARACTER":
                        flush_dialogue()
                        active_speaker = self.clean_text(content)

                    elif l_type == "DIALOGUE":
                        dialogue_buffer.append(self.clean_text(content))

                    elif l_type == "PARENTHETICAL":
                        if active_speaker:
                            dialogue_buffer.append(f"[{content.strip('()').lower()}]")

                    elif l_type == "ACTION":
                        flush_dialogue()
                        action = self.transform_observation(self.clean_text(content))
                        if action:
                            current_scene["content"].append(action)

        flush_dialogue()
        if current_scene["content"]:
            scenes.append(current_scene)
        return scenes


if __name__ == "__main__":
    parser = HannibalParser()
    # Paths...
    if os.path.exists(SCRIPTS_FOLDER):
        pdf_files = [f for f in os.listdir(SCRIPTS_FOLDER) if f.endswith(".pdf")]
        for filename in tqdm(pdf_files):
            data = parser.parse_pdf(os.path.join(SCRIPTS_FOLDER, filename))
            with open(
                os.path.join(
                    OUTPUT_FOLDER, os.path.splitext(filename)[0] + "_memory.json"
                ),
                "w",
                encoding="utf-8",
            ) as f:
                json.dump(data, f, indent=4)
