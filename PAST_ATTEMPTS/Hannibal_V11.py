import re
import os
import json
import argparse
import pdfplumber
import concurrent.futures
from tqdm import tqdm

# --- CONFIGURATION ---
SCRIPTS_FOLDER = "C:/Users/carly/Documents/Coding/episode_scripts/"
OUTPUT_FOLDER = os.path.join(os.getcwd(), "HANNIBAL_CHUNKS_OUTPUT")
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

class HannibalParser:
    def __init__(self):
        # ==========================================
        # 1. THE "NUKE" LIST (Whole Line Killers)
        # ==========================================
        self.WHOLE_LINE_KILLERS = [
            "Hannibal", "TEASER", "ACT ONE", "ACT TWO", "ACT THREE", "ACT FOUR", "ACT FIVE", "ACT SIX",
            "END OF ACT ONE", "END OF ACT TWO", "END OF ACT THREE", "END OF ACT FOUR", "END OF ACT FIVE",
            "END OF TEASER", "FADE IN:", "FADE OUT.", "FADE TO BLACK.", "PROLOGUE", "CUT TO:",
            "FADE TO:", "DISSOLVE TO:", "SMASH TO:", "SMASH TO BLACK.", "MATCH CUT TO:", "TIME CUT TO:", "JUMP CUT TO:",
            "CONTINUED", "OMITTED", "END OF SHOW", "END OF ACT", "END OF EPISODE", "THE END", "FINAL SHOOTING SCRIPT",
            "FINAL DRAFT", "Collated", "WHITE REVISIONS", "BLUE", "PINK", "YELLOW",
            "GREEN", "GOLDENROD", "WHITE", "REVERSE ANGLE", "CLOSE-UP", "NEW ANGLE",
            "HIGH ANGLE", "POP WIDE", "ON THE FLOOR", "POST CREDITS", "END OF SEASON THREE!",
            "A PIXELATED DARKNESS", "THE TELEPHONE MATRIX", "THE MOUTHPIECE",
            "A STILL OF WILL GRAHAM", "A SIGN", "UTTER DARKNESS", "A RECORD NEEDLE",
            "A RIBBON OF CELLULOID", "A SPINNING SPROCKET WHEEL", "A RADIANT VISION",
            "A FULL MOON", "A HALF MOON", "A MOON-SHAPED HOLE", "A SMASHED MIRROR",
            "A TRANSPARENT VESSEL", "THE DRAGON'S WINGS", "A CAGE",
            "Written by", "Created by", "Directed by", "Story by", "Teleplay by", "Based on the characters", 
            "Based on the novel", "Production Draft", "Shooting Script",
            "Executive Producer", "Executive Producers", "PRODUCED BY", "Co-Executive Producer",
            "Associate Producer", "Consulting Producer", "Staff Writer",
            "PROPERTY OF:", "GAUMONT", "CHISWICK", "ALL RIGHTS RESERVED",
            "NO PORTIONS OF THIS SCRIPT", "PERFORMED, OR REPRODUCED", "WRITTEN CONSENT",
            "SCRIPT MAY BE", "QUOTED, OR PUBLISHED"
        ]

        # ==========================================
        # 2. DYNAMIC KILLERS (Regex)
        # ==========================================
        self.DYNAMIC_KILLERS = [
            # S3 specific mechanical/primal sounds
            r"^\s*(?:FWUM|CLICK|BLAM|WHOOSH|CRASH|BANG|SMASH|KLANG|K’CHUK|RING|PLIP|WHAM|P’KEE|PFFT|KLAXON|BUZZ|P'KEET|SHLUCK|PWUM|PLOP|ZAP|SHUNK|FWOOM|SCRITCH|KA-CHOO|BING-BONG|K-SHH+|FWUMMP|THWUB|KERSHICK|SLKGHHH|WHACK|THUNK|BARP|P-KEE|CH-CHUNK|WOOMPF)(?:\.|\s)*$",
            # Repeating SFX patterns
            r"^\s*(?:SLAM|CHOP|PLIP|PWUM|FWUP|BANG|THWUB|BLEEP|EEK)(?:[-\.\s]+(?:SLAM|CHOP|PLIP|PWUM|FWUP|BANG|THWUB|BLEEP|EEK))+\s*$",
            r"^\s*FADE (?:IN|OUT|TO|THROUGH):?",
            r"^\s*(?:CUT|SMASH CUT|BACK|MATCH CUT|TIME CUT|QUICK POP|SMASH BACK|HARD CUT|FLASH CUT|QUICK MATCH CUT|SWIPE) TO:?",
            r"^\s*DISSOLVE TO:?",
            r"^\s*(?:WIDENING|REVERSING|PUSHING|PULLING|PANNING|TILTING|TRACKING|CRANE|DOLLY)\b",
            # Footer filtering
            r"HANNIBAL\s+Ep\.\s+#\d+.*",
            r"\d{2}/\d{2}/\d{2}\s+\d+\.?\s*$",
            r"^\s*\d+\.\s*$",
            r"^\d+\.$",  # Page numbers like "1."
            r"^Hannibal\s+#", # Header junk "Hannibal #303"
            r"\d{1,2}/\d{1,2}/\d{2,4}", # Dates
            r"^\(CONTINUED\)$",
            r"^\s*©", r"^\s*\(c\)", r"^\s*\u00a9",
            r"^\s*[0-9]+$", # Isolated page numbers
            r"^\s*Bryan Fuller", r"^\s*Thomas Harris", r"^\s*David Slade", r"^\s*Steve Lightfoot",
            r"^\s*Tim Hunter", r"^\s*Guillermo Navarro", r"^\s*Michael Rymer",
            r"^\s*Chris Brancato", r"^\s*Jesse Alexander", r"^\s*Martha De Laurentiis"
        ]

        # ==========================================
        # 3. THE "SNIPER" LIST (Partial Removers)
        # ==========================================
        self.PARTIAL_REMOVERS = [
            "(CONT'D)", "(CONT’D)", "(cont'd)", "(cont’d)", "(V.O.)", "(O.S.)", "(O.C.)",
            "(pre-lap)", "(into phone)", "(then)", "(MORE)", "(stunned)", "(whispers)",
            "(calling out)", "(intense)", "(pointing)", "(distorted, dreamlike)",
            "(re: the food)", "(re: the herbs)", "(re: the picture)", "(re: the lens)",
            "(re: the pigs)", "(re: the lunchbox)", "(re: the thermos)",
            "(not bitchy)", "(smiles and it's lovely)", "(almost inaudible)", "...", 
            "we --", "We are--", "we are--", "we are --", "We are --",
            "A CHYRON tells us", "A CHYRON", "RE-USE EP 101", "PRESENT", "FLASHBACK",
            "OMNISCIENT P.O.V.", "WILL’S P.O.V.", "WILL'S P.O.V.", "GEORGIA’S P.O.V.",
            "LONE MAN’S P.O.V.", "ABIGAIL’S P.O.V.", "JACK'S P.O.V.", "WILL’S FEVERISH P.O.V.",
            "ALANA’S POV", "INTERCUT WITH:", "WILL IS HALLUCINATING.", "STOCK FOOTAGE",
            "WILL GRAHAM'S BED", "THE SHADOWY TREES", "FIRELIGHT", "FLASH FORWARD",
            "A PENDULUM", "RETRO-WILL", "RETRO-HANNIBAL", "NOW-WILL", "THE MIRROR",
            "REALITY", "DREAM STATE", "MATCH TO:", "TRANSITION TO:", "SWIPE TO BLACK."
        ]

        self.PROMOTION_BLACKLIST = [
            "HOBBS", "GRAHAM", "LECTER", "CRAWFORD", "BLOOM", "KATZ", "ZELLER", "PRICE", "LOUNDS",
            "GIDEON", "CHILTON", "LASS", "SUTCLIFFE", "MADCHEN", "BOYLE", "BUDISH", "SUMMERS",
            "CARRUTHERS", "SHELL", "STAG", "PENDULUM", "WINSTON", "DOE", "PILLS", "LABEL", "GUN",
            "REVEAL", "CAMERA", "CLOSE ON", "ON WILL", "ON HANNIBAL", "ON JACK", "ON ALANA",
            "A GLASS OF WATER", "THE ALARM CLOCK", "A PICTURE", "THE MORGUE",
            "A SEA URCHIN", "A BEAUTIFUL PIECE", "A TASTEFULLY-ORNATE", "A LARYNGOSCOPIC VIEW",
            "THE STRING SECTION", "THE MISSING INTESTINES", "A KITCHEN CENTRIFUGE",
            "A SNOWY SKY", "A HORRIBLE EYE", "A PAIR OF EYES", "THE BED OVERTURNS",
            "A RAINBOW TROUT", "UNDER THE BED", "A ROW OF CORN", "THE HUMAN MURAL",
            "THE MISSING LEG", "A BEAUTIFUL PLATE", "A SUDDEN PIERCING", "A SLOW SWINGING",
            "THE WENDIGO MAN STAG", "A CUT-GLASS PERFUME", "A BOILING FROTH", "A MAN'S ANGUISHED EYE",
            "A MAN'S FACE", "A PANEL TRUCK", "A SUDDEN LOUD BARKING", "A RIVULET OF BLOOD",
            "A BOOK COVER", "A CAR'S FRONT WHEEL", "A HONEY BEE", "A MAN'S CORPSE",
            "A SMOKE-BLACKENED FACE", "A BLUE SPARK", "A SOFT GOLDEN GLEAM", "A PADDED ENVELOPE",
            "THE BUTCHER BLOCK", "THE MAIN TELESCOPE", "THE REVERSE SIDE", "THE LABYRINTH",
            "A HARD, SOMBER CHORD", "A SOMBER REFRAIN", "A RECENTLY-DUG GRAVE", "A LONG BONE",
            "A SHEET-COVERED BODY", "A PEN SCRAWLS", "A HARPSICHORD NOTE", "A HUGE BONFIRE",
            "A HUNCHED, POWERFUL SILHOUETTE", "SPLAYED PAWS", "A FLASH OF CLAWS", "A FLASH OF FANGS",
            "A DISTORTED MAN", "A SINGLE BRIGHT OBJECT", "A DISEMBODIED VOICE", "A FANGED RESIN MUZZLE",
            "THE DIRE WOLF CRANIUM", "A SHADOW GROWS FAST", "A RIFLE", "A SHEATHED FILLET KNIFE",
            "A WAX PAPER-WRAPPED PACKAGE", "THE SUCKLING PIG", "AN INDUSTRIAL FREEZER",
            "A DESK", "A RACK OF LARGE GLASS", "A ROBIN", "A RAT", "AN INTENSE YOUNG MAN",
            "A CAR'S FRONT WHEEL", "A PARTY CROWD", "A BEAUTIFUL WOMAN", "A RUSHING TAP",
            "AN EMPTY CHAIR", "A POLICE CONSTABLE", "A CCTV CAMERA", "A DYING MAN",
            "A MAN IN HER SHOWER", "A TRUCK'S FRONT WHEEL", "A FALLING TEACUP", "SNOW",
            "SPOTS OF BLOOD", "THE WILDIGO", "A DOOR", "SHARDS OF PORCELAIN", "A WHITE SHEET",
            "THE BODY'S BACK", "THE DOCTOR'S NEEDLE TIP", "A FINAL LENGTH OF THREAD",
            "DUST MOTES", "CASTLE LECTER", "MISCHA LECTER", "A SEVERED ARM", "A HANDFUL OF SALT",
            "A PRECISION KITCHEN BLADE", "A PLATTER OF SKEWERS", "A BOTTLE OF WINE",
            "A SCOLD'S BRIDLE", "A BODY OF WATER", "A SNAIL", "A BURLAP SACK", "PHEASANT MEAT",
            "A CONTACT LENS", "A LAYER OF MAKEUP", "A SAUCER OF BLOOD", "A WINE BOTTLE",
            "A WINEGLASS", "A TRANSPARENT VESSEL", "A FULL MOON", "A HALF MOON",
            "A MOON-SHAPED HOLE", "A GLASS CUTTER", "A SMASHED MIRROR", "A BATHROOM GLASS",
            "A TELEPHONE", "A TELEPHONE RECEIVER", "TRAIN TRACKS", "TRAIN CARS",
            "A FIREFLY LARVA", "A RIPPLE OF WATER", "THE COLLAR", "THE TIE",
            "A BUBBLING SOUP POT", "A SCRIM OF CHEESECLOTH", "A CLEAR ASPIC", "A PENCIL",
            "A PENCIL SKETCH", "A STREAM OF COFFEE", "THE WOODCARVING", "A STRIP OF LIGHT",
            "A VENTILATOR", "THE EEL", "A TRUCK'S FRONT WHEEL", "A SAUCER OF BLOOD",
            "A MEDICINE VIAL", "AN EXTERNAL FIXATION", "A METAL SIGN", "THE NIGHT SKY",
            "A PIXELATED DARKNESS", "THE MOUTHPIECE", "A STILL OF WILL", "A VASE",
            "A PROJECTOR LENS", "A SCREEN", "A COLORLESS VISTA", "THE YELLOW PADDED ENVELOPE",
            "A SINGLE LIP", "THE CITY FOUNTAIN", "A RIBBON OF CELLULOID", "A SPINNING SPROCKET",
            "A MATCH", "A BURNING MATCH HEAD", "A BODY HOIST", "A BLACK WOMAN", "A SHADOW",
            "A MASK", "A VEHICLE'S ENGINE", "THE CLOCK", "A BOTTLE OF AMMONIA", "A TRIPOD",
            "THE VIDEO CAMERA", "A COFFIN", "A FUNERAL DIRECTOR", "A BUCK KNIFE BLADE",
            "THE MAJESTIC TATTOO", "THE ALTAR", "A LONE DIVA", "A LURID NEW YORK POST",
            "A CHUNK OF MASONRY", "A SAUCER OF BLOOD", "A GREAT LEDGER", "A CLIPPING"
        ]

    def clean_text(self, text):
        """Standardizes text and removes dialogue artifacts including S3 specific notes."""
        # Normalize smart quotes and apostrophes
        text = text.replace("\u201c", '"').replace("\u201d", '"')
        text = text.replace("\u2018", "'").replace("\u2019", "'")
        
        # Normalize punctuation and spacing
        text = re.sub(r"\s+([.,?!:;])", r"\1", text) # Remove space before punct
        text = re.sub(r"\s+'\s+", "'", text) # Remove floating apostrophes
        text = re.sub(r"'\s+", "'", text) # Remove space after apostrophe
        text = re.sub(r"\s+'", "'", text) # Remove space before apostrophe
        text = re.sub(r"\.\s\.\s\.", "...", text) # Fix spaced ellipsis
        
        # Fix common pdf extraction ligatures/spacing
        text = text.replace("f i", "fi").replace("f l", "fl").replace("t h", "th")
        
        # Split specific common merged words (safer list). Expanded based on manual proofread.
        # e.g. "puther" -> "put her", "methim" -> "met him", "couldn'thave" -> "couldn't have", "thoughthe" -> "thought he"
        common_starts = r"(but|and|that|what|this|about|let|if|put|cut|out|got|met|set|at|of|in|on|for|with|by|to|do|go|are|was|were|be|been|have|has|had|can|will|would|could|should|did|does|don't|won't|can't|couldn't|wouldn't|shouldn't|didn't|doesn't|isn't|aren't|wasn't|weren't|haven't|hasn't|hadn't|myself|himself|herself|yourself|themselves|ourselves|thought|think|said|says|ask|asked|tell|told|know|knew|see|saw|seen|look|looked|give|gave|take|took|come|came|just|well|like|make|made|where|when|why|who|how|which|then|than|from)"
        common_ends = r"(he's|he|she|it|is|in|if|as|at|him|her|his|have|has|had|an|us|we|my|go)"
        text = re.sub(rf"\b{common_starts}{common_ends}\b", r"\1 \2", text, flags=re.IGNORECASE)

        # Fix broken word spacing (e.g. "t h e" -> "the")
        # Pattern: sequence of 3+ single letters separated by space
        # We target specific common words to avoid false positives (e.g. "a b c")
        text = re.sub(r"\b([a-z])\s+([a-z])\s+([a-z])\b", r"\1\2\3", text) 
        text = re.sub(r"\b([a-z])\s+([a-z])\s+([a-z])\s+([a-z])\b", r"\1\2\3\4", text)
        
        # Case-insensitive removal of partial removers
        
        # Case-insensitive removal of partial removers
        for item in self.PARTIAL_REMOVERS:
            text = re.sub(re.escape(item), "", text, flags=re.IGNORECASE)
            
        # Remove bracketed/parenthesized notes
        text = re.sub(r"\[.*?\]", "", text)
        text = re.sub(r"\((?:NOTE|subtitled|on screen|louder|croaking|into phone|sucks teeth|preparing).*?\)", "", text, flags=re.IGNORECASE)
        # Force remove any remaining (NOTE...) patterns that might have weird spacing or specific content not caught above
        text = re.sub(r"\(NOTE.*?\)", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\(\s*\)", "", text)
        return text.strip()

    def transform_observation(self, text):
        """Converts camera/focus cues into narrative observations."""
        match = re.search(r"^(?:OFF|ON)\s+([A-Z0-9\s\-\'’\.!]+)(.*)", text, re.IGNORECASE)
        if match:
            target = match.group(1).strip().title()
            desc = match.group(2).strip().strip(",.- ")
            if not desc:
                return f"[Observation: Focus on {target}]"
            return f"[Observation: {target} reacts: {desc}]"
        
        if text.upper().startswith("REVEAL"):
            return "We see " + text[6:].strip()
        if text.upper().startswith("REVEAL"):
            return "We see " + text[6:].strip()
        return text

    def clean_scene_content(self, content_list):
        cleaned = []
        in_note = False
        for line in content_list:
            if in_note:
                if ")" in line:
                    in_note = False
                    parts = line.split(")", 1)
                    if len(parts) > 1 and parts[1].strip():
                        cleaned.append(parts[1].strip())
                continue
            
            match = re.search(r"\(NOTE", line, re.IGNORECASE)
            if match:
                if ")" in line:
                    line = re.sub(r"\(NOTE.*?\)", "", line, flags=re.IGNORECASE).strip()
                    if line: cleaned.append(line)
                else:
                    in_note = True
                    pre_note = line[:match.start()].strip()
                    if pre_note: cleaned.append(pre_note)
                continue
            cleaned.append(line)
        return cleaned

    def extract_scene_metadata(self, heading):
        """Extracts location, time, and mental state from sluglines."""
        meta = {"is_dream": False, "is_flashback": False, "time": "UNKNOWN", "location": "UNKNOWN", "pov": None, "characters_present": []}
        heading_upper = heading.upper()

        if any(x in heading_upper for x in ["DREAM", "NIGHTMARE", "HALLUCINATION", "DREAMSCAPE", "FEVERISH", "IN HIS MIND"]):
            meta["is_dream"] = True
        if any(x in heading_upper for x in ["FLASHBACK", "MEMORY", "EARLIER", "YEARS LATER"]):
            meta["is_flashback"] = True

        times = ["DAY", "NIGHT", "MORNING", "EVENING", "DAWN", "DUSK", "TWILIGHT", "PREDAWN", "SUNSET"]
        for t in times:
            if re.search(rf"\b{t}\b", heading_upper):
                meta["time"] = t
                break

        if "POV" in heading_upper or "POINT OF VIEW" in heading_upper or "THROUGH THE EYES" in heading_upper:
            meta["pov"] = "Specific POV"
        else:
            meta["pov"] = None

        meta["characters_present"] = []

        loc = re.sub(r"^[A-Z]*\d+[A-Z]*\s+", "", heading_upper) 
        loc = re.sub(r"\s+[A-Z]*\d+[A-Z]*\.?$", "", loc)
        loc = re.sub(r"^(?:INT\.|EXT\.|I/E|INT/EXT)\s+", "", loc)
        
        for noise in ["- DAY", "- NIGHT", "- MORNING", "- EVENING", "RESUMING", "CONTINUOUS", "PRESENT", "FLASHBACK", "TWILIGHT", "THE NEXT MORNING", "NEXT DAY", "LATER", "SAME TIME", "MOMENTS LATER"]:
            loc = loc.replace(noise, "")
        
        meta["location"] = loc.strip(" -")
        return meta

    def get_line_type(self, raw_line):
        """Determines line type via indentation and keywords."""
        clean = raw_line.strip()
        if not clean: return "EMPTY", ""
        leading = len(raw_line) - len(raw_line.lstrip())

        # 1. SLUGLINES (Prioritize to catch indented scene headers)
        # Standard INT./EXT. headers
        if any(x in clean for x in ["INT.", "EXT.", "I/E", "UNDERWATER", "REALITY", "IN HIS MIND", "IN THE", "UNDEFINED LOCATION"]):
            return "SLUGLINE", clean
        
        # Numbered Scene Headers (e.g. "13 ELISE NICHOLS' BEDROOM")
        # Pattern: Leading number, whitespace, uppercase text, optional trailing number
        # Must be uppercase to avoid numbered dialogue or lists
        if clean.isupper() and re.match(r"^\d+\s+[A-Z]", clean):
             return "SLUGLINE", clean

        # 2. CHARACTER NAMES (Handles S3 NOW/RETRO prefixes)
        if leading >= 25 and clean.isupper() and not clean.startswith("("):
            if not any(clean.startswith(x) for x in ["A ", "THE ", "EXTREME ", "AN "]):
                name_base = re.sub(r"^(?:RETRO|NOW)-", "", clean)
                if name_base not in self.PROMOTION_BLACKLIST and len(clean.split()) <= 4:
                    return "CHARACTER", re.sub(r"\(.*?\)", "", clean).strip()

        # 3. DIALOGUE & PARENTHETICALS
        if 8 <= leading < 30:
            if clean.startswith("(") and clean.endswith(")"):
                return "PARENTHETICAL", clean
            return "DIALOGUE", clean

        return "ACTION", clean

    def is_junk_line(self, text):
        clean = text.strip()
        if any(k in clean for k in self.WHOLE_LINE_KILLERS): return True
        for pattern in self.DYNAMIC_KILLERS:
            if re.search(pattern, text, re.IGNORECASE): return True
        return False

    def parse_pdf(self, pdf_path):
        scenes = []
        # Extract episode ID from filename (e.g. "Hannibal_1x01_Aperitif")
        episode_id = os.path.basename(pdf_path).replace(".pdf", "")
        
        # Initialize START scene with valid defaults to pass validation
        default_meta = {"is_dream": False, "is_flashback": False, "time": "UNKNOWN", "location": "UNKNOWN", "episode": episode_id, "pov": None, "characters_present": []}
        current_scene = {"heading": "START", "metadata": default_meta, "content": []}
        active_speaker = None
        dialogue_buffer = []

        def flush_dialogue():
            nonlocal active_speaker, dialogue_buffer
            if active_speaker and dialogue_buffer:
                speech_raw = " ".join(dialogue_buffer)
                # Clean again on the full block to catch multi-line artifacts like (NOTE...)
                speech = self.clean_text(speech_raw).strip()
                if speech:
                    current_scene["content"].append(f"{active_speaker}: \"{speech}\"")
            active_speaker = None
            dialogue_buffer = []

        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                text_content = page.extract_text(layout=True)
                if not text_content: continue
                
                for line in text_content.split("\n"):
                    if self.is_junk_line(line): continue
                    l_type, content = self.get_line_type(line)
                    
                    if l_type == "SLUGLINE":
                        flush_dialogue()
                        current_scene["content"] = self.clean_scene_content(current_scene["content"])
                        if current_scene["content"]:
                            # ChromaDB Requirement: Flatten list to single string
                            current_scene["content"] = "\n".join(current_scene["content"])
                            scenes.append(current_scene)
                        
                        # Clean heading: Remove leading/trailing scene numbers (e.g. "1", "A25")
                        # Regex targets distinct alphanumeric blocks usually found in script margins
                        clean_heading = re.sub(r"^[A-Z]*\d+[A-Z]*\s+", "", content).strip()
                        clean_heading = re.sub(r"\s+[A-Z]*\d+[A-Z]*$", "", clean_heading).strip()
                        
                        new_meta = self.extract_scene_metadata(clean_heading)
                        new_meta["episode"] = episode_id
                        current_scene = {"heading": clean_heading, "metadata": new_meta, "content": []}
                    
                    elif l_type == "CHARACTER":
                        flush_dialogue()
                        active_speaker = self.clean_text(content)
                        # Add to duplicates check
                        if active_speaker and active_speaker not in current_scene["metadata"]["characters_present"]:
                            current_scene["metadata"]["characters_present"].append(active_speaker)
                        
                    elif l_type == "DIALOGUE":
                        clean_line = self.clean_text(content)
                        if active_speaker:
                            dialogue_buffer.append(clean_line)
                        else:
                            # Orphan dialogue (no speaker) - treat as Action/Text
                            # This handles indented title card text or misinterpreted Action
                            if clean_line:
                                current_scene["content"].append(clean_line)
                        
                    elif l_type == "PARENTHETICAL":
                        cleaned_paren = self.clean_text(content)
                        if active_speaker and cleaned_paren:
                            dialogue_buffer.append(f"[{cleaned_paren.strip('()').lower()}]")
                            
                    elif l_type == "ACTION":
                        flush_dialogue()
                        action = self.transform_observation(self.clean_text(content))
                        if action: current_scene["content"].append(action)

        flush_dialogue()
        current_scene["content"] = self.clean_scene_content(current_scene["content"])
        if current_scene["content"]: 
            # ChromaDB Requirement: Flatten list to single string output
            current_scene["content"] = "\n".join(current_scene["content"])
            scenes.append(current_scene)
        return scenes

def process_file(filename):
    parser = HannibalParser()
    pdf_path = os.path.join(SCRIPTS_FOLDER, filename)
    try:
        data = parser.parse_pdf(pdf_path)
        output_filename = os.path.join(OUTPUT_FOLDER, os.path.splitext(filename)[0] + "_memory.json")
        with open(output_filename, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4)
        return filename
    except Exception as e:
        print(f"Error processing {filename}: {e}")
        return None

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parse Hannibal Scripts")
    parser.add_argument("--file", type=str, help="Specific PDF filename to parse (e.g., Hannibal_1x01_Aperitif.pdf)")
    args = parser.parse_args()

    if os.path.exists(SCRIPTS_FOLDER):
        if args.file:
             pdf_files = [args.file]
             print(f"Targeting single file: {args.file}")
        else:
             pdf_files = [f for f in os.listdir(SCRIPTS_FOLDER) if f.endswith(".pdf")]
             print(f"Found {len(pdf_files)} PDF files to process.")
        
        # Use ProcessPoolExecutor to parallelize parsing
        with concurrent.futures.ProcessPoolExecutor() as executor:
            list(tqdm(executor.map(process_file, pdf_files), total=len(pdf_files)))