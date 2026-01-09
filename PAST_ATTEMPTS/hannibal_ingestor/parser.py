import re
import os
import pdfplumber
from .constants import WHOLE_LINE_KILLERS, DYNAMIC_KILLERS, PROMOTION_BLACKLIST
from .utils import clean_text, transform_observation, clean_scene_content

class HannibalParser:
    def __init__(self):
        self.WHOLE_LINE_KILLERS = WHOLE_LINE_KILLERS
        self.DYNAMIC_KILLERS = DYNAMIC_KILLERS
        self.PROMOTION_BLACKLIST = PROMOTION_BLACKLIST

    def clean_text(self, text):
        return clean_text(text)

    def transform_observation(self, text):
        return transform_observation(text)

    def clean_scene_content(self, content_list):
        return clean_scene_content(content_list)

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
            if re.search(rf"\\b{t}\\b", heading_upper):
                meta["time"] = t
                break

        if "POV" in heading_upper or "POINT OF VIEW" in heading_upper or "THROUGH THE EYES" in heading_upper:
            meta["pov"] = "Specific POV"
        else:
            meta["pov"] = None

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
        if any(x in clean for x in ["INT.", "EXT.", "I/E", "UNDERWATER", "REALITY", "IN HIS MIND", "IN THE", "UNDEFINED LOCATION"]):
            return "SLUGLINE", clean
        
        # Numbered Scene Headers
        if clean.isupper() and re.match(r"^\d+\s+[A-Z]", clean):
             return "SLUGLINE", clean

        # 2. CHARACTER NAMES
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
        episode_id = os.path.basename(pdf_path).replace(".pdf", "")
        
        default_meta = {"is_dream": False, "is_flashback": False, "time": "UNKNOWN", "location": "UNKNOWN", "episode": episode_id, "pov": None, "characters_present": []}
        current_scene = {"heading": "START", "metadata": default_meta, "content": []}
        active_speaker = None
        dialogue_buffer = []

        def flush_dialogue():
            nonlocal active_speaker, dialogue_buffer
            if active_speaker and dialogue_buffer:
                speech_raw = " ".join(dialogue_buffer)
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
                            # ChromaDB Requirement: Flatten list to single string output
                            current_scene["content"] = "\n".join(current_scene["content"])
                            scenes.append(current_scene)
                        
                        clean_heading = re.sub(r"^[A-Z]*\d+[A-Z]*\s+", "", content).strip()
                        clean_heading = re.sub(r"\s+[A-Z]*\d+[A-Z]*$", "", clean_heading).strip()
                        
                        new_meta = self.extract_scene_metadata(clean_heading)
                        new_meta["episode"] = episode_id
                        current_scene = {"heading": clean_heading, "metadata": new_meta, "content": []}
                    
                    elif l_type == "CHARACTER":
                        flush_dialogue()
                        active_speaker = self.clean_text(content)
                        if active_speaker and active_speaker not in current_scene["metadata"]["characters_present"]:
                            current_scene["metadata"]["characters_present"].append(active_speaker)
                        
                    elif l_type == "DIALOGUE":
                        clean_line = self.clean_text(content)
                        if active_speaker:
                            dialogue_buffer.append(clean_line)
                        else:
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
