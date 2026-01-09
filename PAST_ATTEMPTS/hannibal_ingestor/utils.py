import re
from .constants import PARTIAL_REMOVERS

def clean_text(text):
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
    for item in PARTIAL_REMOVERS:
        text = re.sub(re.escape(item), "", text, flags=re.IGNORECASE)
        
    # Remove bracketed/parenthesized notes
    text = re.sub(r"\[.*?\]", "", text)
    text = re.sub(r"\((?:NOTE|subtitled|on screen|louder|croaking|into phone|sucks teeth|preparing).*?\)", "", text, flags=re.IGNORECASE)
    # Force remove any remaining (NOTE...) patterns that might have weird spacing or specific content not caught above
    text = re.sub(r"\(NOTE.*?\)", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\(\s*\)", "", text)
    return text.strip()

def transform_observation(text):
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
    return text

def clean_scene_content(content_list):
    """Standardizes list of content lines into a clean format."""
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
