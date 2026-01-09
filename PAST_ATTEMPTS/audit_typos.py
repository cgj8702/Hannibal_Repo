import json
import glob
import re
import os

OUTPUT_FOLDER = os.path.join(os.getcwd(), "HANNIBAL_CHUNKS_OUTPUT")

def check_typos():
    files = glob.glob(os.path.join(OUTPUT_FOLDER, "*.json"))
    issues = []
    
    # regex for common pdf artifacts
    suspicious_patterns = [
        (r"\s+[.,?!]", "Space before punctuation"), # "Hello ."
        (r"\s+['’]\s+", "Floating apostrophe"), # "don ' t"
        (r"\b[a-zA-Z] [a-zA-Z]\b", "Single letter spaced (possible broken word)"), # "t h e" (heuristic)
        (r"[a-z][A-Z]", "CamelCase inside word (possible missing space)"), # "wordWord"
        (r"\.\.\.\.+", "Excessive dots"),
        (r"\s’", "Space before closing quote/apostrophe"), 
        (r"‘\s", "Space after opening quote/apostrophe")
    ]
    
    total_issues = 0
    
    for f in files:
        fname = os.path.basename(f)
        try:
            with open(f, "r", encoding="utf-8") as r:
                data = json.load(r)
                
            for i, scene in enumerate(data):
                content = scene.get("content", [])
                for j, line in enumerate(content):
                    for pattern, name in suspicious_patterns:
                        if re.search(pattern, line):
                            # Filter out false positives for "Single letter spaced"
                            if name.startswith("Single"):
                                # "A I" is valid (AI). "a b" is susp. 
                                # Let's be strict: detection of lower case isolated letters
                                if not re.search(r"\b[a-z] [a-z]\b", line):
                                     continue
                            
                            issues.append(f"{fname} (S{i}:L{j}): {name} -> '{line[:50]}...'")
                            total_issues += 1
                            if total_issues > 50: break
                if total_issues > 50: break
        except Exception as e:
            issues.append(f"Error reading {fname}: {e}")
            
    if issues:
        print(f"Found {total_issues}+ potential text quality issues:")
        for issue in issues[:50]:
            print(issue)
        if total_issues > 50: print("... truncated ...")
    else:
        print("No suspicious text patterns found.")

if __name__ == "__main__":
    check_typos()
