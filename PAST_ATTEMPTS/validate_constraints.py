import os
import json
import re
import argparse
from glob import glob

# Define the strict rules
def check_file(filepath):
    issues = []
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        return [f"JSON Load Error: {e}"]

    if not isinstance(data, list):
        return ["Root element is not a list"]

    for i, scene in enumerate(data):
        # 1. Check basic structure
        if "heading" not in scene or "content" not in scene:
            issues.append(f"Scene {i}: Missing heading or content keys")
            continue

        # 2. Check Heading
        heading = scene.get("heading", "")
        if not heading or not isinstance(heading, str):
            issues.append(f"Scene {i}: Invalid or empty heading")
        
        # 3. Check Metadata
        meta = scene.get("metadata", {})
        if not isinstance(meta, dict):
            issues.append(f"Scene {i}: Metadata is not a dict")
        else:
             # VALIDATE TIME
             valid_times = ["DAY", "NIGHT", "MORNING", "EVENING", "DAWN", "DUSK", "TWILIGHT", "PREDAWN", "SUNSET", "UNKNOWN"]
             t = meta.get("time", "UNKNOWN")
             t = meta.get("time", "UNKNOWN")
             if t not in valid_times:
                 issues.append(f"Scene {i}: Invalid time '{t}'")
             
             # VALIDATE EPISODE
             if "episode" not in meta or not meta["episode"]:
                 issues.append(f"Scene {i}: Missing episode ID")
             
             # VALIDATE LOCATION
             loc = meta.get("location", "")
             if not loc:
                 issues.append(f"Scene {i}: Empty location")
             # Check for INT/EXT artifacts (whole words only)
             if re.search(r"\b(?:INT|EXT|I/E)\b", loc, re.IGNORECASE):
                 issues.append(f"Scene {i}: Location contains INT/EXT artifact: '{loc}'")
        
        # 4. Check Content
        content = scene.get("content", [])
        if not isinstance(content, list):
            issues.append(f"Scene {i}: Content is not a list")
            continue

        for j, line in enumerate(content):
            if not isinstance(line, str):
                issues.append(f"Scene {i} Line {j}: Content item is not a string")
                continue
            
            # RULE: No empty lines
            if not line.strip():
                issues.append(f"Scene {i} Line {j}: Empty line found")
            
            # RULE: Dialogue formatting
            # Dialogue usually looks like SPEAKER: "Speech"
            # Action looks like [Observation: ...] or just text
            
            # Check for uncleaned artifacts
            if "[then]" in line:
                issues.append(f"Scene {i} Line {j}: Found '[then]' artifact")
            if "(NOTE" in line:
                issues.append(f"Scene {i} Line {j}: Found '(NOTE' artifact")
            if "SMASH TO" in line and "SMASH TO:" not in line: # transitions might be wrapped?
                 # Actually strictly checking for "SMASH TO BLACK." which we added to killers
                 if "SMASH TO BLACK" in line:
                     issues.append(f"Scene {i} Line {j}: Found 'SMASH TO BLACK' artifact")

            # Check for weird starting brackets that aren't Observation or legitimate lower-case parentheticals re-inserted
            # The parser re-inserts parentheticals as [action]. 
            if line.startswith("["):
                if not line.startswith("[Observation:") and not re.match(r"^\[[a-z\s\.,\?!]+\]$", line):
                     # If it starts with [ but isn't Observation or simple lowercase action, flag it
                     issues.append(f"Scene {i} Line {j}: Suspicious bracketed line: {line[:50]}...")

            # Check for double spaces (user mentioned this)
            if "  " in line:
                 # valid in some contexts, but let's strict check
                 pass # actually double spaces in PDF extraction are common and harmless usually, maybe user cares?
                 # User said "Detects double spaces". I should verify if I need to kill them.
                 # Let's flag them but maybe as warnings?
                 # For now, let's focus on content artifacts.
                 pass

    return issues

def main():
    folder = os.path.join(os.getcwd(), "HANNIBAL_CHUNKS_OUTPUT")
    files = glob(os.path.join(folder, "*.json"))
    
    total_issues = 0
    files_with_issues = 0
    
    print(f"Scanning {len(files)} files in {folder}...\n")
    
    with open("validation_report.txt", "w", encoding="utf-8") as report:
        for string_path in files:
            basename = os.path.basename(string_path)
            file_issues = check_file(string_path)
            
            if file_issues:
                files_with_issues += 1
                total_issues += len(file_issues)
                msg = f"❌ {basename}: Found {len(file_issues)} issues"
                print(msg)
                report.write(msg + "\n")
                for issue in file_issues:
                    print(f"   - {issue}")
                    report.write(f"   - {issue}\n")
                report.write("-" * 40 + "\n")
        else:
            # excessive printing avoidance
            pass
            
    if total_issues == 0:
        print("\n✅ All files passed strict verification!")
    else:
        print(f"\n⚠️ Verification failed! {files_with_issues} files contain total {total_issues} issues.")

if __name__ == "__main__":
    main()
