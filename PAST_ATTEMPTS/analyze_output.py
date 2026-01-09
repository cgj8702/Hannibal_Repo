import os
import json
import re

OUTPUT_FOLDER = os.path.join(os.getcwd(), "HANNIBAL_CHUNKS_OUTPUT")

def analyze_files():
    if not os.path.exists(OUTPUT_FOLDER):
        print(f"Output folder not found: {OUTPUT_FOLDER}")
        return

    issues = {
        "missed_sluglines": [],
        "transitions": [],
        "parentheticals": [],
        "brackets": [],
        "smart_quotes": [],
        "empty_content": [],
        "likely_page_numbers": [],
        "double_spaces": []
    }

    files = [f for f in os.listdir(OUTPUT_FOLDER) if f.endswith(".json")]
    
    for filename in files:
        filepath = os.path.join(OUTPUT_FOLDER, filename)
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            print(f"Error reading {filename}: {e}")
            continue

        for i, scene in enumerate(data):
            heading = scene.get("heading", "")
            content = scene.get("content", [])

            if not content:
                issues["empty_content"].append(f"{filename} - Scene {i}: {heading}")

            for j, line in enumerate(content):
                # Missed Sluglines
                if re.search(r"\b(?:INT\.|EXT\.|I/E)\b", line):
                    issues["missed_sluglines"].append(f"{filename} - Scene {i} Line {j}: {line[:50]}...")
                
                # Transitions
                if re.search(r"^(?:CUT|FADE|DISSOLVE|SMASH|MATCH) TO:?", line):
                    issues["transitions"].append(f"{filename} - Scene {i} Line {j}: {line[:50]}...")

                # Parentheticals
                if re.search(r"\((?:CONT'D|V\.O\.|O\.S\.|O\.C\.|NOTE|then)\)", line, re.IGNORECASE):
                    issues["parentheticals"].append(f"{filename} - Scene {i} Line {j}: {line[:50]}...")

                # Brackets (excluding [observation...])
                # We want to catch [then], [to will] etc if user considers them issues, 
                # but user approved removing [then]. 
                if "[" in line and "[Observation:" not in line:
                     issues["brackets"].append(f"{filename} - Scene {i} Line {j}: {line[:50]}...")

                # Smart Quotes
                if any(char in line for char in ["\u201c", "\u201d", "\u2018", "\u2019"]):
                    issues["smart_quotes"].append(f"{filename} - Scene {i} Line {j}: {line[:50]}...")
                
                # Likely Page Numbers / Header junk
                if re.match(r"^\d+\.?\s*$", line.strip()) or re.match(r"^Hannibal.*#\d+", line):
                    issues["likely_page_numbers"].append(f"{filename} - Scene {i} Line {j}: {line[:50]}...")

                # Double spaces (extraction artifacts)
                if "  " in line:
                    issues["double_spaces"].append(f"{filename} - Scene {i} Line {j}: ...{line[line.find('  ')-10:line.find('  ')+20]}...")

    # Report
    print(f"--- Analysis Report ({len(files)} files checked) ---")
    for issue_type, findings in issues.items():
        if findings:
            print(f"\n[{issue_type.upper()}] Found {len(findings)} instances:")
            for item in findings[:10]: # Limit output
                print(f"  - {item}")
            if len(findings) > 10:
                print(f"  ... and {len(findings) - 10} more.")
        else:
            print(f"\n[{issue_type.upper()}] CLEAN")

if __name__ == "__main__":
    analyze_files()
