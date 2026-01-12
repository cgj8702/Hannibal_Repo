
import os
import difflib
import sys
import glob

PARSED_DIR = "ParsedScreenplays"
PROOFREAD_DIR = "Final_Proofread_Screenplays_V2"

def compare_file(basename):
    original_path = os.path.join(PARSED_DIR, basename)
    import re
    
    # Identify Ep Code
    # Assuming basename is Hannibal_1x01_Aperitif.txt
    ep_match = re.search(r'(\d+x\d+)', basename)
    proofread_path = None
    
    if ep_match:
        ep_code = ep_match.group(1)
        # Look for matching file in V2 dir
        candidates = glob.glob(os.path.join(PROOFREAD_DIR, f"*{ep_code}*V2.txt"))
        if candidates:
            proofread_path = candidates[0]
            
    if not proofread_path or not os.path.exists(proofread_path):
        proofread_filename = f"Proofread_{basename}"
        proofread_path = os.path.join(PROOFREAD_DIR, proofread_filename)
    
    if not os.path.exists(original_path):
        print(f"Original file not found: {original_path}")
        return
        
    if not os.path.exists(proofread_path):
        print(f"Proofread file not found: {proofread_path}")
        return
        
    print(f"--- Comparing {basename} ---")
    
    with open(original_path, 'r', encoding='utf-8', errors='replace') as f:
        original_lines = f.readlines()
        
    with open(proofread_path, 'r', encoding='utf-8', errors='replace') as f:
        proofread_lines = f.readlines()
        
    # Stats
    print(f"Original Lines: {len(original_lines)}")
    print(f"Proofread Lines: {len(proofread_lines)}")
    
    # Diff
    diff = difflib.unified_diff(
        original_lines, 
        proofread_lines, 
        fromfile=f'Original_{basename}', 
        tofile=f'Proofread_{basename}',
        n=0 # Minimal context to verify changes
    )
    
    # Calculate crude change metrics
    added = 0
    removed = 0
    
    # Only print first N diffs to avoid spam
    print_limit = 20
    printed = 0
    
    for line in diff:
        if line.startswith('+') and not line.startswith('+++'):
            added += 1
            if printed < print_limit:
                 print(line.strip()[:100]) # Truncate long lines
                 printed += 1
        elif line.startswith('-') and not line.startswith('---'):
            removed += 1
            if printed < print_limit:
                 print(line.strip()[:100])
                 printed += 1
                 
    print(f"\nTotal Added Lines: {added}")
    print(f"Total Removed Lines: {removed}")
    print("-" * 30)

if __name__ == "__main__":
    if len(sys.argv) > 1:
        # Compare specific file from args
        compare_file(sys.argv[1])
    else:
        # Default test set: 1x01 (success case) and 1x06 (failure case)
        compare_file("Hannibal_1x01_Aperitif.txt")
        compare_file("Hannibal_1x06_Entree.txt")
