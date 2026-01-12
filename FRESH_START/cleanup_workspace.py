
import os
import shutil
import glob

ARCHIVE_DIR = "_archive"
KEEP_FILES = [
    "parse_screenplays.py",
    "cleanup_workspace.py",
    "requirements.txt"
]
KEEP_DIRS = [
    "Screenplays",
    "ParsedScreenplays",
    "_archive",
    ".venv",
    ".git",
    ".gemini",
    ".agent" # Keep agent workflows if any
]

def cleanup():
    if not os.path.exists(ARCHIVE_DIR):
        os.makedirs(ARCHIVE_DIR)
        print(f"Created {ARCHIVE_DIR}")

    print("--- Starting Cleanup ---")
    
    # List all items in current directory
    all_items = os.listdir('.')
    
    for item in all_items:
        # Skip kept items
        if item in KEEP_FILES or item in KEEP_DIRS:
            print(f"Keeping: {item}")
            continue
            
        # Move others
        try:
            src = item
            dst = os.path.join(ARCHIVE_DIR, item)
            
            # If destination exists, handle collision
            if os.path.exists(dst):
                base, ext = os.path.splitext(item)
                dst = os.path.join(ARCHIVE_DIR, f"{base}_old{ext}")
            
            shutil.move(src, dst)
            print(f"Archived: {item}")
        except Exception as e:
            print(f"Error moving {item}: {e}")

    print("--- Cleanup Complete ---")

if __name__ == "__main__":
    cleanup()
