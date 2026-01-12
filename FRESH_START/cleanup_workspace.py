
import os
import shutil
import glob

ARCHIVE_DIR = "_archive"
KEEP_FILES = ["parse_screenplays.py", "cleanup_workspace.py", "requirements.txt"]

def cleanup():
    if not os.path.exists(ARCHIVE_DIR):
        os.makedirs(ARCHIVE_DIR)
        print(f"Created {ARCHIVE_DIR}")

    # Move all .py files NOT in KEEP_FILES
    all_py_files = glob.glob("*.py")
    
    count = 0
    for file_path in all_py_files:
        if file_path not in KEEP_FILES:
            try:
                shutil.move(file_path, os.path.join(ARCHIVE_DIR, file_path))
                print(f"Moved {file_path}")
                count += 1
            except Exception as e:
                print(f"Error moving {file_path}: {e}")
                
    # Also move specific log files or temps if needed, but user just said .py
    # Let's clean up check_json_keys.py if it wasn't caught (it is a .py)
    
    print(f"Cleanup complete. {count} files moved to {ARCHIVE_DIR}")

if __name__ == "__main__":
    cleanup()
