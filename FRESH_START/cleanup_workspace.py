
import os
import shutil
import glob

ARCHIVE_DIR = "_archive"
FILES_TO_MOVE = [
    "verify_gcp_access.py",
    "verify_gcp_access_v2.py",
    "gcp_verify_log.txt",
    "debug_extraction_1x06.py",
    "check_models.py",
    "combine_screenplays.py",
    "models_list.txt",
    "models_list_py.txt",
    "prepare_vertex_batch.py",
    "process_batch_results.py",
    "proofread_screenplays.py",
    "remove_first_page.py",
    "upload_to_gcs.py",
    "upload_v2_request.py",
    "batch_requests.jsonl",
    "Proofread_Screenplays.ipynb",
    "ProofreadScreenplays.zip",
    "ParsedScreenplays.zip",
    "debug_2x12.py",
    "debug_output_2x12.txt"
]

def cleanup():
    if not os.path.exists(ARCHIVE_DIR):
        os.makedirs(ARCHIVE_DIR)
        print(f"Created {ARCHIVE_DIR}")

    # Explicit list
    for filename in FILES_TO_MOVE:
        if os.path.exists(filename):
            try:
                shutil.move(filename, os.path.join(ARCHIVE_DIR, filename))
                print(f"Moved {filename}")
            except Exception as e:
                print(f"Error moving {filename}: {e}")
    
    # Pattern matching for the big V1 prediction file
    for filepath in glob.glob("prediction-model-*.jsonl"):
         try:
            basename = os.path.basename(filepath)
            shutil.move(filepath, os.path.join(ARCHIVE_DIR, basename))
            print(f"Moved {basename}")
         except Exception as e:
            print(f"Error moving {filepath}: {e}")

if __name__ == "__main__":
    cleanup()
