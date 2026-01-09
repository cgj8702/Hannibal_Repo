import os
import argparse
import sys
import json
import concurrent.futures
from tqdm import tqdm

# Add parent directory to path so we can import the package
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hannibal_ingestor.parser import HannibalParser

# Configuration
SCRIPTS_FOLDER = "C:/Users/carly/Documents/Coding/episode_scripts/"
OUTPUT_FOLDER = os.path.join(os.getcwd(), "HANNIBAL_CHUNKS_OUTPUT")
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

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
    parser = argparse.ArgumentParser(description="Parse Hannibal Scripts (Refactored)")
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
            
    print("Ingestion Complete.")
