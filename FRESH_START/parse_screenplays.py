import os
import re
import pdfplumber
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

# Configuration
# PDF_FOLDER will be determined in main to ensure it's correct
OUTPUT_FOLDER = "ParsedScreenplays"

def process_pdf(filename, pdf_folder, output_folder):
    """
    Process a single PDF file: extract text and save parsed content.
    Returns a message indicating success or failure.
    """
    pdf_path = os.path.join(pdf_folder, filename)
    txt_filename = os.path.splitext(filename)[0] + ".txt"
    output_path = os.path.join(output_folder, txt_filename)
    
    try:
        full_text = ""
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                text = page.extract_text(layout=True)
                if text:
                    full_text += text + "\n"
        
        # Split and process lines
        lines = full_text.split('\n')
        
        # Regex for Scene Headers with Scene Numbers
        # Expecting: (Optional Whitespace) (Digits) (Whitespace) (INT./EXT.) (Rest of Line)
        scene_regex = re.compile(r'^\s*(\d+)?\s*((?:INT\.|EXT\.).*)', re.IGNORECASE)
        
        parsed_lines = []
        
        for line in lines:
            match = scene_regex.match(line)
            if match:
                 # Found a scene header
                 scene_num = match.group(1) if match.group(1) else "?"
                 scene_text = match.group(2).strip()
                 
                 formatted_header = f"\n=== {scene_num} {scene_text} ===\n"
                 parsed_lines.append(formatted_header)
            else:
                parsed_lines.append(line)
        
        # Write to file
        with open(output_path, "w", encoding="utf-8") as f:
            f.write("\n".join(parsed_lines))
            
        return f"Successfully processed: {filename}"

    except Exception as e:
        return f"Failed to process {filename}: {e}"

import argparse

def parse_screenplays():
    parser = argparse.ArgumentParser(description="Parse screenplay PDFs to text.")
    parser.add_argument("--file", help="Specific PDF filename to process (e.g., 'Screenplay.pdf')")
    args = parser.parse_args()

    # Setup paths
    pdf_folder = os.path.join(os.getcwd(), "Screenplays")
    output_folder_abs = os.path.join(os.getcwd(), OUTPUT_FOLDER)

    if not os.path.exists(output_folder_abs):
        os.makedirs(output_folder_abs)
        print(f"Created output folder: {output_folder_abs}")

    if not os.path.exists(pdf_folder):
         print(f"Error: The PDF_FOLDER '{pdf_folder}' does not exist.")
         return

    if args.file:
        # Process single file
        print(f"Processing single file: {args.file}")
        result = process_pdf(args.file, pdf_folder, output_folder_abs)
        print(result)
        return

    # Gather PDF files
    all_files = os.listdir(pdf_folder)
    pdf_files = [f for f in all_files if f.lower().endswith(".pdf")]
    
    if not pdf_files:
        print("No PDF files found to process.")
        return

    print(f"Found {len(pdf_files)} PDFs. Starting parallel processing...")

    # Use ProcessPoolExecutor for parallel processing
    # We use a max_workers default (usually number of processors)
    with ProcessPoolExecutor() as executor:
        # Submit all tasks
        futures = {executor.submit(process_pdf, pdf, pdf_folder, output_folder_abs): pdf for pdf in pdf_files}
        
        # Use tqdm to show progress as tasks complete
        results = []
        for future in tqdm(as_completed(futures), total=len(pdf_files), unit="pdf", desc="Parsing Screenplays"):
            result = future.result()
            results.append(result)
            
    # Optional: Print summary or errors
    # for res in results:
    #     if "Failed" in res:
    #         print(res)
            
    print(f"\nCompleted! Parsed screenplays saved to: {output_folder_abs}")

if __name__ == "__main__":
    parse_screenplays()
