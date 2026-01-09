import os
from pypdf import PdfReader, PdfWriter

# Configuration
PDF_FOLDER = r"C:/Users/carly/Documents/Coding/episode_scripts/"
OUTPUT_FOLDER = os.path.join(os.getcwd(), "Screenplays")

def remove_first_page_of_pdfs():
    # Ensure output directory exists
    if not os.path.exists(OUTPUT_FOLDER):
        os.makedirs(OUTPUT_FOLDER)
        print(f"Created output folder: {OUTPUT_FOLDER}")

    # Check if input folder exists
    if not os.path.exists(PDF_FOLDER):
        print(f"Error: The PDF_FOLDER '{PDF_FOLDER}' does not exist. Please update the path in the script.")
        return

    # Iterate over files in the directory
    for filename in os.listdir(PDF_FOLDER):
        if filename.lower().endswith(".pdf"):
            input_path = os.path.join(PDF_FOLDER, filename)
            output_path = os.path.join(OUTPUT_FOLDER, filename)

            try:
                reader = PdfReader(input_path)
                
                # Check if PDF has enough pages
                if len(reader.pages) < 1:
                    print(f"Skipping {filename}: Empty PDF.")
                    continue
                elif len(reader.pages) == 1:
                     print(f"Skipping {filename}: Only has one page. Result would be empty.")
                     continue

                writer = PdfWriter()
                
                # Add all pages except the first one (index 0)
                for page in reader.pages[1:]:
                    writer.add_page(page)

                # Write the new PDF
                with open(output_path, "wb") as f:
                    writer.write(f)
                
                print(f"Processed: {filename}")
                
            except Exception as e:
                print(f"Failed to process {filename}: {e}")

if __name__ == "__main__":
    print(f"Processing PDFs from: {PDF_FOLDER}")
    print(f"Saving to: {OUTPUT_FOLDER}")
    remove_first_page_of_pdfs()
