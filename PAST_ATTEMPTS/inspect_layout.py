import pdfplumber
import os

pdf_path = "C:/Users/carly/Documents/Coding/episode_scripts/Hannibal_3x13_The_Wrath_of_the_Lamb.pdf"

with pdfplumber.open(pdf_path) as pdf:
    print(f"Inspecting {os.path.basename(pdf_path)}")
    for i, page in enumerate(pdf.pages[:5]): # Check first 5 pages
        text = page.extract_text(layout=True)
        if not text: continue
        print(f"--- Page {i} ---")
        for line in text.split("\n"):
            if "INT." in line or "EXT." in line:
                leading = len(line) - len(line.lstrip())
                print(f"Line: '{line.strip()[:30]}...' | Indent: {leading}")
