import pdfplumber
import os

pdf_path = "C:/Users/carly/Documents/Coding/episode_scripts/Hannibal_1x01_Aperitif.pdf"

with pdfplumber.open(pdf_path) as pdf:
    with open("pdf_dump.txt", "w", encoding="utf-8") as out:
        for i, page in enumerate(pdf.pages[:15]): 
            text = page.extract_text(layout=True)
            if not text: continue
            out.write(f"--- Page {i+1} ---\n")
            out.write(text)
            out.write("\n" + "-" * 40 + "\n")
    print("Dumped to pdf_dump.txt")
