import pdfplumber
import os

SCRIPTS_FOLDER = "C:/Users/carly/Documents/Coding/episode_scripts/"
filename = "Hannibal_1x01_Aperitif.pdf"
path = os.path.join(SCRIPTS_FOLDER, filename)

with pdfplumber.open(path) as pdf:
    for i in range(3):
        if i < len(pdf.pages):
            page = pdf.pages[i]
            text = page.extract_text(layout=True)
            print(f"--- PAGE {i} OF {filename} ---")
            print(text[:2000] if text else "NO TEXT FOUND") 
            print("--- END OF PAGE ---")
