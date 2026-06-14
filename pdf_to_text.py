import pdfplumber
import os

text = ""

for filename in os.listdir("My_Data"):
    if filename.lower().endswith(".pdf"):
        print(f"Reading: {filename}")

        pdf_path = os.path.join("My_Data", filename)

        with pdfplumber.open(pdf_path) as pdf:
            for page_num, page in enumerate(pdf.pages, start=1):
                print(f"  Page {page_num}")

                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"

with open("output.txt", "w", encoding="utf-8") as f:
    f.write(text)

print("Done!")