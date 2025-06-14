# %%
import re
import os
os.chdir(r"D:\Industry\Projects\machine learning chatbot\Notebooks")
import pytesseract
from pdf2image import convert_from_path
import fitz  # PyMuPDF
import pytesseract
from PIL import Image
import io
import pandas as pd
pytesseract.pytesseract.tesseract_cmd = r'D:\program_files\tesseract-main'

def clean_text(text):
    text = re.sub(r'\n+', ' ', text)  # collapse newlines
    text = re.sub(r'\s{2,}', ' ', text)  # collapse multiple spaces
    text = re.sub(r'Slide \d+ of \d+', '', text)  # remove slide numbers
    return text.strip()

def find_pdfs(root_dir):
    pdf_files = []
    for dirpath, _, filenames in os.walk(root_dir):
        for file in filenames:
            if file.lower().endswith('.pdf'):
                pdf_files.append(os.path.join(dirpath, file))
    return pdf_files




def extract_text_with_ocr_fallback(pdf_path):
    doc = fitz.open(pdf_path)
    results = []

    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        text = page.get_text()
        if text.strip():
            # Text extracted successfully
            results.append(text)
        else:
            # No text found, fallback to OCR of rendered page image
            pix = page.get_pixmap()  # render page to image
            img = Image.open(io.BytesIO(pix.tobytes()))
            ocr_text = pytesseract.image_to_string(img, lang='eng')
            results.append(ocr_text)

    return results

# %%
pdfs = find_pdfs("../Dataset/")
df = pd.DataFrame(pdfs, columns=["pdfs"])
for pdf in pdfs:
    df = pd.concat([df, pd.DataFrame([{"pdfs": pdf}])], ignore_index=True)

# %%

for pdf in pdfs:
    text = extract_text_with_ocr_fallback(pdf)
    for page_idx in range(len(text)):
        page = clean_text(text[page_idx])
        text[page_idx] = page
    df.loc[df.pdfs == pdf, "text"] = text
    break
df

# %%
df

# %%



