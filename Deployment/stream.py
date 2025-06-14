import streamlit as st
from transformers import AutoModelForCausalLM, BitsAndBytesConfig, AutoTokenizer
import torch
import re
import os
import fitz  # PyMuPDF
import pytesseract
from PIL import Image
import io
import pandas as pd

pytesseract.pytesseract.tesseract_cmd = r'D:/program_files/tesseract-main'


# Load model and tokenizer
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4"
)

model = AutoModelForCausalLM.from_pretrained(
    "../Models/checkpoint-6000/",
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
    local_files_only=True
)
tokenizer = AutoTokenizer.from_pretrained("../Models/checkpoint-6000/")

def clean_text(text):
    text = re.sub(r'\n+', ' ', text)
    text = re.sub(r'\s{2,}', ' ', text)
    text = re.sub(r'Slide \d+ of \d+', '', text)
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
            results.append(text)
        else:
            pix = page.get_pixmap()
            img = Image.open(io.BytesIO(pix.tobytes()))
            ocr_text = pytesseract.image_to_string(img, lang='eng')
            results.append(ocr_text)
    return results

def predict_answer(system_prompt: str, question: str, context: str = "") -> str:
    model.eval()
    context_text = f"{context}\n" if context else ""
    input_text = f"<|system|> {system_prompt}\n<|user|> {context_text}Q: {question}\n<|assistant|> A:"
    inputs = tokenizer(input_text, return_tensors="pt", truncation=True, padding=True).to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=100,
            do_sample=False,
            num_beams=5
        )
    pred = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return pred.split("A:")[-1].strip()

# Streamlit App
st.set_page_config(page_title="ML Chatbot", layout="wide")
st.title("🤖 Zaboola Machine Learning Chatbot")

system_prompt = st.text_input("System Prompt", value="You are Zaboola, an expert in machine learning and AI. Your purpose is to help students Leanr")

load_pdf = st.sidebar.checkbox("📄 Load PDF context", value=False)
context = ""

if load_pdf:
    st.sidebar.write("### Select PDF")
    pdfs = find_pdfs("../Dataset/")
    selected_pdf = st.sidebar.selectbox("PDF files:", pdfs)
    if selected_pdf:
        with st.spinner("Extracting PDF text..."):
            text = extract_text_with_ocr_fallback(selected_pdf)
            clean_pages = [clean_text(t) for t in text]
            context = " ".join(clean_pages)
            context = context[:2048]  # Truncate for model limits

# Chat interface
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

prompt = st.chat_input("Ask a question about machine learning...")

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Generating answer..."):
            response = predict_answer(system_prompt, prompt, context)
            st.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})
