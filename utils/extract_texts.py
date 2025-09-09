import os
import pdfplumber
import pandas as pd

def extract_text_from_pdf(pdf_path):
    text = ""
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                if page.extract_text():
                    text += page.extract_text() + "\n"
    except Exception as e:
        print(f"Error reading {pdf_path}: {e}")
    return text

def process_resume_pdfs(root_dir):
    texts = []
    labels = []

    for domain_folder in os.listdir(root_dir):
        domain_path = os.path.join(root_dir, domain_folder)
        if os.path.isdir(domain_path):
            print(f"Processing domain: {domain_folder}")
            for file_name in os.listdir(domain_path):
                if file_name.endswith(".pdf"):
                    pdf_path = os.path.join(domain_path, file_name)
                    extracted_text = extract_text_from_pdf(pdf_path)
                    if extracted_text.strip():
                        texts.append(extracted_text)
                        labels.append(domain_folder)
                    else:
                        print(f"No text extracted from {pdf_path}")

    return texts, labels

if __name__ == "__main__":
    raw_pdfs_path = r"C:\Users\rajas\OneDrive\Desktop\resume_analyzer\data\raw_pdfs"# Replace with your PDF folder's full path

    texts, labels = process_resume_pdfs(raw_pdfs_path)

    df = pd.DataFrame({
        'resume_text': texts,
        'category': labels
    })

    os.makedirs('data', exist_ok=True)
    df.to_csv('data/resume_texts.csv', index=False)

    print(f"Extracted {len(texts)} resumes and saved to data/resume_texts.csv")
