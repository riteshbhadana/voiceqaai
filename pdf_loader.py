import os
import pdfplumber


def load_and_chunk_pdfs(folder_path, chunk_size=500, overlap=50):
    chunks = []
    failed = []

    for file in os.listdir(folder_path):
        if not file.lower().endswith(".pdf"):
            continue

        path = os.path.join(folder_path, file)

        try:
            text = ""
            with pdfplumber.open(path) as pdf:
                for page in pdf.pages:
                    t = page.extract_text()
                    if t:
                        text += t + " "

            text = text.replace("\n", " ").strip()

            if len(text) < 200:
                failed.append(file)
                continue

            words = text.split()
            i = 0
            while i < len(words):
                chunks.append({
                    "text": " ".join(words[i:i + chunk_size]),
                    "source": file
                })
                i += chunk_size - overlap

        except Exception:
            failed.append(file)

    return chunks, failed
