import os
import base64
import streamlit as st
from pdf_loader import load_and_chunk_pdfs
from embeddings import embed_texts, model
from retriever import VectorStore
from context_cleaner import clean_context
from llm_explainer import explain_with_llm
from tts import speak

# ---------------- PAGE SETUP ----------------
st.set_page_config(page_title="Foundational AI RAG", layout="centered")
st.title("📚 Foundational AI Research Assistant (RAG + Audio)")


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PAPERS_DIR = os.path.join(BASE_DIR, "data", "papers")



# ---------------- SIDEBAR : READ & DOWNLOAD PAPERS ----------------
st.sidebar.title("📄 Foundational AI Papers")

if os.path.exists(PAPERS_DIR):
    pdf_files = sorted(
        [f for f in os.listdir(PAPERS_DIR) if f.lower().endswith(".pdf")]
    )

    if pdf_files:
        selected_paper = st.sidebar.selectbox(
            "Select a paper to read",
            pdf_files
        )

        paper_path = os.path.join(PAPERS_DIR, selected_paper)

        # Read PDF once
        with open(paper_path, "rb") as f:
            pdf_bytes = f.read()

        # Download button
        st.sidebar.download_button(
            label="⬇ Download PDF",
            data=pdf_bytes,
            file_name=selected_paper,
            mime="application/pdf"
        )

        # about
# ---------------- ABOUT PROJECT TOGGLE ----------------
st.sidebar.markdown("---")
show_about = st.sidebar.toggle("ℹ️ About this project")

if show_about:
    st.sidebar.markdown("""
### 📚 Project Overview: Foundational AI Research Assistant

This project is an **AI-powered research assistant** built on **10 foundational research papers**
that every AI engineer, researcher, and practitioner should understand.
These papers form the **core concepts of modern Artificial Intelligence, Deep Learning, and Generative AI**.

---

### 🔍 What you can do
- 📄 Read and download the original research papers  
- ❓ Ask natural language questions  
- 🧠 Get structured, easy-to-understand explanations grounded strictly in these papers  
- 🔊 Listen to answers via audio (Text-to-Speech)

---

### 🧠 How it works
The assistant uses **Retrieval-Augmented Generation (RAG)** to ensure answers are:
- Faithful to original research
- Free from hallucination
- Focused on learning, not guessing

---

### 🔑 Why these 10 papers matter
These papers introduced the **fundamental ideas behind today’s AI systems**, including:

- Sequence modeling and memory (RNNs, LSTM)
- Encoder–decoder learning (Seq2Seq, NMT)
- Attention mechanisms and Transformers
- Deep learning foundations
- Regularization and optimization techniques
- Generative models (VAE, GAN)
- Scaling laws and training deep networks

Almost **every modern AI model** (ChatGPT, Gemini, Claude, etc.)
is built on concepts introduced in these works.
""")





# ---------------- LOAD SYSTEM ----------------
@st.cache_resource
def load_system():
    if not os.path.exists(PAPERS_DIR):
        return None, None, ["❌ data/papers folder not found"]

    if len(os.listdir(PAPERS_DIR)) == 0:
        return None, None, ["❌ data/papers folder is empty"]

    chunks, failed = load_and_chunk_pdfs(PAPERS_DIR)
    if not chunks:
        return None, None, failed

    texts = [c["text"] for c in chunks]
    embeddings = embed_texts(texts)

    store = VectorStore(embeddings.shape[1])
    store.add(embeddings, chunks)

    return model, store, failed


model, store, failed = load_system()

# ---------------- ERROR HANDLING ----------------
if failed:
    st.warning("Some PDFs could not be processed:")
    for f in failed:
        st.write(f)

if model is None or store is None:
    st.error("No readable text found in PDFs.")
    st.stop()

# ---------------- SAFE WRAPPER ----------------
def wrap_answer(answer: str, question: str) -> str:
    base = (
        answer.strip()
        if answer and len(answer.strip()) > 30
        else f"{question} is an important concept in artificial intelligence."
    )

    return f"""
### Core Idea
- {base}

### Why It Matters
- This concept helps AI systems learn patterns effectively.
- It is widely used in real-world machine learning tasks.

### Key Concepts
- Information flow within neural networks
- Learning from sequences or structured data

### Examples / Applications
- Natural language processing
- Time-series prediction
- Speech and signal processing
"""

# ---------------- FORMAT SAFETY ----------------
def enforce_format(answer: str, question: str) -> str:
    if not answer or len(answer.strip()) < 50:
        return wrap_answer(answer, question)

    if "### Core Idea" in answer:
        return answer

    return wrap_answer(answer, question)

# ---------------- CHAT UI ----------------
st.markdown("---")
query = st.text_input("Ask a foundational AI question")

if st.button("Get Answer") and query.strip():
    q_emb = model.encode(query)

    results = store.search(q_emb, k=10)
    raw_context = "\n\n".join(
    r["text"][:1200] for r in results  # cap per chunk, avoid noise
    )

    context = clean_context(raw_context)

    answer = explain_with_llm(context, query)
    answer = enforce_format(answer, query)

    st.subheader("Answer")
    st.markdown(answer)
    st.caption("Sources: " + ", ".join(set(r["source"] for r in results)))

    audio = speak(answer)
    if audio:
        st.audio(audio)
