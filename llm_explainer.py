import os
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

MODEL_NAME = "google/flan-t5-large"

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    token=os.getenv("HF_TOKEN")
)

model = AutoModelForSeq2SeqLM.from_pretrained(
    MODEL_NAME,
    token=os.getenv("HF_TOKEN")
)

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)


def explain_with_llm(context, question):
    prompt = f"""
You are an AI tutor specialized in Artificial Intelligence,
Machine Learning, Deep Learning, and Generative AI.

You ONLY answer questions related to AI concepts grounded in
foundational ideas such as LSTM, RNNs, Transformers, Attention,
GANs, VAEs, and deep learning fundamentals.

IMPORTANT RULES:
- Do NOT repeat the question
- Do NOT include instructions in your answer
- Do NOT mention authors, years, or paper titles
- Use simple, student-friendly language
- Stay strictly within AI/ML/DL concepts

TASK:
Explain the following concept clearly and in detail.

Concept:
{question}
"""

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=512
    ).to(device)

    outputs = model.generate(
        **inputs,
        max_new_tokens=1200,
        temperature=0.7,
        do_sample=True,
        top_p=0.95,
        repetition_penalty=1.1,
        no_repeat_ngram_size=3
    )

    text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return text
