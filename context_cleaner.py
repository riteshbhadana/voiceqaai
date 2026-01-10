import re


def clean_context(text):
    text = re.sub(r"\(.*?et al\..*?\)", "", text, flags=re.I)
    text = re.sub(r"Pr\s*\(.*?\)", "", text)
    text = re.sub(r"[∑σλπμΔ∂αβγ≤≥≈≠→←×\[\]\|\=\+]", " ", text)
    text = re.sub(r"\s+", " ", text)

    sentences = []
    for s in text.split("."):
        s = s.strip()
        if len(s) > 20 and s.lower() not in [x.lower() for x in sentences]:
            sentences.append(s)

    return ". ".join(sentences[:20])
