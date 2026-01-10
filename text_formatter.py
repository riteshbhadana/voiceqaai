import re


def clean_sentence(s):
    """
    Remove citations, equations, and noisy tokens
    """
    # Remove citation-like text
    s = re.sub(r"\b(et al\.?|arxiv|icml|nips|neurips|journal|conference)\b", "", s, flags=re.I)

    # Remove math symbols
    s = re.sub(r"[∑σλπμΔ∂αβγ≤≥≈≠→←×]", "", s)

    # Remove equations
    s = re.sub(r"Pr\s*\(.*?\)", "", s)

    # Remove author-only lines
    if len(s.split()) <= 3:
        return ""

    # Remove excessive punctuation
    s = re.sub(r"[•■◆▪]", "", s)

    return s.strip()


def format_with_headings(text, bullet_mode=True):
    sentences = re.split(r"\.|\n", text)

    sections = {
        "Core Idea": [],
        "Why It Matters": [],
        "Key Concepts": [],
        "Examples / Applications": [],
        "Important Notes": []
    }

    for raw in sentences:
        s = clean_sentence(raw)
        if not s or len(s) < 25:
            continue

        lower = s.lower()

        if any(k in lower for k in ["why", "purpose", "goal", "benefit", "important"]):
            sections["Why It Matters"].append(s)

        elif any(k in lower for k in ["lstm", "rnn", "gate", "memory", "cell"]):
            sections["Key Concepts"].append(s)

        elif any(k in lower for k in ["text", "speech", "handwriting", "generate", "task", "application"]):
            sections["Examples / Applications"].append(s)

        elif any(k in lower for k in ["note", "problem", "challenge", "advantage", "limitation"]):
            sections["Important Notes"].append(s)

        else:
            sections["Core Idea"].append(s)

    # Build output
    output = ""
    for title, items in sections.items():
        if not items:
            continue

        output += f"\n### {title}\n"
        for s in items[:5]:
            output += f"• {s}.\n"

    return output
