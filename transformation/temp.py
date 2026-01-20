import re

# ------------------------
# BASIC TEXT CLEANING
# ------------------------

def clean_text(text, lowercase=True):
    if not isinstance(text, str):
        return ""
    text = re.sub(r"<.*?>", " ", text)
    text = re.sub(r"\s+", " ", text)
    text = text.replace("\n", " ").strip()
    return text.lower() if lowercase else text


def clean_name(name):
    if not isinstance(name, str):
        return ""
    name = re.sub(r"\b(dr|prof|professor)\.?\b", "", name, flags=re.IGNORECASE)
    return name.strip().title()


def extract_phd_field(text):
    if not isinstance(text, str):
        return ""
    match = re.search(r"\((.*?)\)", text)
    return match.group(1).strip().lower() if match else text.strip().lower()


def validate_email(email):
    if isinstance(email, str):
        email = email.strip().lower()
        return email if "@" in email else None
    return None


# ------------------------
# SPECIALIZATION HANDLING (IMPROVED)
# ------------------------

SYNONYM_MAP = {
    "ai": "artificial intelligence",
    "ml": "machine learning",
    "dl": "deep learning",
    "cv": "computer vision",
    "nlp": "natural language processing"
}

def normalize_term(term):
    term = term.strip().lower()
    return SYNONYM_MAP.get(term, term)


def specialization_text_to_list(text):
    if not isinstance(text, str):
        return []

    raw_terms = re.split(r"[,/&|]", text)
    cleaned = {
        normalize_term(t)
        for t in raw_terms
        if len(t.strip()) > 2
    }
    return sorted(cleaned)


# ------------------------
# RESEARCH RESOLUTION
# ------------------------

def normalize_research(research):
    if isinstance(research, str) and research.strip():
        return clean_text(research)
    return ""


def infer_research_from_other_fields(bio, specialization):
    signals = []
    if isinstance(bio, str):
        signals.append(bio)
    if isinstance(specialization, list):
        signals.extend(specialization)
    return clean_text(" ".join(signals))


def resolved_research(research, bio, specialization):
    primary = normalize_research(research)
    if primary:
        return primary
    return infer_research_from_other_fields(bio, specialization)


# ------------------------
# COMBINED TEXT FOR CHATBOT
# ------------------------

def combine_texts(bio, research, specialization, phd_field):
    parts = []
    if bio:
        parts.append(bio)
    if research:
        parts.append(research)
    if isinstance(specialization, list):
        parts.extend(specialization)
    if phd_field:
        parts.append(phd_field)
    return " ".join(parts)


# ------------------------
# PUBLICATION CLEANING
# ------------------------

def clean_publications(text):
    if not isinstance(text, str) or not text.strip():
        return []

    text = re.sub(r"<.*?>", " ", text)
    text = re.sub(r"https?://\S+", "", text)
    text = re.sub(r"\bdoi:\S+", "", text, flags=re.IGNORECASE)
    text = re.sub(r"[•|;\n]+", "||", text)
    text = re.sub(r"\s+", " ", text).strip()

    publications = [p.strip() for p in text.split("||") if len(p.strip()) > 25]

    seen = set()
    cleaned = []

    for pub in publications:
        sig = re.sub(r"\d{4}", "", pub.lower())
        sig = re.sub(r"\W+", "", sig)
        if sig not in seen:
            seen.add(sig)
            cleaned.append(pub)

    return cleaned
