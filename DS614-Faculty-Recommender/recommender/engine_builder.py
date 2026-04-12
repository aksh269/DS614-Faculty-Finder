import pickle
from pathlib import Path

def fetch_data():
    from config.settings import PROJECT_ROOT, DATA_DIR
    import os

    # Possible locations for the data file
    candidates = [
        # 1. Environment variable override
        os.getenv("FACULTY_DATA_CSV"),
        # 2. Sibling folder (Standard project structure)
        PROJECT_ROOT.parent / "DS614-Faculty-Finder" / "data" / "cleaned" / "transformed_faculty_data.csv",
        # 3. Local data folder (Container structure)
        DATA_DIR / "transformed_faculty_data.csv",
        # 4. Root folder fallback
        PROJECT_ROOT / "transformed_faculty_data.csv"
    ]

    csv_path = None
    for path in candidates:
        if path and Path(path).exists():
            csv_path = Path(path)
            break

    if not csv_path:
        error_msg = f"Could not find faculty data CSV. Checked: {candidates}"
        print(f"[index_builder] ❌ Error: {error_msg}")
        # Return empty list and let build_index fail gracefully or handled accordingly
        return []

    print(f"[index_builder] 📁 Reading faculty data from: {csv_path}")
    import pandas as pd

    df = pd.read_csv(csv_path)
    df = df.fillna("")
    return df.to_dict("records")


def safe(x) -> str:
    if x is None:
        return ""
    return str(x)


def build_docs(row: dict) -> str:

    return (
        safe(row.get("name")) * 4 + " " +
        safe(row.get("research")) * 3 + " " +
        safe(row.get("specialization")) * 2 + " " +
        safe(row.get("publications")) * 2 + " " +
        safe(row.get("bio"))
    )

def build_clean_text(row: dict) -> str:

    parts = []
    if name := safe(row.get("name")): parts.append(f"Professor {name}.")
    if spec := safe(row.get("specialization")): parts.append(f"Specialization: {spec}.")
    if res := safe(row.get("research")): parts.append(f"Research interests: {res}.")
    if pub := safe(row.get("publications")): parts.append(f"Key publications include: {pub}.")
    if bio := safe(row.get("bio")): parts.append(f"Biography: {bio}.")
    return " ".join(parts)


# ---------------------------------------------------------------------------
# Main builder
# ---------------------------------------------------------------------------

def build_all_indices():
    import numpy as np
    import faiss
    from recommender.preprocessing import preprocess
    from recommender.vectorizer import compute_tf, compute_idf, compute_tfidf
    from config.settings import (
        PHASE1_FACULTY_URL,
        INDEX_FILE,
        FAISS_INDEX_FILE,
        META_FILE,
    )

    data = fetch_data()

    raw_texts = []   # for BERT encoding (natural text)
    docs      = []   # tokenised for TF-IDF (weighted text)
    meta      = []   # original row dicts

    for row in data:
        tfidf_text = build_docs(row)
        bert_text  = build_clean_text(row)

        tokens = preprocess(tfidf_text)
        docs.append(tokens)
        meta.append(row)
        raw_texts.append(bert_text)   # Keep as clean, natural text for BERT

    # -------------------------------------------------------------------
    # Phase A: Build TF-IDF index (unchanged logic)
    # -------------------------------------------------------------------
    print("\n[index_builder] ── Phase A: TF-IDF index ─────────────────────")
    idf     = compute_idf(docs)
    vectors = []
    for tokens in docs:
        tf  = compute_tf(tokens)
        vec = compute_tfidf(tf, idf)
        vectors.append(vec)

    with open(INDEX_FILE, "wb") as f:
        pickle.dump((vectors, meta, idf), f)
    print(f"[index_builder] TF-IDF index saved → {INDEX_FILE}")

    # -------------------------------------------------------------------
    # Phase B: Build BERT + FAISS index
    # -------------------------------------------------------------------
    print("\n[index_builder] ── Phase B: BERT + FAISS index ────────────────")
    from recommender.embedder import encode_texts   # lazy import (model loads here)

    print(f"[index_builder] Encoding {len(raw_texts)} faculty profiles with BERT…")
    embeddings = encode_texts(raw_texts)   # shape: (N, 384), float32, L2-normalised

    # Build a FAISS flat index using Inner Product (== cosine sim for L2-norm vectors)
    dimension = embeddings.shape[1]        # 384
    faiss_index = faiss.IndexFlatIP(dimension)
    faiss_index.add(embeddings)

    # Save FAISS index
    FAISS_INDEX_FILE.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(faiss_index, str(FAISS_INDEX_FILE))
    print(f"[index_builder] FAISS index saved  → {FAISS_INDEX_FILE}  ({faiss_index.ntotal} vectors)")

    # Save metadata separately (FAISS stores vectors but not the original rows)
    with open(META_FILE, "wb") as f:
        pickle.dump(meta, f)
    print(f"[index_builder] Metadata saved      → {META_FILE}")

    print("\n[index_builder] ✅ All indices built successfully!\n")
