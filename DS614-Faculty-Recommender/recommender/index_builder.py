"""
index_builder.py — Builds BOTH the TF-IDF index (pickle) AND the FAISS vector index.

Two indices are created side-by-side so that hybrid search can use both:

TF-IDF index  (vectors.pkl):
    Contains: (tfidf_vectors_list, metadata_list, idf_dict)
    Used for: keyword-level exact-match scoring

FAISS index   (faiss.index + meta.pkl):
    Contains: FAISS IndexFlatIP with L2-normalised BERT embeddings
    Used for: semantic similarity — finds "meaning-alike" faculty

How FAISS works (IndexFlatIP):
    - "IP" = Inner Product (dot product)
    - Since embeddings are L2-normalised, dot product == cosine similarity
    - We store embeddings as a flat array → perfect for our dataset size (<1000 faculty)
    - index.search(query_vec, k) returns the top-k nearest faculty in ~ms

Run this script once (and re-run whenever faculty data changes):
    python scripts/build_index.py
"""

import pickle
import numpy as np
import faiss
from pathlib import Path

from recommender.preprocessing import preprocess
from recommender.vectorizer import compute_tf, compute_idf, compute_tfidf

from config.settings import (
    PHASE1_FACULTY_URL,
    INDEX_FILE,
    FAISS_INDEX_FILE,
    META_FILE,
)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def fetch_data():
    from config.settings import PROJECT_ROOT
    csv_path = (
        PROJECT_ROOT.parent
        / "DS614-Faculty-Finder"
        / "data"
        / "cleaned"
        / "transformed_faculty_data.csv"
    )

    print(f"[index_builder] Reading faculty data from {csv_path}")
    import pandas as pd

    df = pd.read_csv(csv_path)
    df = df.fillna("")
    return df.to_dict("records")


def safe(x) -> str:
    if x is None:
        return ""
    return str(x)


def build_docs(row: dict) -> str:
    """
    Construct a weighted text document for a faculty member.

    Fields are repeated to give them higher TF-IDF importance:
        name × 4, research × 3, specialization × 2, publications × 2, bio × 1
    """
    return (
        safe(row.get("name")) * 4 + " " +
        safe(row.get("research")) * 3 + " " +
        safe(row.get("specialization")) * 2 + " " +
        safe(row.get("publications")) * 2 + " " +
        safe(row.get("bio"))
    )


# ---------------------------------------------------------------------------
# Main builder
# ---------------------------------------------------------------------------

def build_index():
    data = fetch_data()

    raw_texts = []   # for BERT encoding
    docs      = []   # tokenised for TF-IDF
    meta      = []   # original row dicts

    for row in data:
        text   = build_docs(row)
        tokens = preprocess(text)
        docs.append(tokens)
        meta.append(row)
        raw_texts.append(text)   # keep unprocessed for BERT

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
