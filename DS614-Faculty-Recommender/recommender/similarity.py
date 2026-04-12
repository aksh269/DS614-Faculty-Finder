import math
import pickle
import numpy as np
import faiss

from config.settings import (
    INDEX_FILE,
    FAISS_INDEX_FILE,
    META_FILE,
    HYBRID_ALPHA,
    HYBRID_BETA,
)
from recommender.preprocessing import preprocess
from recommender.vectorizer import compute_tf, compute_tfidf


# ---------------------------------------------------------------------------
# Cosine similarity for TF-IDF sparse dicts
# ---------------------------------------------------------------------------

def cosine(v1: dict, v2: dict) -> float:
    common = set(v1) & set(v2)
    num    = sum(v1[w] * v2[w] for w in common)
    d1     = math.sqrt(sum(x * x for x in v1.values()))
    d2     = math.sqrt(sum(x * x for x in v2.values()))
    if d1 == 0 or d2 == 0:
        return 0.0
    return num / (d1 * d2)


# ---------------------------------------------------------------------------
# Publication-intent detection
# ---------------------------------------------------------------------------

_PUB_INTENT_KEYWORDS = {
    "publication", "publications", "published", "paper", "papers",
    "journal", "journals", "article", "articles", "research output",
    "citations", "cited", "conference", "work", "written",
}

def _has_publication_intent(query: str) -> bool:
    """
    Returns True if the user's query is specifically asking about publications
    (e.g. "faculty with many publications in statistics").
    """
    q_lower = query.lower()
    return any(kw in q_lower for kw in _PUB_INTENT_KEYWORDS)


def _pub_score(row: dict) -> float:
    """
    Rate how rich a faculty member's publication data is.
    Returns a multiplier:
      - 1.2  → has meaningful publication content  (boost)
      - 0.6  → empty / placeholder publications    (strong penalty)
    This is ONLY applied when the query has publication intent.
    """
    pub = str(row.get("publications", "") or "").strip()
    EMPTY_MARKERS = {"not_available", "n/a", "none", "na", "-", ""}
    if pub.lower() in EMPTY_MARKERS or len(pub) < 10:
        return 0.6   # penalise: no useful publication data
    return 1.2       # reward: has actual publication content


# ---------------------------------------------------------------------------
# Hybrid Search
# ---------------------------------------------------------------------------

def hybrid_search(query: str, top_k: int = 5) -> list[dict]:
    """
    Find the top-k faculty members that best match the query using hybrid search.

    Steps:
      1. Load TF-IDF index → score every faculty (exact match signal)
      2. Load FAISS index  → score top_k*3 candidates (semantic signal)
      3. Merge scores: Final = HYBRID_ALPHA * tfidf + HYBRID_BETA * bert
      4. Sort by final score and return top_k results

    Args:
        query:  the (possibly LLM-expanded) search text
        top_k:  number of results to return

    Returns:
        List of result dicts with keys: name, specialization, research,
        mail, publications, pub_links, profile_url,
        tfidf_score, bert_score, score
    """

    # ── Load TF-IDF index ────────────────────────────────────────────────
    with open(INDEX_FILE, "rb") as f:
        tfidf_vectors, tfidf_meta, idf = pickle.load(f)

    # ── Detect publication intent in query ───────────────────────────────
    pub_intent = _has_publication_intent(query)
    if pub_intent:
        print("[hybrid_search] 📚 Publication intent detected — applying pub score modifier")

    # ── TF-IDF query vector ───────────────────────────────────────────────
    tokens = preprocess(query)
    if not tokens:
        return []
    tf    = compute_tf(tokens)
    q_vec = compute_tfidf(tf, idf)

    # ── TF-IDF scores for every faculty (fast, small dataset) ─────────────
    tfidf_scores = {}
    for i, (vec, row) in enumerate(zip(tfidf_vectors, tfidf_meta)):
        tfidf_scores[i] = cosine(q_vec, vec)

    # ── BERT + FAISS scores ───────────────────────────────────────────────
    from recommender.embedder import encode_query   # lazy import

    with open(META_FILE, "rb") as f:
        faiss_meta = pickle.load(f)

    faiss_index = faiss.read_index(str(FAISS_INDEX_FILE))

    q_emb = encode_query(query)                     # shape: (384,)
    q_emb = q_emb.reshape(1, -1)                   # FAISS needs (1, D)

    # Search top_k*3 so we have enough candidates after merging
    n_search = min(top_k * 3, faiss_index.ntotal)
    bert_dists, bert_idxs = faiss_index.search(q_emb, n_search)
    # bert_dists[0]: cosine similarities (since vectors are L2-normalised)
    # bert_idxs[0]:  corresponding row indices in meta

    bert_scores = {}
    for score, idx in zip(bert_dists[0], bert_idxs[0]):
        if idx >= 0:   # FAISS returns -1 for padding
            # Clip to [0, 1] — cosine can be slightly negative for dissimilar vectors
            bert_scores[int(idx)] = float(max(0.0, score))

    # ── Merge scores ──────────────────────────────────────────────────────
    # Union of indices from both legs
    all_indices = set(tfidf_scores.keys()) | set(bert_scores.keys())

    combined = []
    for i in all_indices:
        ts = tfidf_scores.get(i, 0.0)
        bs = bert_scores.get(i, 0.0)
        final = HYBRID_ALPHA * ts + HYBRID_BETA * bs

        row = tfidf_meta[i]  # original faculty dict

        # ── Publication intent modifier ───────────────────────────────────
        # If the user is specifically asking about publications, penalise
        # faculty who have no publication data, and reward those who do.
        if pub_intent:
            final *= _pub_score(row)

        name = row.get("name", "")
        profile_url = row.get("profile_url", "")
        if not profile_url:
            slug = name.lower().replace(" ", "-").replace(".", "")
            profile_url = f"https://www.daiict.ac.in/faculty/{slug}"

        combined.append({
            "name":          name,
            "specialization":row.get("specialization", ""),
            "research":      row.get("research", ""),
            "mail":          row.get("mail", ""),
            "publications":  row.get("publications", ""),
            "pub_links":     row.get("pub_links", []),
            "profile_url":   profile_url,
            "tfidf_score":   round(ts, 4),
            "bert_score":    round(bs, 4),
            "score":         round(final, 4),
        })

    combined.sort(key=lambda x: x["score"], reverse=True)
    return combined[:top_k]


# ---------------------------------------------------------------------------
# Backwards-compatible wrapper (used by streamlit_app.py directly)
# ---------------------------------------------------------------------------

def get_recommendations(query: str, top_k: int = 5) -> list[dict]:
    """
    Drop-in replacement for the old TF-IDF get_recommendations.
    Now delegates to hybrid_search().
    """
    return hybrid_search(query, top_k)