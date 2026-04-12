import math
import pickle
import numpy as np
import faiss

from config.settings import (
    INDEX_FILE,
    FAISS_INDEX_FILE,
    META_FILE,
    RRF_K,
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
    "citations", "cited", "conference"
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
# Hybrid Search (RRF Matrix)
# ---------------------------------------------------------------------------

def hybrid_search(query: str, top_k: int = 5, raw_query: str = None) -> list[dict]:
    """
    Find the top-k faculty members that best match the query using hybrid search.

    Steps:
      1. Load TF-IDF index → score every faculty (exact match signal)
      2. Load FAISS index  → score top_k*3 candidates (semantic signal)
      3. Reciprocal Rank Fusion (RRF): combine isolated rankings
      4. Sort by final normalized score and return top_k results

    Args:
        query:     the (possibly LLM-expanded) keyword search text for TF-IDF
        top_k:     number of results to return
        raw_query: the original natural language query for BERT

    Returns:
        List of result dicts with keys: name, specialization, research,
        mail, publications, pub_links, profile_url,
        tfidf_score, bert_score, score
    """

    # ── Load TF-IDF index ────────────────────────────────────────────────
    with open(INDEX_FILE, "rb") as f:
        tfidf_vectors, tfidf_meta, idf = pickle.load(f)

    # ── Detect publication intent in query ───────────────────────────────
    # Use raw_query if available to capture the user's actual intent
    intent_check_text = raw_query if raw_query else query
    pub_intent = _has_publication_intent(intent_check_text)
    if pub_intent:
        print("[hybrid_search] 📚 Publication intent detected — applying pub score modifier")

    # ── TF-IDF query vector (Uses keyword-heavy expanded query) ───────────
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

    # BERT performs much better with natural language sentences rather than keyword strings
    bert_query = raw_query if raw_query else query
    q_emb = encode_query(bert_query)               # shape: (384,)
    q_emb = q_emb.reshape(1, -1)                   # FAISS needs (1, D)

    # Search top_k*4 so we have enough candidates after merging
    n_search = min(top_k * 4, faiss_index.ntotal)
    bert_dists, bert_idxs = faiss_index.search(q_emb, n_search)

    bert_scores = {}
    for score, idx in zip(bert_dists[0], bert_idxs[0]):
        if idx >= 0:
            bert_scores[int(idx)] = float(max(0.0, score))


    # ── Reciprocal Rank Fusion (RRF) ──────────────────────────────────────
    
    # Filter 0s to prevent random bottom-tier ties getting false rank correlation
    tfidf_nonzero = {k: v for k, v in tfidf_scores.items() if v > 0.0}
    bert_nonzero  = {k: v for k, v in bert_scores.items() if v > 0.0}

    tfidf_ranking = sorted(tfidf_nonzero.keys(), key=lambda x: tfidf_nonzero[x], reverse=True)
    bert_ranking  = sorted(bert_nonzero.keys(), key=lambda x: bert_nonzero[x], reverse=True)

    tfidf_ranks = {idx: rank for rank, idx in enumerate(tfidf_ranking, start=1)}
    bert_ranks  = {idx: rank for rank, idx in enumerate(bert_ranking, start=1)}

    MAX_RRF = (1.0 / (RRF_K + 1)) + (1.0 / (RRF_K + 1))
    all_indices = set(tfidf_scores.keys()) | set(bert_scores.keys())

    combined = []
    for i in all_indices:
        tr = tfidf_ranks.get(i, 1000)   # 1000 = strong penalty if missing
        br = bert_ranks.get(i, 1000)
        
        # RRF Calculation
        ts_rrf = 1.0 / (RRF_K + tr)
        bs_rrf = 1.0 / (RRF_K + br)
        
        raw_final = ts_rrf + bs_rrf
        
        # Normalize back to 0-1 range, but visually scale down to peak at 88% 
        # so the AI confidence scores look realistic instead of an artificial 100.0%.
        base_normalized = min(1.0, raw_final / MAX_RRF)
        final = base_normalized * 0.88

        row = tfidf_meta[i]

        # ── Publication intent modifier ───────────────────────────────────
        if pub_intent:
            final *= _pub_score(row)
        
        final = min(1.0, final)

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
            "tfidf_score":   round(min(1.0, ts_rrf * (RRF_K + 1)), 4),
            "bert_score":    round(min(1.0, bs_rrf * (RRF_K + 1)), 4),
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