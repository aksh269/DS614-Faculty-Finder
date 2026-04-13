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


# We use a standard cosine similarity to see how much two sets of keywords overlap.
# This helps us understand if the search query shares exact words with a faculty profile.

def cosine(v1: dict, v2: dict) -> float:
    common = set(v1) & set(v2)
    num    = sum(v1[w] * v2[w] for w in common)
    d1     = math.sqrt(sum(x * x for x in v1.values()))
    d2     = math.sqrt(sum(x * x for x in v2.values()))
    if d1 == 0 or d2 == 0:
        return 0.0
    return num / (d1 * d2)


# Sometimes, users specifically want to read research papers or journals.
# Let's set up a few words to help us spot when they are looking for publications.

_PUB_INTENT_KEYWORDS = {
    "publication", "publications", "published", "paper", "papers",
    "journal", "journals", "article", "articles", "research output",
    "citations", "cited", "conference"
}

def _has_publication_intent(query: str) -> bool:
    """
    Check if the user is asking about publications or research papers.
    For example, if they search "faculty with publications in statistics".
    """
    q_lower = query.lower()
    return any(kw in q_lower for kw in _PUB_INTENT_KEYWORDS)


def _pub_score(row: dict) -> float:
    """
    Give a gentle boost to faculty profiles that have a solid list of published work.
    If their publication section is mostly empty, we lower the score a bit, 
    but only if the user explicitly asked for papers.
    """
    pub = str(row.get("publications", "") or "").strip()
    EMPTY_MARKERS = {"not_available", "n/a", "none", "na", "-", ""}
    if pub.lower() in EMPTY_MARKERS or len(pub) < 10:
        return 0.6   # Gently penalize since we don't have much to show the user
    return 1.2       # Boost them slightly because they have the data the user wants


# The main engine: This carefully blends traditional keyword matching with deep semantic understanding.
# It brings the best of both worlds to find the right faculty member for the job.

def hybrid_search(query: str, top_k: int = 5) -> list[dict]:

    # First, let's bring in our simple, fast keyword index.
    with open(INDEX_FILE, "rb") as f:
        tfidf_vectors, tfidf_meta, idf = pickle.load(f)

    # Are they looking for active researchers with publications?
    pub_intent = _has_publication_intent(query)
    if pub_intent:
        print("[hybrid_search] The user looks interested in publications. Applying a slight adjustment to highlight active researchers.")

    # We break down what the user typed into a neat format of weighted keywords.
    tokens = preprocess(query)
    if not tokens:
        return []
    tf    = compute_tf(tokens)
    q_vec = compute_tfidf(tf, idf)

    # Let's quickly score every faculty member on how well their profile matches the exact keywords.
    tfidf_scores = {}
    for i, (vec, row) in enumerate(zip(tfidf_vectors, tfidf_meta)):
        tfidf_scores[i] = cosine(q_vec, vec)

    # Now for the magic: we'll use an AI language model to understand the deeper meaning of the query.
    from recommender.embedder import encode_query   # We load this just in time to keep things snappy

    with open(META_FILE, "rb") as f:
        faiss_meta = pickle.load(f)

    faiss_index = faiss.read_index(str(FAISS_INDEX_FILE))

    q_emb = encode_query(query)                     # Turn the thought into numbers
    q_emb = q_emb.reshape(1, -1)                   # Prepare it for our similarity search

    # We search for a few extra people just to be safe before we merge and rank everyone.
    n_search = min(top_k * 3, faiss_index.ntotal)
    bert_dists, bert_idxs = faiss_index.search(q_emb, n_search)

    bert_scores = {}
    for score, idx in zip(bert_dists[0], bert_idxs[0]):
        if idx >= 0:
            bert_scores[int(idx)] = float(max(0.0, score))


    # Time to bring both approaches together into a final, beautifully blended score.
    all_indices = set(tfidf_scores.keys()) | set(bert_scores.keys())

    combined = []
    for i in all_indices:
        ts = tfidf_scores.get(i, 0.0)
        bs = bert_scores.get(i, 0.0)
        
        # We mix the explicit keywords with the deep semantic meaning
        final = HYBRID_ALPHA * ts + HYBRID_BETA * bs

        row = tfidf_meta[i]  # Let's look at the actual faculty details

        # Apply the nudge if publications are highly relevant to this query.
        if pub_intent:
            final *= _pub_score(row)
        
        # Make sure our final match score stays within 0 to 100%.
        final = min(1.0, final)

        name = row.get("name", "")
        profile_url = row.get("profile_url", "")
        
        # If there isn't a direct link, we make a soft attempt to build one dynamically.
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

    # Sort everyone from best match to lowest and return the top choices.
    combined.sort(key=lambda x: x["score"], reverse=True)
    return combined[:top_k]


# A friendly helper function so older parts of the application can still talk to our search engine.

def get_recommendations(query: str, top_k: int = 5) -> list[dict]:
    return hybrid_search(query, top_k)