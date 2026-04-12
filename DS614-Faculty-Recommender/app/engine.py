"""
engine.py — Main search orchestration pipeline.

Full flow for each user query:
  ┌─────────────────────┐
  │  1. Parse query     │  extract top-k number from "top 3 machine learning"
  └────────┬────────────┘
           │
  ┌────────▼────────────┐
  │  2. LLM Expansion   │  Gemini: "AI for medicine" → ["healthcare AI", "medical imaging"]
  └────────┬────────────┘          (graceful skip if no GEMINI_API_KEY)
           │
  ┌────────▼────────────┐
  │  3. Hybrid Search   │  0.3×TF-IDF + 0.7×BERT via FAISS
  └────────┬────────────┘
           │
  ┌────────▼────────────┐
  │  4. LLM Explanation │  Gemini generates "Recommended because…" per result
  └────────┬────────────┘          (graceful skip if no GEMINI_API_KEY)
           │
  ┌────────▼────────────┐
  │  5. Return results  │  enriched dicts with score, explanation, expanded_keywords
  └─────────────────────┘
"""

from recommender.similarity import hybrid_search
from recommender.query_parser import parse_query
from recommender.llm_layer import expand_query, explain_result, is_llm_available


def search(query: str, top_k: int = 5) -> list[dict]:
    """
    Full AI-powered faculty search pipeline.

    Args:
        query: raw user query string (e.g. "top 3 AI for healthcare")

    Returns:
        List of enriched faculty dicts
    """

    # ── Step 1: Parse query ("top N …" extraction) ────────────────────────
    clean_q, k = parse_query(query)

    if not clean_q.strip():
        return []

    # ── Step 2: LLM Query Expansion ──────────────────────────────────────
    if is_llm_available():
        expanded_keywords, search_text = expand_query(clean_q)
    else:
        expanded_keywords, search_text = [], clean_q

    # ── Step 3: Hybrid Search (TF-IDF + BERT via FAISS) ──────────────────
    results = hybrid_search(query=search_text, top_k=k)

    # ── Step 4: LLM Result Explanation ────────────────────────────────────
    for r in results:
        explanation = ""
        if is_llm_available():
            explanation = explain_result(
                query=clean_q,
                faculty_name=r.get("name", ""),
                specialization=r.get("specialization", ""),
                research=r.get("research", ""),
            )
        r["explanation"]        = explanation
        r["expanded_keywords"]  = expanded_keywords

        # Print to server log for debugging
        print(
            f"  [{r['score']:.3f}] {r['name']}"
            f"  tfidf={r['tfidf_score']:.3f}  bert={r['bert_score']:.3f}"
        )

    return results
