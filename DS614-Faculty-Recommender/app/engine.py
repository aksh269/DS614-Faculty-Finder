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


def search(query: str, top_k: int = 5, use_gemini: bool = True) -> list[dict]:
    """
    Full AI-powered faculty search pipeline.

    Args:
        query: raw user query string (e.g. "top 3 AI for healthcare")
        top_k: default number of results if not specified in query
        use_gemini: toggle LLM expansion and explanation

    Returns:
        List of enriched faculty dicts
    """

    # ── Step 1: Parse query ("top N …" extraction) ────────────────────────
    clean_q, parsed_k = parse_query(query)
    # If the user typed "top 10", parse_query returns 10. Otherwise it returns 5.
    # We will favor parsed_k if it differs from the default 5, otherwise fallback to UI's top_k.
    final_k = parsed_k if parsed_k != 5 else top_k

    if not clean_q.strip():
        return []

    # ── Step 2: LLM Query Expansion ──────────────────────────────────────
    if use_gemini and is_llm_available():
        expanded_keywords, search_text = expand_query(clean_q)
    else:
        expanded_keywords, search_text = [], clean_q

    # ── Step 3: Hybrid Search (TF-IDF + BERT via FAISS) ──────────────────
    # Pass search_text (expanded) for TF-IDF, and clean_q (raw) for BERT
    results = hybrid_search(query=search_text, top_k=final_k, raw_query=clean_q)

    # ── Step 4: LLM Result Explanation ────────────────────────────────────
    for r in results:
        explanation = ""
        if use_gemini and is_llm_available():
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
