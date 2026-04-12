from recommender.similarity import hybrid_search
from recommender.query_parser import parse_query
from recommender.llm_layer import expand_query, explain_result, is_llm_available


def search(query: str, top_k: int = 5) -> list[dict]:

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
