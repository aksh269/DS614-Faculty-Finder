
import os
import json
import re

# --------------------------------------------------------------------------
# Initialise Gemini client (graceful — no crash if key absent)
# --------------------------------------------------------------------------
_client = None
_GEMINI_AVAILABLE = False

try:
    import google.generativeai as genai
    _api_key = os.environ.get("GEMINI_API_KEY", "")
    if _api_key:
        genai.configure(api_key=_api_key)
        _client = genai.GenerativeModel("gemini-2.0-flash")
        _GEMINI_AVAILABLE = True
        print("[llm_layer] Gemini client ready ✓")
    else:
        print("[llm_layer] ⚠️  GEMINI_API_KEY not set — LLM features disabled (graceful fallback active)")
except ImportError:
    print("[llm_layer] ⚠️  google-generativeai not installed — LLM features disabled")


# --------------------------------------------------------------------------
# 1. Query Expansion
# --------------------------------------------------------------------------

_EXPAND_PROMPT = """\
You are a research-domain keyword extractor for a faculty search engine.

Given the user's search query, extract 4-6 precise technical keyword phrases
that best describe the research domain(s) being requested.

Rules:
- Return ONLY a valid JSON array of strings, nothing else
- Phrases should be academic/technical (e.g. "medical imaging", "deep learning")
- Keep each phrase short (2-4 words max)
- Do NOT include generic words like "faculty", "professor", "research", "want"

Query: {query}

JSON array:"""


def expand_query(user_query: str) -> tuple[list[str], str]:
    """
    Use Gemini to expand a natural-language query into technical keywords.

    Args:
        user_query: raw user search string

    Returns:
        (keywords_list, expanded_text)
        - keywords_list: e.g. ["machine learning", "healthcare AI"]
        - expanded_text: the keywords joined as a single string for embedding
          Falls back to ([], user_query) if Gemini is unavailable.
    """
    if not _GEMINI_AVAILABLE:
        return [], user_query

    try:
        prompt = _EXPAND_PROMPT.format(query=user_query)
        response = _client.generate_content(prompt)
        raw = response.text.strip()

        # Parse JSON array from response
        # Sometimes the model wraps it in markdown, so strip that
        raw = re.sub(r"```(?:json)?", "", raw).strip().strip("`").strip()
        keywords = json.loads(raw)

        if isinstance(keywords, list) and len(keywords) > 0:
            expanded_text = " ".join(keywords)
            print(f"[llm_layer] Query expanded → {keywords}")
            return keywords, expanded_text

    except Exception as e:
        print(f"[llm_layer] Query expansion failed ({e}), using raw query")

    return [], user_query


# --------------------------------------------------------------------------
# 2. Result Explanation
# --------------------------------------------------------------------------

_EXPLAIN_PROMPT = """\
You are an AI assistant explaining a faculty recommendation to a student.

The student searched for: "{query}"
The recommended faculty member is: {name}
Their research areas: {research}
Their specialization: {specialization}

Write ONE concise sentence (max 30 words) explaining why this faculty member
is a good match for the student's query. Be specific — mention actual research
topics. Do NOT start with "This faculty" or "I".

Explanation:"""


def explain_result(
    query: str,
    faculty_name: str,
    specialization: str,
    research: str,
) -> str:
    """
    Generate a natural-language explanation for why a faculty member was recommended.

    Args:
        query:          original (or expanded) user search query
        faculty_name:   faculty member's name
        specialization: their listed specialization
        research:       their research areas/interests

    Returns:
        A one-sentence explanation string, or an empty string if unavailable.
    """
    if not _GEMINI_AVAILABLE:
        return ""

    try:
        prompt = _EXPLAIN_PROMPT.format(
            query=query,
            name=faculty_name,
            research=research or "not specified",
            specialization=specialization or "not specified",
        )
        response = _client.generate_content(prompt)
        explanation = response.text.strip()
        return explanation

    except Exception as e:
        print(f"[llm_layer] Explanation generation failed ({e})")
        return ""


def is_llm_available() -> bool:
    """Returns True if Gemini is configured and available."""
    return _GEMINI_AVAILABLE
