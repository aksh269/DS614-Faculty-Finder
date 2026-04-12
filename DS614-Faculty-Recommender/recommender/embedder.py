"""
embedder.py — BERT Sentence Embeddings via sentence-transformers.

Why all-MiniLM-L6-v2?
    - 384-dimensional embeddings (compact but rich)
    - ~5x faster than full BERT, but almost as accurate for semantic similarity
    - No GPU required — runs fine on CPU
    - Downloads once (~85MB) and is cached locally by the library

What it does:
    - Converts any text (faculty profile or search query) into a dense vector
      where semantically similar texts have vectors that are close together in
      the 384-dimensional space.  This lets us find "healthcare AI" faculty
      even when the user types "medicine machine learning".
"""

import numpy as np
from sentence_transformers import SentenceTransformer

# --------------------------------------------------------------------------
# Model is loaded ONCE and reused for every call.
# The first time this module is imported, the model is downloaded if not cached.
# --------------------------------------------------------------------------
MODEL_NAME = "all-MiniLM-L6-v2"

print(f"[embedder] Loading BERT model '{MODEL_NAME}' (downloads once if not cached)…")
_model = SentenceTransformer(MODEL_NAME)
print("[embedder] Model loaded ✓")


def encode_texts(texts: list[str]) -> np.ndarray:
    """
    Encode a list of strings into a 2-D numpy array of shape (N, 384).

    Used by index_builder.py to encode the entire faculty corpus at once.
    batch_size=32 keeps memory usage low on CPU.

    Args:
        texts: list of raw profile strings (one per faculty member)

    Returns:
        numpy array of float32 embeddings, shape (len(texts), 384)
    """
    embeddings = _model.encode(
        texts,
        batch_size=32,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,   # L2-normalized → cosine sim == dot product
    )
    return embeddings.astype("float32")


def encode_query(query: str) -> np.ndarray:
    """
    Encode a single search query string into a 1-D numpy array of shape (384,).

    Used by similarity.py at search time.

    Args:
        query: the (possibly LLM-expanded) search text

    Returns:
        1-D float32 numpy array of length 384
    """
    embedding = _model.encode(
        query,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    return embedding.astype("float32")
