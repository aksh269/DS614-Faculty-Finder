"""
build_index.py — Rebuild both the TF-IDF index and the FAISS vector index.

Run this script whenever faculty data changes:
    python scripts/build_index.py

On first run:
    - Downloads all-MiniLM-L6-v2 BERT model (~85MB, cached after that)
    - Creates index/vectors.pkl   (TF-IDF index for hybrid keyword leg)
    - Creates index/faiss.index   (FAISS index for hybrid BERT leg)
    - Creates index/meta.pkl      (faculty metadata for FAISS lookup)
"""

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from recommender.index_builder import build_index

if __name__ == "__main__":
    build_index()
