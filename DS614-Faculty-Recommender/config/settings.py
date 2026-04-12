from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

PHASE1_API_BASE = "http://localhost:8000"
PHASE1_FACULTY_ENDPOINT = "/faculty"

PHASE1_FACULTY_URL = PHASE1_API_BASE + PHASE1_FACULTY_ENDPOINT

DATA_DIR = PROJECT_ROOT / "data"
INDEX_DIR = PROJECT_ROOT / "index"

INDEX_FILE = INDEX_DIR / "vectors.pkl"       # TF-IDF index (still used for hybrid)
STOPWORDS_PATH = DATA_DIR / "stopwords.txt"

# --- NEW: Deep Learning / Hybrid Search paths & weights ---
FAISS_INDEX_FILE = INDEX_DIR / "faiss.index"  # FAISS vector index
META_FILE        = INDEX_DIR / "meta.pkl"      # faculty metadata for FAISS results

# Hybrid search blending weights: Final = ALPHA*TF-IDF + BETA*BERT
HYBRID_ALPHA = 0.5   # weight for TF-IDF (keyword signal)
HYBRID_BETA  = 0.5   # weight for BERT  (semantic signal)
