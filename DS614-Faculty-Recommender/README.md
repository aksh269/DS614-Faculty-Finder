# ScholarMatch — AI-Powered Faculty Recommender System

---

# Project Overview

**ScholarMatch** is an intelligent search engine designed to help students and researchers find relevant faculty members based on their research interests with high precision. Unlike simple keyword matching, this system uses a **Hybrid AI Approach** combining traditional **TF-IDF Vector Space Models** with modern **BERT Sentence Embeddings** and **Vector Databases (FAISS)**.

The system is further enhanced by a **Generative AI Layer (Google Gemini)** that understands natural language queries, expands them into technical keywords, and provides human-like explanations for each recommendation. It is deployed as a high-performance web application using **Streamlit**.

**Live Deployment:** [https://ds614-faculty-finder-the-data-engineers.streamlit.app](https://ds614-faculty-finder-the-data-engineers.streamlit.app)

---

# Tech Stack

| Category | Tools / Technologies | Purpose |
|--------|---------------------|---------|
| Programming Language | Python 3.10 | Core logic and AI pipeline |
| Deep Learning | Sentence-Transformers (BERT) | Semantic understanding of research themes |
| Vector Database | FAISS (Facebook AI) | High-speed similarity search across high-dimensional vectors |
| Generative AI | Google Gemini Pro | Query expansion and natural language result explanation |
| Frontend Framework | Streamlit | Interactive web UI with real-time AI insights |
| Containerization | Docker | Reproducible deployment and orchestration |

---

# Recommendation Algorithm

ScholarMatch implements a multi-layered information retrieval system that leverages both statistical keyword analysis and deep learning semantic understanding.

### 1. Preprocessing (`recommender/preprocessing.py`)
Raw text is transformed into a normalized format suitable for AI processing:
- **Phrase Merging**: Domain-specific terms like "Deep Learning" are merged into single tokens (e.g., `deep_learning`) to ensure they are treated as unique concept units.
- **Cleaning & Filtering**: Stopwords are removed, and text is cleaned of noise to focus on core research expertise.

### 2. Hybrid Vectorization (`recommender/vectorizer.py` & `recommender/embedder.py`)
The system creates two distinct representations for every faculty profile:

#### **A. Statistical Signal (TF-IDF)**
Calculates the relative importance of keywords using the formula:
$$ \text{TF-IDF}(t, d) = \frac{f(t,d)}{|d|} \times \log\left(\frac{N + 1}{DF(t) + 1}\right) + 1 $$
This ensures rare technical terms have a higher impact on search results.

#### **B. Semantic Signal (BERT Embeddings)**
Uses the `all-MiniLM-L6-v2` BERT model to convert text into **384-dimensional dense vectors**. This allows the system to match "heart disease" with "cardiovascular research" even if they share no common keywords.

### 3. Document Weighting (`recommender/index_builder.py`)
Relevance is further refined by weighting profile sections:
- **Name**: 4x weight
- **Research Interests**: 3x weight
- **Specialization**: 2x weight
- **Publications**: 2x weight

### 4. Similarity & Ranking (Hybrid & FAISS)
ScholarMatch uses **FAISS (Facebook AI Similarity Search)** to index vectors and compute similarities in milliseconds. The final ranking is determined by a **Hybrid Score**:

$$ \text{Final Score} = 0.3 \times \text{TF-IDF Cosine} + 0.7 \times \text{BERT Similarity} $$

- The **BERT** signal (0.7 weight) handles deep meaning.
- The **TF-IDF** signal (0.3 weight) ensures exact technical acronyms (like VLSI or IoT) still get high priority.

### 5. Generative AI Layer (LLM)
Powered by **Google Gemini**, the system includes:
- **Query Expansion**: Automatically translates vague user queries into a list of 4-6 precise technical keywords.
- **Explainable AI**: Generates a one-sentence justification for every recommendation (e.g., *"This faculty is a match because they specialize in NLP and clinical imaging...*").

### 6. Publication Intent Filter
A specialized logic layer that detects if a user is searching for research-heavy matches. It applies a **Penalty-Boost System**:
- Faculty with verified publication data receive a **1.2x score boost**.
- Profiles with missing/empty publication data in the query context receive a **0.6x penalty**.

---

# Project Structure

```text
DS614-Faculty-Recommender/
│
├── app/                         # Orchestration logic
│   └── engine.py                # Pipeline: Parse -> Expand -> Hybrid Search -> Explain
│
├── recommender/                 # AI Engine Implementation
│   ├── embedder.py              # BERT encoding logic
│   ├── llm_layer.py             # Gemini API integration
│   ├── preprocessing.py         # Text normalization
│   ├── vectorizer.py            # Custom TF-IDF logic
│   ├── similarity.py            # Hybrid search & ranking (FAISS)
│   └── query_parser.py          # Intelligent query parsing
│
├── index/                       # Vector storage
│   ├── faiss.index              # BERT vector db
│   └── vectors.pkl              # TF-IDF index
│
├── UI/                          # Interactive Frontend
│   └── streamlit_app.py         # Streamlit App with AI Insight UI
│
├── requirements.txt             # Project dependencies
└── README.md                    # This documentation
```

---

# How to Run Manually (Local Setup)

### Step 1: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 2: Build the AI Search Index
Before searching, you must generate the hybrid vector indices:
```bash
python scripts/build_index.py
```
*This downloads the BERT model (~85MB) and processes all faculty data into FAISS and TF-IDF formats.*

### Step 3: Set Gemini API Key (Optional)
To enable LLM expansion and explanations:
```bash
export GEMINI_API_KEY="your_api_key"
```

### Step 4: Run the App
```bash
streamlit run UI/streamlit_app.py
```

---

# How to Run with Docker
```bash
docker build -t scholar-match .
docker run -p 8080:8080 scholar-match
```

---

# Team

👩‍💻 **Sanjana Nathani**
- **Student ID:** 202518002  
- **Program:** M.Sc. Data Science  
- **Institution:** Dhirubhai Ambani University (DAU), Gandhinagar  
- **Role in Project:** AI Systems & Data Engineer  

👨‍💻 **Aksh Patel**  
- **Student ID:** 202518046  
- **Program:** M.Sc. Data Science  
- **Institution:** Dhirubhai Ambani University (DAU), Gandhinagar  
- **Role in Project:** Backend & Data Engineer  

---
*ScholarMatch is part of the larger DS614 Big Data Engineering project, demonstrating the end-to-end application of Modern AI and Data Engineering principles.*
