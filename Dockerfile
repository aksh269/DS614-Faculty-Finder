FROM python:3.10-slim

WORKDIR /app

# System dependencies for FAISS / Torch / Scikit-learn
RUN apt-get update && apt-get install -y \
    build-essential \
    python3-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY DS614-Faculty-Recommender/requirements.txt .
RUN python -m pip install --upgrade pip && \
    python -m pip install --no-cache-dir -r requirements.txt

# Copy Recommender code
COPY DS614-Faculty-Recommender/ .

# Ensure Data exists for the index builder (for a self-contained container)
# We copy the cleaned data from Finder phase into the local data folder
COPY DS614-Faculty-Finder/data/cleaned/transformed_faculty_data.csv data/

# Build the AI Search Indices (BERT embeddings + FAISS)
# This downloads the BERT model (~85MB) during the build phase
RUN python scripts/build_index.py

# Expose ports
EXPOSE 8501

# Start the ScholarMatch application
# Handles PORT for Railway/Cloud deployments, or default 8501 for local Docker
CMD if [ -n "$PORT" ]; then \
    streamlit run UI/streamlit_app.py --server.port $PORT --server.address 0.0.0.0 --server.headless true; \
    else \
    streamlit run UI/streamlit_app.py --server.port 8501 --server.address 0.0.0.0 --server.headless true; \
    fi

