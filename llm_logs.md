# LLM Interaction Logs

This document contains realistic questions asked to LLMs during Faculty Finder development, including debugging scenarios, deployment issues, and complex integration problems.

---

## Question 1: scrapy spider runs but saves empty csv file, the items are being yielded i can see them in logs but csv has only headers???

**LLM Generated Output:**

This is a classic Scrapy pipeline issue. The items are being created but not reaching the export pipeline. Here are the most likely causes:

**1. Items Not Being Returned Properly:**
```python
# ❌ WRONG - Using print instead of yield
def parse(self, response):
    for faculty in response.css('.faculty-card'):
        item = {
            'name': faculty.css('.name::text').get(),
            'email': faculty.css('.email::text').get()
        }
        print(item)  # This won't export to CSV!

# ✅ CORRECT - Must use yield or return
def parse(self, response):
    for faculty in response.css('.faculty-card'):
        item = {
            'name': faculty.css('.name::text').get(),
            'email': faculty.css('.email::text').get()
        }
        yield item  # This exports to CSV
```

**2. FEED_EXPORT Settings Not Configured:**
```python
# In settings.py or custom_settings
FEEDS = {
    'faculty_data.csv': {
        'format': 'csv',
        'encoding': 'utf8',
        'store_empty': False,  # Don't create file if no items
        'overwrite': True,
    }
}

# Or use command line
# scrapy crawl faculty_spider -o faculty_data.csv
```

**3. Item Pipeline Dropping Items:**
```python
# Check your pipelines.py
class FacultyPipeline:
    def process_item(self, item, spider):
        # ❌ WRONG - Forgot to return item
        if item.get('name'):
            item['name'] = item['name'].strip()
        # Missing return statement!

# ✅ CORRECT - Must return item
class FacultyPipeline:
    def process_item(self, item, spider):
        if item.get('name'):
            item['name'] = item['name'].strip()
        return item  # Critical!
```

**4. Spider Closing Before Items Are Processed:**
```python
# If scraping async or with delays
CONCURRENT_REQUESTS = 1  # Reduce concurrency
DOWNLOAD_DELAY = 1  # Add delay between requests

# Ensure spider waits for all items
ITEM_PIPELINES = {
    'myproject.pipelines.FacultyPipeline': 300,
}
```

**Debugging Steps:**
1. Check spider logs for "Scraped from" messages - if missing, items aren't yielded
2. Add logging to pipeline: `logger.info(f"Processing item: {item}")`
3. Verify CSV file permissions and path
4. Run with verbose logging: `scrapy crawl faculty_spider -L DEBUG`
5. Check if `DropItem` exception is being raised in pipelines

---

## Question 2: sqlite3.OperationalError: database is locked when trying to insert faculty data from multiple scrapy requests

**LLM Generated Output:**

SQLite doesn't handle concurrent writes well. This is a common issue when Scrapy's concurrent requests try to write simultaneously.

**Problem:** SQLite locks the entire database file during writes, causing conflicts.

**Solution 1: Use Connection Pooling with Timeouts**
```python
import sqlite3
from pathlib import Path

class SqlConnectionManager:
    def __init__(self, db_path: str):
        self.db_path = Path(db_path)
    
    def connection(self):
        try:
            # Add timeout and isolation level
            conn = sqlite3.connect(
                self.db_path,
                timeout=30.0,  # Wait up to 30 seconds for lock
                isolation_level='DEFERRED',  # More permissive locking
                check_same_thread=False
            )
            conn.execute("PRAGMA journal_mode=WAL;")  # Write-Ahead Logging
            conn.execute("PRAGMA busy_timeout=30000;")  # 30 second timeout
            return conn
        except sqlite3.DatabaseError as e:
            raise RuntimeError(f"Database connection error: {e}")
```

**Solution 2: Limit Scrapy Concurrency**
```python
# In settings.py
CONCURRENT_REQUESTS = 1  # Only 1 request at a time
CONCURRENT_REQUESTS_PER_DOMAIN = 1
DOWNLOAD_DELAY = 0.5  # Half second between requests

# Or use AutoThrottle
AUTOTHROTTLE_ENABLED = True
AUTOTHROTTLE_START_DELAY = 1
AUTOTHROTTLE_MAX_DELAY = 3
```

**Solution 3: Batch Inserts in Pipeline**
```python
class DatabasePipeline:
    def __init__(self):
        self.items_buffer = []
        self.buffer_size = 50  # Insert every 50 items
    
    def process_item(self, item, spider):
        self.items_buffer.append(item)
        
        if len(self.items_buffer) >= self.buffer_size:
            self.flush_buffer()
        
        return item
    
    def flush_buffer(self):
        if not self.items_buffer:
            return
        
        conn = None
        try:
            conn = self.db_manager.connection()
            cursor = conn.cursor()
            
            # Batch insert
            cursor.executemany("""
                INSERT OR REPLACE INTO faculty 
                (faculty_id, name, mail, specialization)
                VALUES (?, ?, ?, ?)
            """, [(item['id'], item['name'], item['email'], item['spec']) 
                  for item in self.items_buffer])
            
            conn.commit()
            self.items_buffer = []
        except sqlite3.OperationalError as e:
            if conn:
                conn.rollback()
            logger.error(f"Database error: {e}")
        finally:
            if conn:
                conn.close()
    
    def close_spider(self, spider):
        self.flush_buffer()  # Flush remaining items
```

**Solution 4: Use WAL Mode (Write-Ahead Logging)**
```python
# Enable WAL mode for better concurrency
conn = sqlite3.connect('faculty.db')
conn.execute("PRAGMA journal_mode=WAL;")
conn.execute("PRAGMA synchronous=NORMAL;")  # Faster writes
conn.close()

# WAL allows multiple readers + 1 writer simultaneously
```

**Solution 5: Retry Logic for Locked Database**
```python
import time

def insert_with_retry(conn, query, params, max_retries=5):
    for attempt in range(max_retries):
        try:
            cursor = conn.cursor()
            cursor.execute(query, params)
            conn.commit()
            return
        except sqlite3.OperationalError as e:
            if "database is locked" in str(e) and attempt < max_retries - 1:
                time.sleep(0.1 * (2 ** attempt))  # Exponential backoff
                continue
            raise
```

**Best Practice for Production:**
```python
# For high-concurrency scenarios, consider PostgreSQL instead
# But for Faculty Finder with moderate data, SQLite + WAL + batching works fine

# settings.py
ITEM_PIPELINES = {
    'faculty_finder.pipelines.DatabasePipeline': 300,
}

# Reduce concurrency for SQLite
CONCURRENT_REQUESTS = 2
DOWNLOAD_DELAY = 0.25
```

---

## Question 3: fastapi cors error when calling from streamlit app, added CORSMiddleware but still getting blocked by cors policy

**LLM Generated Output:**

CORS issues with FastAPI usually involve incorrect middleware configuration or origin mismatches.

**Problem:** Development vs production origin differences, or middleware order issues.

**Solution 1: Proper CORS Configuration**
```python
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(
    title="Faculty Finder API",
    description="API to fetch faculty details",
    version="1.0.0"
)

# ❌ WRONG - Middleware added AFTER routes
app.include_router(router)
app.add_middleware(CORSMiddleware, ...)  # Too late!

# ✅ CORRECT - Middleware BEFORE routes
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:8501",  # Streamlit default port
        "http://localhost:8080",
        "https://your-app.railway.app",
        "https://ds614-faculty-finder-production.up.railway.app"
    ],
    allow_credentials=True,
    allow_methods=["*"],  # Or ["GET", "POST", "PUT", "DELETE"]
    allow_headers=["*"],
)

app.include_router(router)  # Routes added after middleware
```

**Solution 2: Environment-Based Origins**
```python
import os

# For Railway deployment
ALLOWED_ORIGINS = os.getenv(
    "ALLOWED_ORIGINS",
    "http://localhost:8501,http://localhost:8080"
).split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Set in Railway env vars:
# ALLOWED_ORIGINS=https://your-streamlit-app.com,https://your-api.railway.app
```

**Solution 3: Handle Preflight Requests**
```python
from fastapi import Request
from fastapi.responses import Response

@app.options("/{path:path}")
async def options_handler(request: Request, path: str):
    """Handle CORS preflight requests"""
    return Response(
        status_code=200,
        headers={
            "Access-Control-Allow-Origin": request.headers.get("origin", "*"),
            "Access-Control-Allow-Methods": "GET, POST, PUT, DELETE, OPTIONS",
            "Access-Control-Allow-Headers": "Content-Type, Authorization",
            "Access-Control-Max-Age": "3600",
        }
    )
```

**Solution 4: Streamlit Proxy Configuration**
```python
# If Streamlit is proxying requests, configure properly
# In Streamlit app
import streamlit as st
import requests

# ❌ WRONG - Relative URL won't work across domains
response = requests.get("/api/faculty")

# ✅ CORRECT - Full URL with proper origin
API_BASE_URL = os.getenv("API_URL", "http://localhost:8000")
response = requests.get(f"{API_BASE_URL}/api/faculty")
```

**Solution 5: Railway-Specific Issues**
```python
# Railway assigns dynamic URLs, ensure CORS allows them
# Check Railway logs for actual origin being sent

# Wildcard for development (NOT for production)
if os.getenv("ENVIRONMENT") == "development":
    allow_origins = ["*"]
else:
    allow_origins = [
        "https://ds614-faculty-finder-production.up.railway.app"
    ]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allow_origins,
    allow_credentials=False if "*" in allow_origins else True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

**Debugging Checklist:**
1. ✅ Check browser console for exact CORS error and origin
2. ✅ Verify OPTIONS preflight request returns 200
3. ✅ Inspect response headers: `Access-Control-Allow-Origin` should match request origin
4. ✅ Test with curl: `curl -H "Origin: http://localhost:8501" -I http://localhost:8000/api/faculty`
5. ✅ Ensure middleware is added BEFORE routes
6. ✅ Check if `allow_credentials=True` conflicts with `allow_origins=["*"]`

---

## Question 4: streamlit app works locally but crashes on railway with "ModuleNotFoundError" even though requirements.txt has all packages

**LLM Generated Output:**

This is usually a path issue, missing system dependencies, or incorrect Python version.

**Common Causes:**

**1. Relative Import Issues:**
```python
# ❌ WRONG - Relative imports break in production
from utils.search import search_faculty
from models.tfidf import TFIDFVectorizer

# ✅ CORRECT - Absolute imports from project root
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from faculty_recommender.utils.search import search_faculty
from faculty_recommender.models.tfidf import TFIDFVectorizer
```

**2. Missing requirements.txt Entries:**
```txt
# Make sure ALL dependencies are listed with versions
streamlit>=1.28.0
pandas>=2.0.0
scikit-learn>=1.3.0
numpy>=1.24.0

# Don't forget indirect dependencies
altair>=5.0.0  # Streamlit uses this
pillow>=10.0.0  # For image handling
```

**3. Python Version Mismatch:**
```toml
# Create runtime.txt in project root
python-3.9.18

# Or in railway.toml
[build]
builder = "NIXPACKS"

[build.nixpacksPlan]
pythonVersion = "3.9"
```

**4. Incorrect Start Command:**
```toml
# railway.toml
[build]
builder = "NIXPACKS"

[deploy]
startCommand = "streamlit run app.py --server.port=$PORT --server.address=0.0.0.0"
healthcheckPath = "/"
healthcheckTimeout = 100
restartPolicyType = "ON_FAILURE"
```

**5. Missing Data Files:**
```python
# If you're loading faculty.db or other data files
import os
from pathlib import Path

# ❌ WRONG - Hardcoded path
DB_PATH = "/Users/you/project/data/faculty.db"

# ✅ CORRECT - Relative to script location
BASE_DIR = Path(__file__).parent
DB_PATH = BASE_DIR / "data" / "database" / "faculty.db"

# Ensure data files are in git (not .gitignored)
# Check .gitignore doesn't exclude *.db files
```

**6. Streamlit Config for Production:**
```toml
# Create .streamlit/config.toml
[server]
port = 8080
address = "0.0.0.0"
headless = true
enableCORS = false
enableXsrfProtection = false

[browser]
gatherUsageStats = false
```

**Complete Railway Setup:**

**Dockerfile (if using Docker):**
```dockerfile
FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8080

CMD streamlit run app.py --server.port=8080 --server.address=0.0.0.0
```

**railway.toml:**
```toml
[build]
builder = "NIXPACKS"

[deploy]
startCommand = "streamlit run app.py --server.port=$PORT --server.address=0.0.0.0 --server.headless=true"
restartPolicyType = "ON_FAILURE"
restartPolicyMaxRetries = 3
```

**Debugging Steps:**
1. Check Railway logs for exact import error
2. Verify all files are committed to git: `git ls-files`
3. Test locally with production settings: `streamlit run app.py --server.headless=true`
4. Check Python version: `python --version` locally vs Railway
5. Ensure `__init__.py` files exist in all package directories

---

## Question 5: tf-idf cosine similarity returning same scores for all faculty, search results are not ranked properly

**LLM Generated Output:**

This usually means your TF-IDF vectors are not being calculated correctly, or all documents have identical vectors.

**Common Issues:**

**1. IDF Not Calculated (All Terms Have Same Weight):**
```python
# ❌ WRONG - Only using term frequency, no IDF
def calculate_tfidf(term, doc, all_docs):
    tf = doc.count(term) / len(doc)
    return tf  # Missing IDF!

# ✅ CORRECT - Include IDF calculation
import math

def calculate_tfidf(term, doc, all_docs):
    # Term Frequency
    tf = doc.count(term) / len(doc) if len(doc) > 0 else 0
    
    # Inverse Document Frequency
    docs_with_term = sum(1 for d in all_docs if term in d)
    idf = math.log(len(all_docs) / (1 + docs_with_term))
    
    return tf * idf
```

**2. Not Normalizing Vectors:**
```python
# ❌ WRONG - Vectors not normalized, cosine similarity breaks
def cosine_similarity(vec1, vec2):
    dot_product = sum(a * b for a, b in zip(vec1, vec2))
    return dot_product  # Missing normalization!

# ✅ CORRECT - Normalize vectors
import math

def cosine_similarity(vec1, vec2):
    dot_product = sum(a * b for a, b in zip(vec1, vec2))
    
    magnitude1 = math.sqrt(sum(a * a for a in vec1))
    magnitude2 = math.sqrt(sum(b * b for b in vec2))
    
    if magnitude1 == 0 or magnitude2 == 0:
        return 0.0
    
    return dot_product / (magnitude1 * magnitude2)
```

**3. Vocabulary Mismatch Between Query and Documents:**
```python
# ❌ WRONG - Query and docs use different vocabularies
query_vector = vectorize(query, query_vocab)  # Different vocab!
doc_vector = vectorize(doc, doc_vocab)
similarity = cosine_similarity(query_vector, doc_vector)  # Meaningless!

# ✅ CORRECT - Use same vocabulary for query and documents
# Build vocabulary from all documents first
all_terms = set()
for doc in all_docs:
    all_terms.update(doc.split())

vocabulary = sorted(all_terms)  # Consistent ordering

# Then vectorize using same vocabulary
query_vector = vectorize(query, vocabulary)
doc_vector = vectorize(doc, vocabulary)
```

**4. Stopwords Not Removed:**
```python
# Common words dominate similarity
STOPWORDS = {
    'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
    'of', 'with', 'is', 'are', 'was', 'were', 'been', 'be', 'have', 'has'
}

def preprocess(text):
    # Lowercase and tokenize
    tokens = text.lower().split()
    
    # Remove stopwords
    tokens = [t for t in tokens if t not in STOPWORDS]
    
    # Remove punctuation
    tokens = [t.strip('.,!?;:()[]{}') for t in tokens]
    
    return tokens
```

**5. Field Boosting Not Applied:**
```python
# For Faculty Finder, research interests should weigh more than bio
def create_document_vector(faculty_data):
    # Combine fields with different weights
    research_text = faculty_data.get('research', '')
    specialization_text = faculty_data.get('specialization', '')
    bio_text = faculty_data.get('bio', '')
    
    # Boost important fields
    combined = (
        research_text * 3 +  # Research interests 3x weight
        specialization_text * 2 +  # Specialization 2x weight
        bio_text  # Bio 1x weight
    )
    
    return vectorize(combined, vocabulary)
```

**Complete TF-IDF Implementation:**
```python
import math
from collections import Counter

class TFIDFVectorizer:
    def __init__(self, documents):
        self.documents = documents
        self.vocabulary = self._build_vocabulary()
        self.idf = self._calculate_idf()
    
    def _build_vocabulary(self):
        """Build vocabulary from all documents"""
        all_terms = set()
        for doc in self.documents:
            all_terms.update(doc.split())
        return sorted(all_terms)
    
    def _calculate_idf(self):
        """Calculate IDF for each term"""
        idf = {}
        num_docs = len(self.documents)
        
        for term in self.vocabulary:
            docs_with_term = sum(1 for doc in self.documents if term in doc)
            idf[term] = math.log(num_docs / (1 + docs_with_term))
        
        return idf
    
    def vectorize(self, text):
        """Convert text to TF-IDF vector"""
        tokens = text.split()
        term_freq = Counter(tokens)
        
        vector = []
        for term in self.vocabulary:
            tf = term_freq.get(term, 0) / len(tokens) if len(tokens) > 0 else 0
            tfidf = tf * self.idf.get(term, 0)
            vector.append(tfidf)
        
        return vector
    
    def cosine_similarity(self, vec1, vec2):
        """Calculate cosine similarity between two vectors"""
        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        
        mag1 = math.sqrt(sum(a * a for a in vec1))
        mag2 = math.sqrt(sum(b * b for b in vec2))
        
        if mag1 == 0 or mag2 == 0:
            return 0.0
        
        return dot_product / (mag1 * mag2)
```

**Debugging TF-IDF:**
```python
# Print top terms and their TF-IDF scores
def debug_tfidf(query, doc, vectorizer):
    query_vec = vectorizer.vectorize(query)
    doc_vec = vectorizer.vectorize(doc)
    
    # Show top terms
    term_scores = sorted(
        zip(vectorizer.vocabulary, doc_vec),
        key=lambda x: x[1],
        reverse=True
    )[:10]
    
    print("Top terms in document:")
    for term, score in term_scores:
        print(f"  {term}: {score:.4f}")
    
    similarity = vectorizer.cosine_similarity(query_vec, doc_vec)
    print(f"\nCosine similarity: {similarity:.4f}")
```

---

## Question 6: railway deployment keeps failing with "Railpack" error even though i have Dockerfile, how to force it to use docker?

**LLM Generated Output:**

Railway auto-detects build methods and might ignore your Dockerfile. You need to explicitly configure it.

**Solution 1: Create railway.toml**
```toml
[build]
builder = "DOCKERFILE"
dockerfilePath = "Dockerfile"

[deploy]
startCommand = "streamlit run app.py --server.port=$PORT --server.address=0.0.0.0"
healthcheckPath = "/"
healthcheckTimeout = 100
restartPolicyType = "ON_FAILURE"
restartPolicyMaxRetries = 3
```

**Solution 2: Set Root Directory**
```bash
# If your structure is:
# Faculty-project/
#   ├── DS614-Faculty-Finder/
#   │   └── Dockerfile
#   └── DS614-Faculty-Recommender/
#       └── Dockerfile

# In Railway dashboard:
# Settings → Root Directory → DS614-Faculty-Recommender
```

**Solution 3: Verify Dockerfile is Production-Ready**
```dockerfile
FROM python:3.9-slim

WORKDIR /app

# Copy requirements first (caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Railway provides $PORT dynamically
ENV PORT=8080
EXPOSE $PORT

# Use Railway's PORT variable
CMD streamlit run app.py --server.port=$PORT --server.address=0.0.0.0 --server.headless=true
```

**Solution 4: Check .dockerignore**
```
# .dockerignore
__pycache__/
*.pyc
*.pyo
*.pyd
.Python
env/
venv/
.git/
.gitignore
*.md
!README.md
.DS_Store
*.log

# Don't ignore data files!
# !data/database/faculty.db
```

**Common Pitfalls:**
- ❌ Hardcoding port 8501 instead of using `$PORT`
- ❌ Dockerfile not in the configured root directory
- ❌ Missing `railway.toml` in repository root
- ❌ Using `CMD python app.py` instead of `streamlit run`
- ❌ `.dockerignore` excluding necessary data files

**Debugging:**
1. Check Railway build logs for detection message
2. Verify `railway.toml` is committed to git
3. Test Dockerfile locally: `docker build -t test . && docker run -p 8080:8080 test`
4. Ensure environment variables are set in Railway dashboard
5. Check Railway service settings for "Builder" option

---

## Question 7: sqlite query returns None when faculty exists, checked with DB Browser and data is there

**LLM Generated Output:**

This is usually a case sensitivity issue, incorrect query syntax, or connection problems.

**Common Issues:**

**1. Case-Sensitive LIKE Queries:**
```python
# ❌ WRONG - SQLite LIKE is case-sensitive by default
cursor.execute("""
    SELECT * FROM faculty 
    WHERE name LIKE ?
""", (f"%{search_term}%",))

# ✅ CORRECT - Use COLLATE NOCASE or LOWER()
cursor.execute("""
    SELECT * FROM faculty 
    WHERE name LIKE ? COLLATE NOCASE
""", (f"%{search_term}%",))

# Or
cursor.execute("""
    SELECT * FROM faculty 
    WHERE LOWER(name) LIKE LOWER(?)
""", (f"%{search_term}%",))
```

**2. Not Fetching Results:**
```python
# ❌ WRONG - Execute but don't fetch
cursor.execute("SELECT * FROM faculty WHERE faculty_id = ?", (id,))
return cursor  # Returns cursor object, not data!

# ✅ CORRECT - Fetch results
cursor.execute("SELECT * FROM faculty WHERE faculty_id = ?", (id,))
result = cursor.fetchone()  # Returns single row or None
return result

# Or for multiple rows
results = cursor.fetchall()  # Returns list of rows
```

**3. Connection Not Committed (for INSERTs/UPDATEs):**
```python
# ❌ WRONG - Changes not committed
conn = sqlite3.connect('faculty.db')
cursor = conn.cursor()
cursor.execute("INSERT INTO faculty VALUES (?, ?)", (id, name))
conn.close()  # Lost changes!

# ✅ CORRECT - Commit before closing
conn = sqlite3.connect('faculty.db')
cursor = conn.cursor()
cursor.execute("INSERT INTO faculty VALUES (?, ?)", (id, name))
conn.commit()  # Save changes
conn.close()
```

**4. Row Factory Not Set (Returns Tuples Instead of Dicts):**
```python
# ❌ Returns tuples: (1, 'Dr. Smith', 'smith@example.com')
conn = sqlite3.connect('faculty.db')
cursor = conn.cursor()
cursor.execute("SELECT * FROM faculty")
row = cursor.fetchone()
print(row['name'])  # TypeError: tuple indices must be integers!

# ✅ CORRECT - Use Row factory for dict-like access
import sqlite3

conn = sqlite3.connect('faculty.db')
conn.row_factory = sqlite3.Row  # Enable dict-like access
cursor = conn.cursor()
cursor.execute("SELECT * FROM faculty")
row = cursor.fetchone()
print(row['name'])  # Works!
print(dict(row))  # Convert to dict
```

**5. Incorrect Parameter Binding:**
```python
# ❌ WRONG - String formatting (SQL injection risk!)
faculty_id = "FAC001"
cursor.execute(f"SELECT * FROM faculty WHERE faculty_id = '{faculty_id}'")

# ❌ WRONG - Wrong placeholder style
cursor.execute("SELECT * FROM faculty WHERE faculty_id = %s", (faculty_id,))

# ✅ CORRECT - Use ? placeholders
cursor.execute("SELECT * FROM faculty WHERE faculty_id = ?", (faculty_id,))
```

**Complete Query Example:**
```python
from pathlib import Path
import sqlite3

class FacultyDatabase:
    def __init__(self, db_path: str):
        self.db_path = Path(db_path)
    
    def get_faculty_by_id(self, faculty_id: str):
        """Get single faculty member by ID"""
        conn = None
        try:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row  # Dict-like access
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT faculty_id, name, mail, specialization, research
                FROM faculty
                WHERE faculty_id = ?
            """, (faculty_id,))
            
            row = cursor.fetchone()
            return dict(row) if row else None
        
        except sqlite3.Error as e:
            logger.error(f"Database error: {e}")
            return None
        
        finally:
            if conn:
                conn.close()
    
    def search_faculty(self, query: str):
        """Search faculty by name or specialization"""
        conn = None
        try:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            search_pattern = f"%{query}%"
            cursor.execute("""
                SELECT faculty_id, name, mail, specialization
                FROM faculty
                WHERE 
                    name LIKE ? COLLATE NOCASE OR
                    specialization LIKE ? COLLATE NOCASE OR
                    research LIKE ? COLLATE NOCASE
                LIMIT 50
            """, (search_pattern, search_pattern, search_pattern))
            
            rows = cursor.fetchall()
            return [dict(row) for row in rows]
        
        except sqlite3.Error as e:
            logger.error(f"Search error: {e}")
            return []
        
        finally:
            if conn:
                conn.close()
```

**Debugging Queries:**
```python
# Enable SQL logging
import logging
logging.basicConfig(level=logging.DEBUG)

# Or use connection trace
def trace_callback(statement):
    print(f"Executing: {statement}")

conn = sqlite3.connect('faculty.db')
conn.set_trace_callback(trace_callback)
```

---

## Question 8: pandas read_csv works locally but fails on railway with "FileNotFoundError" even though file is in repo

**LLM Generated Output:**

This is a path issue - relative paths work locally but break in production.

**Problem:** Working directory is different in Railway vs local.

**Solution 1: Use Absolute Paths Relative to Script**
```python
# ❌ WRONG - Relative to current working directory
df = pd.read_csv('data/cleaned/faculty_cleaned.csv')

# ✅ CORRECT - Relative to script location
import os
from pathlib import Path

BASE_DIR = Path(__file__).parent
CSV_PATH = BASE_DIR / 'data' / 'cleaned' / 'faculty_cleaned.csv'

df = pd.read_csv(CSV_PATH)
```

**Solution 2: Check File Exists Before Reading**
```python
import os
from pathlib import Path

def load_faculty_data():
    base_dir = Path(__file__).parent
    csv_path = base_dir / 'data' / 'cleaned' / 'faculty_cleaned.csv'
    
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Faculty data not found at {csv_path}\n"
            f"Current directory: {os.getcwd()}\n"
            f"Files in data/: {list((base_dir / 'data').glob('**/*'))}"
        )
    
    return pd.read_csv(csv_path)
```

**Solution 3: Ensure File is in Git (Not .gitignored)**
```bash
# Check if file is tracked
git ls-files | grep faculty_cleaned.csv

# If missing, check .gitignore
cat .gitignore

# Common mistake - excluding all CSVs
# *.csv  # Remove this line!

# Or explicitly include
!data/cleaned/*.csv
```

**Solution 4: Use Environment Variables for Paths**
```python
import os
from pathlib import Path

# Default to relative path, allow override
DATA_DIR = os.getenv('DATA_DIR', Path(__file__).parent / 'data')
CSV_PATH = Path(DATA_DIR) / 'cleaned' / 'faculty_cleaned.csv'

df = pd.read_csv(CSV_PATH)

# In Railway, set env var if needed:
# DATA_DIR=/app/data
```

**Solution 5: Bundle Data in Docker Image**
```dockerfile
FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Explicitly copy data directory
COPY data/ ./data/

# Copy rest of application
COPY . .

CMD ["python", "app.py"]
```

**Debugging File Paths:**
```python
import os
from pathlib import Path

def debug_paths():
    print(f"Current working directory: {os.getcwd()}")
    print(f"Script location: {Path(__file__).parent}")
    print(f"Files in current dir: {os.listdir('.')}")
    
    data_dir = Path(__file__).parent / 'data'
    if data_dir.exists():
        print(f"Files in data/: {list(data_dir.glob('**/*'))}")
    else:
        print("data/ directory not found!")

# Call this in your app to debug Railway deployment
debug_paths()
```

---

## Notes

- These are real debugging scenarios from Faculty Finder development
- Tech stack: SQLite, Scrapy, FastAPI, Streamlit, Railway, custom TF-IDF/VSM
- Focus on production deployment issues, database queries, and search algorithm bugs
- Date: February 2026
