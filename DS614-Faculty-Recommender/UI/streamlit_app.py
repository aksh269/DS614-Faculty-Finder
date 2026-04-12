import streamlit as st
import sys
from pathlib import Path
import requests
from bs4 import BeautifulSoup

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from app.engine import search
from recommender.query_parser import parse_query
from recommender.llm_layer import is_llm_available, configure_gemini
from config.settings import META_FILE, FAISS_INDEX_FILE, INDEX_FILE

# ── Auto-Initialize Index if missing (Critical for fresh Cloud deployments) ──
def initialize_indices():
    # Defensive import to avoid issues if the module is partially loaded
    try:
        import recommender.engine_builder as ib
        if not (META_FILE.exists() and FAISS_INDEX_FILE.exists() and INDEX_FILE.exists()):
            with st.spinner("🚀 First-time setup: Building AI search indices..."):
                ib.build_all_indices()
                st.success("✅ AI Indices built successfully!")
    except Exception as e:
        st.sidebar.error(f"⚠️ Index Auto-build failed: {e}")

initialize_indices()

# ── Sidebar Manual Maintenance ──
with st.sidebar:
    st.markdown("---")
    st.markdown("### 🛠 Maintenance")
    if st.button("🔄 Rebuild AI Index", help="Force a rebuild of BERT & FAISS indices"):
        with st.spinner("🧠 Rebuilding indices..."):
            try:
                import recommender.engine_builder as ib
                ib.build_all_indices()
                st.sidebar.success("✅ Index rebuilt!")
                st.rerun()
            except Exception as e:
                st.sidebar.error(f"❌ Build failed: {e}")

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ScholarMatch",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Load CSS ──────────────────────────────────────────────────────────────────
css_path = Path(__file__).parent / "styles.css"
if css_path.exists():
    with open(css_path) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Extra styles for AI-upgrade UI elements
st.markdown("""
<style>
/* ── AI Badge ── */
.ai-badge {
    display: inline-flex; align-items: center; gap: 6px;
    background: linear-gradient(135deg, #6366f1, #8b5cf6);
    color: #fff; font-size: 0.72rem; font-weight: 600;
    padding: 3px 10px; border-radius: 20px; letter-spacing: 0.04em;
    margin-bottom: 8px;
}

/* ── Expanded keywords pills ── */
.expanded-tag {
    display: inline-block;
    background: rgba(99,102,241,0.15); color: #6366f1;
    border: 1px solid rgba(99,102,241,0.3);
    font-size: 0.72rem; padding: 2px 9px; border-radius: 20px;
    margin: 2px 3px 2px 0;
}

/* ── Score breakdown bar ── */
.score-breakdown { margin-top: 8px; font-size: 0.72rem; color: #94a3b8; }
.score-breakdown span { color: #c7d2fe; font-weight: 600; }

/* ── LLM explanation box ── */
.explanation-box {
    background: rgba(99,102,241,0.08);
    border-left: 3px solid #6366f1;
    border-radius: 6px;
    padding: 10px 14px;
    margin-top: 12px;
    font-size: 0.85rem;
    color: #cbd5e1;
    line-height: 1.5;
}
.explanation-label {
    font-size: 0.7rem; font-weight: 700; color: #818cf8;
    letter-spacing: 0.06em; text-transform: uppercase; margin-bottom: 4px;
}

/* ── Search mode chip ── */
.mode-chip {
    display:inline-flex; align-items:center; gap:5px;
    background: rgba(16,185,129,0.15); color:#34d399;
    border:1px solid rgba(52,211,153,0.3);
    font-size:0.73rem; font-weight:600; padding:3px 10px;
    border-radius:20px; margin-left:10px;
}
</style>
""", unsafe_allow_html=True)


# ── Helper Functions ──────────────────────────────────────────────────────────

@st.cache_data(ttl=3600)
def get_faculty_photo(name, profile_url):
    """Scrape faculty photo from DAIICT profile page, fallback to avatar."""
    if not profile_url or profile_url == "-":
        return f"https://ui-avatars.com/api/?name={name.replace(' ', '+')}&size=200&background=6366f1&color=fff&bold=true"
    try:
        response = requests.get(profile_url, timeout=5)
        if response.status_code == 200:
            soup = BeautifulSoup(response.content, 'html.parser')
            img = (
                soup.find('img', class_='faculty-photo') or
                soup.find('img', alt=lambda x: x and 'photo' in x.lower()) or
                soup.select_one('.profile-image img') or
                soup.select_one('img[src*="faculty"]') or
                soup.select_one('img[src*="staff"]')
            )
            if img and img.get('src'):
                img_url = img['src']
                if img_url.startswith('/'):
                    img_url = f"https://www.daiict.ac.in{img_url}"
                elif not img_url.startswith('http'):
                    base_url = "/".join(profile_url.split("/")[:3])
                    img_url = f"{base_url}/{img_url.lstrip('/')}"
                return img_url
    except Exception:
        pass
    return f"https://ui-avatars.com/api/?name={name.replace(' ', '+')}&size=200&background=6366f1&color=fff&bold=true"


def extract_keywords(text, max_keywords=8):
    if not text or text == "-":
        return []
    keywords = []
    if text.startswith('[') and text.endswith(']'):
        try:
            import ast
            items = ast.literal_eval(text)
            if isinstance(items, list):
                keywords = [str(k).strip() for k in items if k]
        except Exception:
            pass
    if not keywords:
        for delimiter in [',', ';', '|', '\n']:
            if delimiter in text:
                keywords = [k.strip() for k in text.split(delimiter) if k.strip()]
                break
    if not keywords:
        words = text.split()
        keywords = [' '.join(words[i:i+3]) for i in range(0, min(len(words), 15), 3)]
    cleaned = []
    for kw in keywords:
        kw = kw.replace('[', '').replace(']', '').replace("'", '').replace('"', '').strip()
        if kw and 'not_available' not in kw.lower() and len(kw) > 2:
            cleaned.append(kw)
            if len(cleaned) >= max_keywords:
                break
    return cleaned


# ── Header ────────────────────────────────────────────────────────────────────
# llm_on is checked inside the sidebar now, but we need a generic check here for the hero banner
llm_on_header = is_llm_available()

st.markdown(f"""
<div class="hero">
    <h1>🎓 <span class="gradient-text">ScholarMatch AI</span></h1>
    <p>
        Discover faculty members matching your research interests with AI-powered precision
        <span class="mode-chip">
            {"✨ Hybrid AI + LLM" if llm_on_header else "⚡ Hybrid AI (BERT + TF-IDF)"}
        </span>
    </p>
</div>
""", unsafe_allow_html=True)


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🤖 AI Configuration")
    user_api_key = st.text_input("Gemini API Key (Optional)", type="password", help="Enter your Gemini API key to enable LLM-powered query expansion and ranking explanations.")
    if user_api_key:
        configure_gemini(user_api_key)

    llm_on = is_llm_available()
    
    st.markdown("## ⚙️ Display Settings")
    use_gemini       = st.toggle("Enable Gemini (AI Assistant)", llm_on, disabled=not llm_on, help="Requires GEMINI_API_KEY. Uncheck to run pure Hybrid Search.")
    show_scores      = st.toggle("Show Match Scores",     True,  help="Display similarity scores")
    show_keywords    = st.toggle("Show Keywords",         True,  help="Display research keywords as tags")
    show_explanation = st.toggle("Show AI Explanation",   use_gemini, disabled=not use_gemini, help="Show why each result was recommended")
    show_score_breakdown = st.toggle("Show Score Breakdown", False, help="Show TF-IDF vs BERT individual scores")

    st.markdown("---")
    st.markdown("### 💡 Smart Search Tips")
    st.markdown("""
    **Specify number of results:**
    - "top 3 machine learning"
    - "5 best computer vision"
    - "first 10 IoT experts"

    **Natural language (with LLM):**
    - "I want someone working in AI for healthcare"
    - "faculty doing speech and language research"

    **Or just search:**
    - "machine learning"
    - "VLSI design"

    *Default: 5 results*
    """)


# ── Search Input ──────────────────────────────────────────────────────────────
col_search_1, col_search_2 = st.columns([4, 1])

with col_search_1:
    query = st.text_input(
        "🔍 Search by research interest or expertise",
        placeholder="e.g., top 3 machine learning, 'AI for healthcare', computer vision...",
        label_visibility="collapsed"
    )

with col_search_2:
    search_clicked = st.button("🔎 Search", use_container_width=True, type="primary")


# ── Search Logic ──────────────────────────────────────────────────────────────
if search_clicked or query:

    if not query.strip():
        if search_clicked:
            st.warning("⚠️ Please enter a research interest or keyword")
        st.stop()

    clean_query, top_k = parse_query(query)

    with st.spinner("🧠 Running AI-powered hybrid search…"):
        results = search(clean_query, top_k=top_k, use_gemini=use_gemini)

    if not results:
        st.error("❌ No matching faculty found. Try different keywords.")
        st.stop()

    # ── Show expanded keywords if LLM was used ────────────────────────────
    expanded_kws = results[0].get("expanded_keywords", []) if results else []
    if expanded_kws and llm_on:
        kw_pills = "".join(f'<span class="expanded-tag">🔑 {kw}</span>' for kw in expanded_kws)
        st.markdown(
            f'<div style="margin-bottom:10px;">'
            f'<span class="ai-badge">✨ LLM Query Expansion</span> '
            f'Your query was expanded to: {kw_pills}'
            f'</div>',
            unsafe_allow_html=True
        )

    st.success(f"✅ Found {len(results)} matching faculty member{'s' if len(results) > 1 else ''}"
               f" — Hybrid AI (BERT + TF-IDF)")

    # ── Results ───────────────────────────────────────────────────────────
    for idx, faculty in enumerate(results, start=1):
        with st.container():
            st.markdown(f'<div class="faculty-card" style="animation-delay: {idx * 0.08}s;">', unsafe_allow_html=True)

            col_avatar, col_info, col_score = st.columns([1.5, 4, 1.5])

            # ── Avatar ────────────────────────────────────────────────────
            with col_avatar:
                name        = faculty.get('name', 'Faculty')
                profile_url = faculty.get("profile_url", "")
                photo_url   = get_faculty_photo(name, profile_url)
                st.markdown(f"""
                <div class="avatar-container">
                    <img src="{photo_url}" class="avatar-img" alt="{name}">
                </div>
                """, unsafe_allow_html=True)

            # ── Faculty Info ───────────────────────────────────────────────
            with col_info:
                st.markdown('<div class="faculty-info">', unsafe_allow_html=True)

                rank_badge = {1: "🥇", 2: "🥈", 3: "🥉"}.get(idx, "")
                faculty_name = faculty.get('name', 'Unknown')
                st.markdown(f"""
                <div class="faculty-name">
                    <span class="name-text">{faculty_name}</span>
                    <span class="rank-badge">{rank_badge}</span>
                </div>
                """, unsafe_allow_html=True)

                email = faculty.get("mail", "")
                if email:
                    st.markdown(f'''
                    <div class="email-info">
                        <a href="mailto:{email}">
                            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="vertical-align:middle;margin-right:4px;">
                                <path d="M4 4h16c1.1 0 2 .9 2 2v12c0 1.1-.9 2-2 2H4c-1.1 0-2-.9-2-2V6c0-1.1.9-2 2-2z"></path>
                                <polyline points="22,6 12,13 2,6"></polyline>
                            </svg>
                            {email}
                        </a>
                    </div>
                    ''', unsafe_allow_html=True)

                if show_keywords:
                    specialization = faculty.get("specialization", "")
                    research       = faculty.get("research", "")
                    combined_text  = f"{specialization} {research}"
                    keywords       = extract_keywords(combined_text)
                    if keywords:
                        keywords_html = "".join([f'<span class="keyword-tag">{kw}</span>' for kw in keywords])
                        st.markdown(f'<div class="keywords-container">{keywords_html}</div>', unsafe_allow_html=True)

                # ── LLM Explanation ───────────────────────────────────────
                if show_explanation:
                    explanation = faculty.get("explanation", "")
                    if explanation:
                        st.markdown(f"""
                        <div class="explanation-box">
                            <div class="explanation-label">🤖 AI Insight</div>
                            {explanation}
                        </div>
                        """, unsafe_allow_html=True)

                if profile_url and profile_url != "-":
                    st.markdown(f"""
                    <div style="margin-top: 1.5rem;">
                        <a href="{profile_url}" target="_blank" style="font-size:0.9rem;letter-spacing:0.02em;">
                            Full Profile Details ↗
                        </a>
                    </div>
                    """, unsafe_allow_html=True)

                st.markdown('</div>', unsafe_allow_html=True)

            # ── Score Column ───────────────────────────────────────────────
            with col_score:
                if show_scores:
                    score         = faculty.get("score", 0)
                    tfidf_score   = faculty.get("tfidf_score", 0)
                    bert_score    = faculty.get("bert_score", 0)
                    score_percent = int(score * 100)
                    st.markdown(f"""
                    <div class="score-container">
                        <div class="score-label">MATCH SCORE</div>
                        <div class="score-value">{score_percent}%</div>
                        <div class="score-bar">
                            <div class="score-fill" style="width:{score_percent}%"></div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                    if show_score_breakdown:
                        st.markdown(f"""
                        <div class="score-breakdown">
                            TF-IDF:&nbsp;<span>{int(tfidf_score*100)}%</span><br>
                            BERT:&nbsp;&nbsp;&nbsp;<span>{int(bert_score*100)}%</span>
                        </div>
                        """, unsafe_allow_html=True)

            st.markdown('</div>', unsafe_allow_html=True)


# ── Footer ────────────────────────────────────────────────────────────────────
if not search_clicked:
    st.markdown("---")
    st.markdown("""
    <div style="text-align:center;color:#666;padding:20px;">
        <p>🎯 Enter your research interests above to find matching faculty members</p>
        <p style="font-size:0.9em;">
            Powered by <strong>BERT + FAISS + Hybrid Search</strong>
            &nbsp;·&nbsp; Team "The Data Engineers"
        </p>
    </div>
    """, unsafe_allow_html=True)
