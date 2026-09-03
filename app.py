# -*- coding: utf-8 -*-
"""
=============================================================
  Movie Recommendation System  -  Step 5: Streamlit App
=============================================================
Run with:
    streamlit run app.py
"""

import os
import sys
import time
import math
import requests

# pyrefly: ignore [missing-import]
import streamlit as st

# ── path setup ────────────────────────────────────────────────────────────────
APP_DIR  = os.path.dirname(os.path.abspath(__file__))
SRC_DIR  = os.path.join(APP_DIR, "src")
sys.path.insert(0, SRC_DIR)

# pyrefly: ignore [missing-import]
from recommendation_engine import MovieRecommender
# pyrefly: ignore [missing-import]
from tmdb_helper import get_poster_url, FALLBACK_POSTER

# ── constants ─────────────────────────────────────────────────────────────────
TMDB_PAGE = "https://www.imdb.com/title/"
STAR = "\u2605"
STAR_EMPTY = "\u2606"


# ══════════════════════════════════════════════════════════════════════════════
#  PAGE CONFIG  (must be first Streamlit call)
# ══════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="CineMatch – Movie Recommender",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ══════════════════════════════════════════════════════════════════════════════
#  GLOBAL CSS
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
/* ── Google Font ─────────────────────────────────────────────────── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&family=Playfair+Display:wght@700&display=swap');

/* ── Root tokens ─────────────────────────────────────────────────── */
:root {
    --bg-base:       #070b14;
    --bg-card:       rgba(20,26,46,0.82);
    --bg-card-hover: rgba(28,36,62,0.95);
    --accent:        #f5c518;
    --accent2:       #e87c1e;
    --accent-blue:   #4fa3e0;
    --text-primary:  #eef0f5;
    --text-muted:    #8b93ab;
    --border:        rgba(255,255,255,0.08);
    --glow:          rgba(245,197,24,0.18);
    --radius:        16px;
    --radius-sm:     10px;
}

/* ── Full-page dark background ───────────────────────────────────── */
html, body, [data-testid="stAppViewContainer"], [data-testid="stMain"] {
    background-color: var(--bg-base) !important;
}
[data-testid="stAppViewContainer"]::before {
    content: "";
    position: fixed; inset: 0;
    background:
        radial-gradient(ellipse 80% 50% at 20% 20%, rgba(79,163,224,0.06) 0%, transparent 60%),
        radial-gradient(ellipse 60% 40% at 80% 80%, rgba(245,197,24,0.05) 0%, transparent 50%);
    pointer-events: none; z-index: 0;
}

/* ── Sidebar ─────────────────────────────────────────────────────── */
[data-testid="stSidebar"] {
    background: rgba(10,14,26,0.97) !important;
    border-right: 1px solid var(--border);
}
[data-testid="stSidebar"] * { color: var(--text-primary) !important; }

/* ── Global typography ────────────────────────────────────────────── */
*, p, div, span, label, h1, h2, h3 {
    font-family: 'Inter', sans-serif !important;
    color: var(--text-primary);
}

/* ── Hero section ─────────────────────────────────────────────────── */
.hero-wrap {
    background: linear-gradient(135deg, rgba(10,14,26,0.98) 0%, rgba(20,28,58,0.95) 100%);
    border: 1px solid var(--border);
    border-radius: 24px;
    padding: 52px 48px 44px;
    margin-bottom: 32px;
    position: relative;
    overflow: hidden;
}
.hero-wrap::before {
    content:"";
    position:absolute; inset:0;
    background: radial-gradient(ellipse 60% 80% at 50% -20%, rgba(245,197,24,0.12) 0%, transparent 70%);
    pointer-events: none;
}
.hero-badge {
    display: inline-block;
    background: linear-gradient(90deg, rgba(245,197,24,0.15), rgba(232,124,30,0.15));
    border: 1px solid rgba(245,197,24,0.35);
    border-radius: 50px;
    padding: 4px 16px;
    font-size: 12px;
    font-weight: 600;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    color: var(--accent) !important;
    margin-bottom: 18px;
}
.hero-title {
    font-family: 'Playfair Display', serif !important;
    font-size: clamp(2.4rem, 4vw, 3.6rem) !important;
    font-weight: 700 !important;
    background: linear-gradient(135deg, #ffffff 30%, var(--accent) 120%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    line-height: 1.15 !important;
    margin: 0 0 10px !important;
}
.hero-sub {
    font-size: 1.05rem !important;
    color: var(--text-muted) !important;
    margin: 0 0 32px !important;
    line-height: 1.6 !important;
}
.stat-chip {
    display: inline-flex; align-items: center; gap: 6px;
    background: rgba(255,255,255,0.05);
    border: 1px solid var(--border);
    border-radius: 50px;
    padding: 6px 14px;
    font-size: 13px;
    font-weight: 500;
    color: var(--text-muted) !important;
    margin-right: 10px; margin-top: 8px;
}
.stat-chip b { color: var(--accent) !important; }

/* ── Search input styling ─────────────────────────────────────────── */
[data-testid="stTextInput"] input {
    background: rgba(255,255,255,0.05) !important;
    border: 1.5px solid rgba(255,255,255,0.14) !important;
    border-radius: 12px !important;
    color: var(--text-primary) !important;
    font-size: 1rem !important;
    padding: 14px 18px !important;
    transition: border-color 0.25s, box-shadow 0.25s;
}
[data-testid="stTextInput"] input:focus {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 3px rgba(245,197,24,0.18) !important;
}

/* ── Buttons ──────────────────────────────────────────────────────── */
[data-testid="stButton"] > button {
    background: linear-gradient(135deg, #f5c518 0%, #e87c1e 100%) !important;
    color: #070b14 !important;
    font-weight: 700 !important;
    font-size: 1rem !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 12px 32px !important;
    cursor: pointer !important;
    transition: transform 0.2s, box-shadow 0.2s !important;
    box-shadow: 0 4px 20px rgba(245,197,24,0.30) !important;
}
[data-testid="stButton"] > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 28px rgba(245,197,24,0.45) !important;
}
[data-testid="stButton"] > button:active { transform: translateY(0) !important; }

/* ── Selectbox / slider ───────────────────────────────────────────── */
[data-testid="stSelectbox"] > div > div,
[data-testid="stSlider"] {
    background: rgba(255,255,255,0.04) !important;
    border-radius: 10px !important;
}

/* ── Movie cards ─────────────────────────────────────────────────── */
.movie-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 0;
    overflow: hidden;
    transition: transform 0.28s cubic-bezier(.34,1.56,.64,1),
                border-color 0.25s, box-shadow 0.28s;
    position: relative;
    height: 100%;
}
.movie-card:hover {
    transform: translateY(-6px) scale(1.01);
    border-color: rgba(245,197,24,0.45);
    box-shadow: 0 20px 45px rgba(0,0,0,0.55), 0 0 20px var(--glow);
    background: var(--bg-card-hover);
}
.movie-card:hover .rank-badge { background: var(--accent); color: #070b14 !important; }

.poster-wrap {
    width: 100%; aspect-ratio: 2/3; overflow: hidden;
    background: rgba(15,20,40,0.8);
    position: relative;
}
.poster-wrap img {
    width: 100%; height: 100%;
    object-fit: cover;
    display: block;
    transition: transform 0.35s ease;
}
.movie-card:hover .poster-wrap img { transform: scale(1.06); }

.poster-overlay {
    position: absolute; inset: 0;
    background: linear-gradient(to top, rgba(7,11,20,0.97) 0%, rgba(7,11,20,0.2) 55%, transparent 100%);
}
.rank-badge {
    position: absolute; top: 12px; left: 12px;
    background: rgba(7,11,20,0.85);
    border: 1px solid var(--border);
    color: var(--accent) !important;
    font-weight: 800; font-size: 13px;
    border-radius: 8px;
    padding: 4px 10px;
    backdrop-filter: blur(6px);
    transition: background 0.25s, color 0.25s;
}
.sim-pill {
    position: absolute; top: 12px; right: 12px;
    background: linear-gradient(135deg, rgba(245,197,24,0.2), rgba(232,124,30,0.2));
    border: 1px solid rgba(245,197,24,0.4);
    border-radius: 50px; padding: 3px 10px;
    font-size: 12px; font-weight: 600;
    color: var(--accent) !important;
    backdrop-filter: blur(6px);
}
.card-body {
    padding: 14px 16px 18px;
}
.card-title {
    font-size: 15px !important; font-weight: 700 !important;
    color: var(--text-primary) !important;
    margin: 0 0 5px !important; line-height: 1.3 !important;
    display: -webkit-box; -webkit-line-clamp: 2;
    -webkit-box-orient: vertical; overflow: hidden;
}
.card-genres {
    font-size: 11.5px !important;
    color: var(--accent-blue) !important;
    margin: 0 0 10px !important; font-weight: 500 !important;
    display: -webkit-box; -webkit-line-clamp: 1;
    -webkit-box-orient: vertical; overflow: hidden;
}
.card-meta {
    display: flex; align-items: center; gap: 10px;
    flex-wrap: wrap; margin-top: 6px;
}
.meta-chip {
    display: flex; align-items: center; gap: 4px;
    background: rgba(255,255,255,0.06);
    border-radius: 6px; padding: 3px 8px;
    font-size: 12px; font-weight: 600;
}
.meta-chip.rating { color: var(--accent) !important; }
.meta-chip.votes  { color: var(--text-muted) !important; }
.card-overview {
    font-size: 12.5px !important;
    color: var(--text-muted) !important;
    margin: 10px 0 0 !important; line-height: 1.55 !important;
    display: -webkit-box; -webkit-line-clamp: 3;
    -webkit-box-orient: vertical; overflow: hidden;
}
.imdb-link {
    display: inline-block; margin-top: 10px;
    font-size: 11px; font-weight: 700;
    color: var(--accent) !important;
    text-decoration: none; letter-spacing: 0.5px;
}

/* ── Input movie detail panel ─────────────────────────────────────── */
.detail-panel {
    background: linear-gradient(135deg, rgba(18,24,48,0.95), rgba(14,18,36,0.98));
    border: 1px solid rgba(245,197,24,0.22);
    border-radius: 20px;
    padding: 28px 32px;
    margin-bottom: 32px;
    display: flex; gap: 28px;
    align-items: flex-start;
}
.detail-poster {
    width: 130px; flex-shrink: 0;
    border-radius: 12px; overflow: hidden;
    box-shadow: 0 8px 30px rgba(0,0,0,0.55);
}
.detail-poster img { width: 100%; display: block; }
.detail-info h2 {
    font-family: 'Playfair Display', serif !important;
    font-size: 1.7rem !important; font-weight: 700 !important;
    margin: 0 0 8px !important; line-height: 1.2 !important;
    color: var(--text-primary) !important;
}
.detail-genres {
    font-size: 13px; font-weight: 500;
    color: var(--accent-blue) !important;
    margin-bottom: 10px;
}
.detail-stats {
    display: flex; flex-wrap: wrap; gap: 8px; margin-bottom: 12px;
}
.stat-pill {
    background: rgba(255,255,255,0.07); border: 1px solid var(--border);
    border-radius: 50px; padding: 5px 14px;
    font-size: 13px; font-weight: 600;
}
.detail-overview {
    font-size: 14px !important; line-height: 1.7 !important;
    color: var(--text-muted) !important;
    margin: 0 !important;
}

/* ── Section headers ──────────────────────────────────────────────── */
.section-head {
    display: flex; align-items: center; gap: 14px;
    margin: 36px 0 22px;
}
.section-head h3 {
    font-size: 1.4rem !important; font-weight: 700 !important;
    margin: 0 !important; color: var(--text-primary) !important;
}
.section-line {
    flex: 1; height: 1px;
    background: linear-gradient(to right, rgba(245,197,24,0.35), transparent);
}
.section-count {
    background: rgba(245,197,24,0.12); border: 1px solid rgba(245,197,24,0.3);
    color: var(--accent) !important; font-size: 12px; font-weight: 700;
    border-radius: 50px; padding: 3px 12px;
}

/* ── Search result rows ───────────────────────────────────────────── */
.search-row {
    background: rgba(255,255,255,0.03);
    border: 1px solid var(--border);
    border-radius: 10px; padding: 12px 16px;
    margin-bottom: 8px;
    display: flex; align-items: center; gap: 14px;
    transition: background 0.2s, border-color 0.2s;
}
.search-row:hover {
    background: rgba(245,197,24,0.07);
    border-color: rgba(245,197,24,0.3);
}
.search-row-title { font-size: 15px; font-weight: 600; flex: 1; }
.search-row-genre { font-size: 12px; color: var(--accent-blue) !important; }
.search-row-rating { font-size: 13px; color: var(--accent) !important; font-weight: 700; }

/* ── Pipeline diagram ─────────────────────────────────────────────── */
.pipeline {
    display: flex; align-items: center; justify-content: center;
    flex-wrap: wrap; gap: 0; margin: 24px 0;
}
.pipe-node {
    background: rgba(255,255,255,0.04);
    border: 1px solid var(--border); border-radius: 12px;
    padding: 14px 20px; text-align: center; min-width: 110px;
}
.pipe-node-icon { font-size: 22px; margin-bottom: 4px; }
.pipe-node-label {
    font-size: 12px; font-weight: 600;
    color: var(--text-primary) !important; margin: 0;
}
.pipe-node-sub {
    font-size: 10px; color: var(--text-muted) !important; margin: 0;
}
.pipe-arrow { font-size: 20px; color: var(--accent) !important; padding: 0 4px; }

/* ── No results ───────────────────────────────────────────────────── */
.no-results {
    text-align: center; padding: 60px 20px;
    color: var(--text-muted);
}
.no-results-icon { font-size: 56px; margin-bottom: 16px; }
.no-results h3 { font-size: 1.3rem !important; color: var(--text-primary) !important; }

/* ── Toast-like info box ──────────────────────────────────────────── */
.info-box {
    background: rgba(79,163,224,0.1); border: 1px solid rgba(79,163,224,0.3);
    border-radius: 10px; padding: 12px 18px;
    font-size: 14px; color: var(--accent-blue) !important;
    margin-bottom: 20px;
}

/* ── Hide Streamlit chrome ────────────────────────────────────────── */
#MainMenu, footer, header { visibility: hidden; }
[data-testid="stToolbar"] { display: none; }
.stDeployButton { display: none; }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def poster_url(movie_dict):
    return get_poster_url(movie_dict)


def star_rating(score, max_score=10):
    """Return filled/empty stars for a 0-10 rating scaled to 5 stars."""
    if score is None:
        return "N/A"
    filled = round((float(score) / max_score) * 5)
    return STAR * filled + STAR_EMPTY * (5 - filled)


def fmt_votes(v):
    if v is None:
        return "N/A"
    v = int(v)
    if v >= 1_000_000:
        return "{:.1f}M".format(v / 1_000_000)
    if v >= 1_000:
        return "{:.0f}K".format(v / 1_000)
    return str(v)


def release_year(rd):
    try:
        return str(rd)[:4]
    except Exception:
        return "N/A"


def genres_display(g):
    if not g or str(g) in ("None", "nan"):
        return "Unknown"
    parts = str(g).split()
    return "  ·  ".join(parts[:4])


# ══════════════════════════════════════════════════════════════════════════════
#  LOAD MODEL  (cached across re-runs)
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_resource(show_spinner=False)
def load_recommender():
    try:
        # pyrefly: ignore [missing-import]
        from adaptive_recommender import AdaptiveRecommender
        return AdaptiveRecommender()
    except Exception:
        return MovieRecommender()

with st.spinner("🎬 Loading CineMatch engine…"):
    try:
        rec = load_recommender()
        loaded_ok = True
    except Exception as e:
        st.error("**Failed to load model:** " + str(e))
        st.info("Run `python src/build_model.py` first, then restart Streamlit.")
        st.stop()


# ══════════════════════════════════════════════════════════════════════════════
#  SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("""
    <div style="text-align:center; padding: 10px 0 24px;">
        <div style="font-size:42px;">🎬</div>
        <div style="font-family:'Playfair Display',serif; font-size:22px; font-weight:700;
                    background:linear-gradient(135deg,#fff,#f5c518);
                    -webkit-background-clip:text; -webkit-text-fill-color:transparent;">
            CineMatch
        </div>
        <div style="font-size:12px; color:#8b93ab; margin-top:4px;">
            AI Movie Recommender
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### ⚙️ Settings")
    
    # Model Indicator
    engine_name = "Adaptive Semantic Metaheuristic" if getattr(rec, 'model_mode', '') == "asmr" else "Semantic Hybrid"
    st.markdown(f"""
        <div style="background-color:rgba(255,255,255,0.05); padding:10px; border-radius:8px; margin-bottom:15px; font-size:13px;">
          <span style="color:#8b93ab;">Decision Engine:</span><br/>
          <strong style="color:var(--text); font-weight:600;">{engine_name}</strong>
          <div style="font-size:11px; color:#8b93ab; margin-top:4px;">
            Powered by Sentence Transformers & Differential Evolution (ASMR)
          </div>
        </div>
    """, unsafe_allow_html=True)
    
    debug_mode = st.toggle("🛠️ Developer Debug Mode", value=False)
    
    n_recs = st.slider(
        "Number of recommendations", min_value=3, max_value=20, value=10, step=1
    )

    show_overview = st.toggle("Show overviews on cards", value=False)

    st.markdown("---")
    st.markdown("### 🔍 Quick Search")
    quick_q = st.text_input(
        "Search for a title", placeholder="e.g. batman, star wars…",
        key="sidebar_search", label_visibility="collapsed"
    )
    if quick_q.strip():
        hits = rec.search_movies(quick_q.strip(), limit=8)
        if hits:
            for h in hits:
                yr = release_year(h.get("release_date", ""))
                st.markdown(
                    '<div class="search-row">'
                    '<span class="search-row-title">{}</span>'
                    '<span class="search-row-rating">⭐ {}</span>'
                    '</div>'.format(
                        "{} <span style='font-size:11px;color:#8b93ab;'>({})".format(
                            h.get("title", ""), yr
                        ) + "</span>",
                        h.get("vote_average", "?"),
                    ),
                    unsafe_allow_html=True,
                )
        else:
            st.caption("No matches found.")

    st.markdown("---")
    st.markdown("### 📊 Dataset Stats")
    st.markdown(
        '<span class="stat-chip">🎥 <b>{:,}</b> movies</span>'
        '<span class="stat-chip">🔠 <b>{:,}</b> TF-IDF features</span>'.format(
            rec.movie_count, rec.feature_count
        ),
        unsafe_allow_html=True,
    )
    st.markdown(
        '<span class="stat-chip">⚡ Load time <b>{:.2f}s</b></span>'
        '<span class="stat-chip">🧠 Model <b>NearestNeighbors</b></span>'.format(
            rec.load_time
        ),
        unsafe_allow_html=True,
    )

    st.markdown("---")
    st.caption("Built with ❤️  using Python · scikit-learn · Streamlit")


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN AREA
# ══════════════════════════════════════════════════════════════════════════════

# ── Hero ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero-wrap">
    <div class="hero-badge">🎬 Content-Based · TF-IDF · Cosine Similarity</div>
    <h1 class="hero-title">Discover Your Next<br>Favourite Movie</h1>
    <p class="hero-sub">
      Type any movie title and CineMatch will find the most similar films<br>
      from a curated library of <b>45,000+</b> titles using AI-powered text analysis.
    </p>
</div>
""", unsafe_allow_html=True)

# ── Main search bar ────────────────────────────────────────────────────────────
col_input, col_btn = st.columns([5, 1], gap="small")
with col_input:
    movie_title = st.text_input(
        "movie_title_input",
        placeholder="🎬  Enter a movie title, e.g.  Toy Story, Inception, The Dark Knight…",
        label_visibility="collapsed",
        key="main_search",
    )
with col_btn:
    search_clicked = st.button("Find Movies 🍿", use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
#  TABS
# ══════════════════════════════════════════════════════════════════════════════
tab_recs, tab_search, tab_pipeline, tab_about = st.tabs([
    "🎯 Recommendations", "🔎 Explore Titles", "⚙️ How It Works", "ℹ️ About"
])


# ─────────────────────────────────────────────────────────────────────────────
#  TAB 1 : RECOMMENDATIONS
# ─────────────────────────────────────────────────────────────────────────────
with tab_recs:
    query = movie_title.strip() if movie_title else ""

    if not query:
        # Landing suggestions
        st.markdown("""
        <div class="no-results">
            <div class="no-results-icon">🍿</div>
            <h3>Search for a movie above to get started</h3>
            <p style="color:#8b93ab; margin-top:8px;">
              Try: <b>Toy Story</b> · <b>Inception</b> · <b>The Dark Knight</b> ·
              <b>Avatar</b> · <b>Titanic</b>
            </p>
        </div>
        """, unsafe_allow_html=True)

    else:
        # ── Fetch movie details ───────────────────────────────────────────────
        details = rec.get_movie_details(query)

        if "error" in details:
            # Try partial search fallback
            suggestions = rec.search_movies(query, limit=8)
            st.markdown("""
            <div class="no-results">
                <div class="no-results-icon">🤔</div>
                <h3>"{}" not found in the dataset</h3>
                <p style="color:#8b93ab;">Did you mean one of these?</p>
            </div>
            """.format(query), unsafe_allow_html=True)

            if suggestions:
                for sug in suggestions:
                    yr  = release_year(sug.get("release_date", ""))
                    st.markdown(
                        '<div class="search-row">'
                        '<span class="search-row-title">{} <span style="font-size:11px;color:#8b93ab;">({})</span></span>'
                        '<span class="search-row-genre">{}</span>'
                        '<span class="search-row-rating">⭐ {}</span>'
                        '</div>'.format(
                            sug.get("title", ""),
                            yr,
                            genres_display(sug.get("genres", "")),
                            sug.get("vote_average", "?"),
                        ),
                        unsafe_allow_html=True,
                    )
        else:
            # ── Input movie detail panel ──────────────────────────────────────
            poster = poster_url(details)
            year   = release_year(details.get("release_date", ""))
            rtg    = details.get("vote_average")
            vts    = details.get("vote_count")
            pop    = details.get("popularity")

            if poster:
                poster_html = f'<img src="{poster}" alt="poster"/>'
            else:
                poster_html = '<div style="display:flex; align-items:center; justify-content:center; width:100%; height:100%; background:rgba(255,255,255,0.05); color:#8b93ab; font-weight:600; font-size:16px;">No Poster</div>'

            st.markdown(
                '<div class="detail-panel">'
                '  <div class="detail-poster">{}</div>'
                '  <div class="detail-info">'
                '    <h2>{}</h2>'
                '    <div class="detail-genres">{}</div>'
                '    <div class="detail-stats">'
                '      <span class="stat-pill">⭐ {} / 10</span>'
                '      <span class="stat-pill">🗳️ {} votes</span>'
                '      <span class="stat-pill">📅 {}</span>'
                '      <span class="stat-pill">🔥 Popularity {}</span>'
                '    </div>'
                '    <p class="detail-overview">{}</p>'
                '  </div>'
                '</div>'.format(
                    poster_html,
                    details.get("title", query),
                    genres_display(details.get("genres", "")),
                    rtg if rtg is not None else "N/A",
                    fmt_votes(vts),
                    year,
                    r"{:.1f}".format(float(pop)) if pop else "N/A",
                    str(details.get("overview", "No overview available."))[:380] + "…",
                ),
                unsafe_allow_html=True,
            )

            # ── Get recommendations ───────────────────────────────────────────
            with st.spinner("Finding similar movies…"):
                t0   = time.time()
                recs = rec.recommend(query, n=n_recs)
                qt   = time.time() - t0

            if recs and "error" in recs[0]:
                st.warning(recs[0]["message"])
            elif not recs:
                st.warning("No recommendations found for this title.")
            else:
                # Section header
                st.markdown(
                    '<div class="section-head">'
                    '  <h3>🎯 Because you liked <em>{}</em></h3>'
                    '  <div class="section-line"></div>'
                    '  <span class="section-count">{} results · {:.3f}s</span>'
                    '</div>'.format(query, len(recs), qt),
                    unsafe_allow_html=True,
                )

                # ── Render cards in rows of 5 ─────────────────────────────────
                COLS = 5
                for row_start in range(0, len(recs), COLS):
                    chunk = recs[row_start: row_start + COLS]
                    cols  = st.columns(len(chunk), gap="medium")
                    for col, (rank_offset, r) in zip(cols, enumerate(chunk)):
                        rank    = row_start + rank_offset + 1
                        sim_pct = int(r.get("similarity_score", 0) * 100)
                        p_url   = poster_url(r)
                        rtg     = r.get("vote_average")
                        vts     = r.get("vote_count")
                        imdb    = r.get("imdb_id")
                        yr      = release_year(r.get("release_date", ""))
                        ov      = str(r.get("overview") or "")

                        imdb_href = (
                            '<a href="{}{}" target="_blank" class="imdb-link">IMDb ↗</a>'.format(
                                TMDB_PAGE, imdb
                            ) if imdb else ""
                        )
                        overview_block = (
                            '<p class="card-overview">{}</p>'.format(ov[:160])
                            if show_overview and ov else ""
                        )
                        
                        # EXPLAINABILITY UI 
                        g_str = ", ".join(r.get("matching_genres", []))
                        g_str = g_str if g_str else "None"
                        k_str = ", ".join(r.get("matching_keywords", []))
                        k_str = k_str if k_str else "None"
                        explain_text = r.get("explanation", "")
                        
                        explain_html = '<details style="margin-top:14px; font-size:11.5px; color:#8b93ab;">'
                        explain_html += '<summary style="cursor:pointer; color:#f5c518; font-weight:600; outline:none; user-select:none;">Why this movie?</summary>'
                        explain_html += '<div style="margin-top:8px; line-height:1.5; background:rgba(255,255,255,0.03); padding:10px; border-radius:8px; border:1px solid rgba(255,255,255,0.05);">'
                        
                        if "asmr_contrib_pct" in r:
                            pct = r["asmr_contrib_pct"]
                            explain_html += '<div style="font-size:10.5px; opacity: 0.9; margin-bottom: 6px;">'
                            explain_html += f'✓ <b>Semantic contribution:</b> {pct[0]}%<br/>'
                            explain_html += f'✓ <b>Genre contribution:</b> {pct[1]}%<br/>'
                            explain_html += f'✓ <b>Rating contribution:</b> {pct[2]}%<br/>'
                            explain_html += f'✓ <b>Popularity contribution:</b> {pct[3]}%<br/>'
                            explain_html += f'✓ <b>Franchise contribution:</b> {pct[4]}%<br/>'
                            explain_html += '</div>'
                        elif "semantic_score" in r:
                            explain_html += '<div style="margin-bottom:4px;">✓ <b>Semantic understanding:</b> Confirmed Match</div>'
                            if g_str != "None":
                                explain_html += f'<div style="margin-bottom:4px;">✓ <b>Similar genres:</b> {g_str}</div>'
                            if r.get("franchise_score", 0) > 0:
                                explain_html += '<div style="margin-bottom:4px;">✓ <b>Related franchise / sequel</b></div>'
                        else:
                            explain_html += f'<div style="margin-bottom:4px;">✓ <b>Similar genres:</b> {g_str}</div>'
                            explain_html += f'<div style="margin-bottom:4px;">✓ <b>Similar themes:</b> {k_str}</div>'
                            
                        explain_html += f'<div style="margin-bottom:8px;">✓ <b>Content match:</b> {sim_pct}%</div>'
                        explain_html += f'<i style="color:#a6adc2;">{explain_text}</i>'
                        
                        if debug_mode and "semantic_score" in r:
                            explain_html += '<hr style="border: 0; border-top: 1px solid rgba(255,255,255,0.1); margin: 8px 0;" />'
                            if hasattr(rec, 'asmr_weights'):
                                w = rec.asmr_weights
                                explain_html += f'<div style="margin-bottom:4px; font-weight:bold; font-size:10.5px; color:#f5c518;">ASMR Target Weights: [{w[0]:.2f}, {w[1]:.2f}, {w[2]:.2f}, {w[3]:.2f}, {w[4]:.2f}]</div>'
                            explain_html += f'<div style="font-family:monospace; font-size:10px;">Sem: {r["semantic_score"]:.3f}<br/>'
                            explain_html += f'Gen: {r["genre_score"]:.3f}<br/>'
                            explain_html += f'Rat: {r["rating_score"]:.3f}<br/>'
                            explain_html += f'Pop: {r["popularity_score"]:.3f}<br/>'
                            explain_html += f'Fra: {r.get("franchise_score",0):.3f}<br/>'
                            explain_html += f'<b style="color:#f5c518">Final Score: {r["final_score"]:.3f}</b></div>'
                            
                        explain_html += '</div></details>'

                        if p_url:
                            card_poster_html = f'<img src="{p_url}" alt="{r.get("title", "")}" loading="lazy"/>'
                        else:
                            card_poster_html = '<div style="display:flex; align-items:center; justify-content:center; width:100%; height:100%; background:rgba(255,255,255,0.05); color:#8b93ab; font-weight:600;">No Poster</div>'

                        with col:
                            st.markdown(
                                '<div class="movie-card">'
                                '  <div class="poster-wrap">'
                                '    {}'
                                '    <div class="poster-overlay"></div>'
                                '    <div class="rank-badge">#{}</div>'
                                '    <div class="sim-pill">{}% match</div>'
                                '  </div>'
                                '  <div class="card-body">'
                                '    <p class="card-title">{}</p>'
                                '    <p class="card-genres">{}</p>'
                                '    <div class="card-meta">'
                                '      <span class="meta-chip rating">⭐ {}</span>'
                                '      <span class="meta-chip votes">🗳 {}</span>'
                                '      <span class="meta-chip votes">📅 {}</span>'
                                '    </div>'
                                '    {}'
                                '    {}'
                                '    {}'
                                '  </div>'
                                '</div>'.format(
                                    card_poster_html,
                                    rank,
                                    sim_pct,
                                    r.get("title", "Unknown"),
                                    genres_display(r.get("genres", "")),
                                    "{:.1f}".format(float(rtg)) if rtg else "N/A",
                                    fmt_votes(vts),
                                    yr,
                                    overview_block,
                                    imdb_href,
                                    explain_html
                                ),
                                unsafe_allow_html=True,
                            )


# ─────────────────────────────────────────────────────────────────────────────
#  TAB 2 : EXPLORE / SEARCH
# ─────────────────────────────────────────────────────────────────────────────
with tab_search:
    st.markdown("### 🔭 Explore the Movie Library")
    st.markdown(
        "Search for any partial title to browse the full dataset of **{:,}** movies.".format(
            rec.movie_count
        )
    )

    explore_q    = st.text_input(
        "Explore query", placeholder="Type part of a title…",
        label_visibility="collapsed", key="explore_q"
    )
    explore_limit = st.slider("Results limit", 5, 50, 20, key="explore_limit")

    if explore_q.strip():
        t0      = time.time()
        e_hits  = rec.search_movies(explore_q.strip(), limit=explore_limit)
        e_time  = time.time() - t0

        if e_hits:
            st.markdown(
                '<div class="info-box">Found {} results for <b>"{}"</b> in {:.3f}s</div>'.format(
                    len(e_hits), explore_q, e_time
                ),
                unsafe_allow_html=True,
            )
            for h in e_hits:
                yr = release_year(h.get("release_date", ""))
                st.markdown(
                    '<div class="search-row">'
                    '  <div style="flex:1">'
                    '    <span class="search-row-title">{}</span>'
                    '    <span style="font-size:11px;color:#8b93ab;"> ({})</span><br/>'
                    '    <span class="search-row-genre">{}</span>'
                    '  </div>'
                    '  <span class="search-row-rating">⭐ {}</span>'
                    '</div>'.format(
                        h.get("title", ""),
                        yr,
                        genres_display(h.get("genres", "")),
                        h.get("vote_average", "?"),
                    ),
                    unsafe_allow_html=True,
                )
        else:
            st.markdown(
                '<div class="no-results"><div class="no-results-icon">🕵️</div>'
                '<h3>No titles match "{}"</h3></div>'.format(explore_q),
                unsafe_allow_html=True,
            )
    else:
        st.markdown(
            '<div class="no-results"><div class="no-results-icon">🔍</div>'
            '<h3>Enter a search term above</h3></div>',
            unsafe_allow_html=True,
        )


# ─────────────────────────────────────────────────────────────────────────────
#  TAB 3 : PIPELINE
# ─────────────────────────────────────────────────────────────────────────────
with tab_pipeline:
    st.markdown("### ⚙️  How CineMatch Works")

    st.markdown("""
    <div class="pipeline">
      <div class="pipe-node">
        <div class="pipe-node-icon">📄</div>
        <p class="pipe-node-label">movies.csv</p>
        <p class="pipe-node-sub">45,466 raw rows</p>
      </div>
      <div class="pipe-arrow">→</div>
      <div class="pipe-node">
        <div class="pipe-node-icon">🧹</div>
        <p class="pipe-node-label">Preprocessing</p>
        <p class="pipe-node-sub">Clean · Deduplicate</p>
      </div>
      <div class="pipe-arrow">→</div>
      <div class="pipe-node">
        <div class="pipe-node-icon">🏷️</div>
        <p class="pipe-node-label">Feature Tags</p>
        <p class="pipe-node-sub">genres + overview</p>
      </div>
      <div class="pipe-arrow">→</div>
      <div class="pipe-node">
        <div class="pipe-node-icon">📐</div>
        <p class="pipe-node-label">TF-IDF</p>
        <p class="pipe-node-sub">20,000 features</p>
      </div>
      <div class="pipe-arrow">→</div>
      <div class="pipe-node">
        <div class="pipe-node-icon">🤝</div>
        <p class="pipe-node-label">NearestNeighbors</p>
        <p class="pipe-node-sub">Cosine · Brute</p>
      </div>
      <div class="pipe-arrow">→</div>
      <div class="pipe-node">
        <div class="pipe-node-icon">🎯</div>
        <p class="pipe-node-label">Top-N Results</p>
        <p class="pipe-node-sub">Ranked by similarity</p>
      </div>
    </div>
    """, unsafe_allow_html=True)

    c1, c2 = st.columns(2, gap="large")

    with c1:
        st.markdown("#### 🧹 Data Preprocessing")
        st.markdown("""
- Loaded **45,466** raw rows from `movies.csv`
- Removed **17** exact duplicates
- Resolved **13** duplicate TMDB IDs (kept richest record)
- Dropped **259** rows with no genres AND no overview
- Final dataset: **45,171** clean movies

#### 📐 TF-IDF Vectorisation
- Combined `genres + overview` into a unified `tags` column
- Applied **TF-IDF** with bigrams (`ngram_range=(1,2)`)
- Vocabulary: **20,000 features** (`max_features=20000`)
- Result: sparse CSR matrix of shape **45,171 × 20,000**
- Matrix density: **0.15%** → only **15.6 MB** in memory
        """)

    with c2:
        st.markdown("#### 🤝 Similarity Search")
        st.markdown("""
- Used **sklearn NearestNeighbors** (cosine metric, brute-force)
- Works directly on the sparse CSR matrix
- No N×N dense similarity matrix ever computed
- A full dense matrix would require **~15 GB** RAM
- Actual memory usage: **~35 MB** (sparse + metadata)

#### ⚡ Query Time
- Model loads once and is cached in memory
- Each recommendation query: **< 50 ms** for 45k movies
- Returns `similarity = 1 − cosine_distance`
- Similarity 1.0 = identical · 0.0 = completely different

#### 🎯 Result Ranking
- Self-match removed automatically
- Duplicate titles resolved by `vote_count × vote_average`
- Results sorted by descending cosine similarity
        """)

    st.markdown("---")
    st.markdown("#### 📁 Project File Structure")
    st.code("""
Movie-Recommendation-System/
├── movies.csv.csv              ← original dataset (untouched)
├── app.py                      ← Streamlit application (this file)
├── dataset/
│   └── movies_cleaned.csv      ← 45,171 cleaned movies
├── models/
│   ├── tfidf_vectorizer.pkl    ← fitted TF-IDF vectorizer (750 KB)
│   ├── movie_neighbors.pkl     ← fitted NearestNeighbors (15.6 MB)
│   └── movie_metadata.pkl      ← movie metadata DataFrame (18.7 MB)
└── src/
    ├── analyze_dataset.py      ← Step 1: dataset analysis
    ├── preprocess_data.py      ← Step 2: cleaning pipeline
    ├── build_model.py          ← Step 3: TF-IDF + NN model
    ├── recommendation_engine.py← Step 4: MovieRecommender class
    └── test_recommender.py     ← Step 4: CLI tests
    """, language="")


# ─────────────────────────────────────────────────────────────────────────────
#  TAB 4 : ABOUT
# ─────────────────────────────────────────────────────────────────────────────
with tab_about:
    st.markdown("### ℹ️  About CineMatch")

    a1, a2 = st.columns([3, 2], gap="large")

    with a1:
        st.markdown("""
**CineMatch** is a professional content-based movie recommendation system
built as a structured internship project.

#### 🎓 What is Content-Based Filtering?
Content-based filtering recommends items similar to what a user already likes,
based on the *content* of the items themselves — not on what other users chose.

For movies, we analyse:
- **Genres** — Action, Comedy, Drama…
- **Overview** — the plot synopsis text

These are combined into a single `tags` document per movie, then transformed
into numerical vectors using **TF-IDF** (Term Frequency–Inverse Document Frequency).

#### 🔢 TF-IDF Explained
TF-IDF assigns higher weights to words that:
- Appear **often** in a specific movie's description (high TF)
- Appear **rarely** across all movies (high IDF)

This means genre-specific and plot-specific vocabulary gets emphasised,
while generic stop words ("the", "a", "is") are automatically removed.

#### 📐 Cosine Similarity
Two movie vectors are compared using **cosine similarity** —
the cosine of the angle between them in 20,000-dimensional space.
- Score = **1.0** → identical direction = very similar movies
- Score = **0.0** → perpendicular = completely different topics

#### ⚡ Why NearestNeighbors?
Instead of a slow O(N²) similarity computation, we use sklearn's
`NearestNeighbors` which at query time only computes distances
from the *single query vector* to all 45k movie vectors.
No full matrix is ever stored.
        """)

    with a2:
        st.markdown("#### 📊 Model Specifications")
        specs = {
            "Dataset"         : "TMDB 45k Movies",
            "Total movies"    : "{:,}".format(rec.movie_count),
            "TF-IDF features" : "{:,}".format(rec.feature_count),
            "Bigrams"         : "Yes (1,2)-grams",
            "Stop words"      : "English removed",
            "Min document freq": "2 movies",
            "Sublinear TF"    : "Yes (log scaling)",
            "Similarity"      : "Cosine",
            "Algorithm"       : "Brute-force",
            "Matrix format"   : "Sparse CSR",
            "Matrix memory"   : "~15.6 MB",
            "Query time"      : "< 50 ms",
        }
        for k, v in specs.items():
            st.markdown(
                '<div style="display:flex;justify-content:space-between;'
                'padding:7px 12px;border-bottom:1px solid rgba(255,255,255,0.06);">'
                '<span style="color:#8b93ab;font-size:13px;">{}</span>'
                '<span style="font-weight:600;font-size:13px;">{}</span>'
                '</div>'.format(k, v),
                unsafe_allow_html=True,
            )

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("#### 🛠️ Technology Stack")
        techs = ["Python 3.12", "pandas", "scikit-learn", "joblib", "Streamlit", "TMDB Dataset", "TMDB API"]
        for t in techs:
            st.markdown(
                '<span style="display:inline-block;background:rgba(245,197,24,0.1);'
                'border:1px solid rgba(245,197,24,0.3);border-radius:50px;'
                'padding:4px 14px;font-size:12px;font-weight:600;'
                'color:#f5c518;margin:3px 4px 3px 0;">{}</span>'.format(t),
                unsafe_allow_html=True,
            )

        st.markdown("<br><br>", unsafe_allow_html=True)
        st.markdown(
            '<div style="font-size:12px; color:#8b93ab;">'
            'This product uses the TMDB API but is not endorsed or certified by TMDB.<br>'
            'Movie metadata and posters are provided by TMDB.'
            '</div>',
            unsafe_allow_html=True,
        )
