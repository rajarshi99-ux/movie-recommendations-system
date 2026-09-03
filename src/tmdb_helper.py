# -*- coding: utf-8 -*-
"""
tmdb_helper.py
==============
Lightweight TMDB API utility for the CineMatch app.

Responsibilities:
  - Load TMDB_API_KEY from .env (never from source code)
  - Provide get_poster_url(title) with st.cache_data caching
  - Strip parenthesised year suffixes from titles before querying
  - Return a safe fallback URL when poster unavailable or API down
  - Never print / expose the API key

Usage:
    from tmdb_helper import get_poster_url, TMDB_API_AVAILABLE
    url = get_poster_url("Inception")
"""

import os
import re
import requests
# pyrefly: ignore [missing-import]
import streamlit as st

from dotenv import load_dotenv

# ── Load .env from the project root (parent of this file's src/ directory) ──
_HERE        = os.path.dirname(os.path.abspath(__file__))
_PROJECT     = os.path.dirname(_HERE)
_ENV_PATH    = os.path.join(_PROJECT, ".env")
load_dotenv(dotenv_path=_ENV_PATH, override=False)

from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# ── API config ───────────────────────────────────────────────────────────────
_TMDB_API_KEY    = os.getenv("TMDB_API_KEY", "")
_TMDB_SEARCH_URL = "https://api.themoviedb.org/3/search/movie"
_TMDB_IMG_BASE   = "https://image.tmdb.org/t/p/w500"
_TIMEOUT         = 6   # seconds per network call

FALLBACK_POSTER = (
    "https://placehold.co/342x513/1a1f35/f5c518?text=No+Poster"
)

# Public flag so app.py can show/hide API-related UI
TMDB_API_AVAILABLE = bool(_TMDB_API_KEY)

_session = requests.Session()
_retries = Retry(total=3, backoff_factor=1, status_forcelist=[ 429, 500, 502, 503, 504 ])
_session.mount("https://", HTTPAdapter(max_retries=_retries))

# ── Year-suffix stripper ─────────────────────────────────────────────────────
_YEAR_RE = re.compile(r"\s*\(\d{4}\)\s*$")

def _clean_title_for_query(title: str) -> str:
    """
    Remove a trailing year like '(1995)' from a movie title before
    sending it to the TMDB search API.

    'Toy Story (1995)'  ->  'Toy Story'
    'Inception'         ->  'Inception'
    """
    return _YEAR_RE.sub("", str(title)).strip()


@st.cache_data(ttl=86400, show_spinner=False)
def _verify_image_url(url: str) -> bool:
    """Perform a fast HEAD request to ensure the TMDB asset hasn't been purged."""
    try:
        r = _session.head(url, timeout=2.0)
        return r.status_code == 200
    except Exception:
        return False

@st.cache_data(ttl=86400, show_spinner=False)
def _fetch_tmdb_poster_by_id(tmdb_id) -> str:
    """Fetch fresh poster metadata directly from TMDB by ID as a fallback."""
    if not TMDB_API_AVAILABLE or not tmdb_id:
        return None
        
    try:
        url = f"https://api.themoviedb.org/3/movie/{tmdb_id}"
        resp = _session.get(url, params={"api_key": _TMDB_API_KEY}, timeout=_TIMEOUT)
        if resp.status_code == 200:
            data = resp.json()
            new_path = data.get("poster_path")
            if new_path:
                if new_path.startswith("/"):
                    return _TMDB_IMG_BASE + new_path
                return _TMDB_IMG_BASE + "/" + new_path
    except Exception:
        pass
        
    return None

def get_poster_url(movie: dict):
    """
    Return a w500 TMDB poster URL for the given movie dictionary.

    Strategy
    --------
    1. Check if the movie dict has a valid `poster_path`.
    2. Construct the TMDB image URL and Verify it actually loads (not 404).
    3. If the local path is stale/missing, dynamically fallback to TMDB API using 'id'.
    4. If both fail, return None. The UI will render the "No Poster" HTML element.
    """
    if not isinstance(movie, dict):
        return None

    path = movie.get("poster_path")
    if path and isinstance(path, str) and str(path).lower() not in ("none", "nan", ""):
        url = ""
        if path.startswith("/"):
            url = _TMDB_IMG_BASE + path
        elif path.startswith("http"):
            url = path
        else:
            url = _TMDB_IMG_BASE + "/" + path
            
        if _verify_image_url(url):
            return url
            
    # Fallback to TMDB API dynamically if local metadata is stale or missing
    tmdb_id = movie.get("id")
    if tmdb_id:
        new_url = _fetch_tmdb_poster_by_id(tmdb_id)
        if new_url:
            return new_url

    return None
