# -*- coding: utf-8 -*-
"""
=============================================================
  Movie Recommendation System  -  Step 4: Recommendation Engine
=============================================================
Provides the MovieRecommender class which loads the pre-built
TF-IDF vectorizer, NearestNeighbors model, and movie metadata
from the models/ directory, then serves content-based movie
recommendations without ever constructing a dense similarity
matrix.

Design decisions:
  - artefacts loaded once at construction time
  - all queries operate on the pre-fitted sparse NearestNeighbors
    model; no new vectorisation or similarity computation occurs
  - cosine similarity = 1 - cosine_distance  (sklearn convention)
  - duplicate titles resolved by choosing the record with the
    highest (vote_count * vote_average) composite score so that
    well-known versions of a shared title are preferred
  - unknown titles return an explicit dict with 'error' key
    instead of raising an exception, keeping callers safe
"""

import os
import re
import time
import joblib
import numpy as np
import pandas as pd

# ------------------------------------------------------------------ #
#  Default paths (relative to this file's parent directory)            #
# ------------------------------------------------------------------ #
_HERE        = os.path.dirname(os.path.abspath(__file__))
_PROJECT     = os.path.dirname(_HERE)
_MODELS_DIR  = os.path.join(_PROJECT, "models")

DEFAULT_TFIDF_PATH = os.path.join(_MODELS_DIR, "tfidf_vectorizer.pkl")
DEFAULT_NN_PATH    = os.path.join(_MODELS_DIR, "movie_neighbors.pkl")
DEFAULT_META_PATH  = os.path.join(_MODELS_DIR, "movie_metadata.pkl")


class MovieRecommender:
    """
    Content-based movie recommendation engine.

    Parameters
    ----------
    tfidf_path : str
        Path to the joblib-serialised TfidfVectorizer.
    nn_path : str
        Path to the joblib-serialised NearestNeighbors model.
    meta_path : str
        Path to the joblib-serialised movie metadata DataFrame.

    Usage
    -----
    rec = MovieRecommender()
    results = rec.recommend("Toy Story", n=10)
    for r in results:
        print(r["title"], r["similarity_score"])
    """

    # Columns returned in each recommendation dict
    _RETURN_COLS = [
        "title", "genres", "overview", "similarity_score",
        "vote_average", "vote_count", "popularity",
        "release_date", "poster_path", "imdb_id",
    ]

    def __init__(
        self,
        tfidf_path: str = DEFAULT_TFIDF_PATH,
        nn_path:    str = DEFAULT_NN_PATH,
        meta_path:  str = DEFAULT_META_PATH,
    ):
        t0 = time.time()
        self._check_file(tfidf_path, "tfidf_vectorizer.pkl")
        self._check_file(nn_path,    "movie_neighbors.pkl")
        self._check_file(meta_path,  "movie_metadata.pkl")

        self.vectorizer  = joblib.load(tfidf_path)
        self.nn_model    = joblib.load(nn_path)
        self.metadata    = joblib.load(meta_path)   # pandas DataFrame
        self.load_time   = time.time() - t0

        # Ensure the metadata index is contiguous (matches the NN model rows)
        self.metadata = self.metadata.reset_index(drop=True)

        # Build a lower-cased title -> [row indices] lookup for fast search
        self._title_index = self._build_title_index()

        # Precompute maximum popularity for normalization [0, 1]
        self._max_pop = float(pd.to_numeric(self.metadata['popularity'], errors='coerce').fillna(0).max())
        if self._max_pop <= 0:
            self._max_pop = 1.0

    # ---------------------------------------------------------------- #
    #  Private helpers                                                   #
    # ---------------------------------------------------------------- #

    @staticmethod
    def _check_file(path: str, label: str) -> None:
        if not os.path.exists(path):
            raise FileNotFoundError(
                "Model artefact '{}' not found at: {}\n"
                "Run src/build_model.py first.".format(label, path)
            )

    def _build_title_index(self) -> dict:
        """
        Returns a dict mapping lower-cased stripped title ->
        list of integer row indices in self.metadata.
        Built once at construction; O(1) lookups afterwards.
        """
        index = {}
        for idx, title in enumerate(self.metadata["title"]):
            key = str(title).lower().strip()
            index.setdefault(key, []).append(idx)
        return index

    def _resolve_title(self, movie_title: str):
        """
        Find the best row index for a given title string.

        Duplicate title handling:
          If several rows share the same normalised title, select
          the one with the highest (vote_count * vote_average)
          composite score.  This ensures well-known, widely-rated
          versions of a title are chosen over obscure entries with
          the same name (e.g. remakes, TV films).

        Returns
        -------
        (row_index: int, chosen_row: pd.Series)
        or raises KeyError if the title is not found.
        """
        key = str(movie_title).lower().strip()
        if key not in self._title_index:
            raise KeyError(movie_title)

        candidates = self._title_index[key]
        if len(candidates) == 1:
            idx = candidates[0]
            return idx, self.metadata.iloc[idx]

        # Multiple matches - pick by composite popularity score
        best_idx = max(
            candidates,
            key=lambda i: (
                float(self.metadata.at[i, "vote_count"] or 0)
                * float(self.metadata.at[i, "vote_average"] or 0)
            ),
        )
        return best_idx, self.metadata.iloc[best_idx]

    @staticmethod
    def _row_to_dict(row: pd.Series, similarity: float) -> dict:
        """Convert a metadata row + similarity score to a result dict."""
        result = {"similarity_score": round(float(similarity), 4)}
        for col in [
            "title", "genres", "overview", "vote_average",
            "vote_count", "popularity", "release_date",
            "poster_path", "imdb_id",
        ]:
            val = row.get(col, None)
            # Normalise pandas NA to Python None for clean JSON serialisation
            result[col] = None if pd.isna(val) else val
        return result

    # ---------------------------------------------------------------- #
    #  Public API                                                        #
    # ---------------------------------------------------------------- #

    def recommend(self, movie_title: str, n: int = 10) -> list:
        """
        Return the top-n content-based recommendations for a movie.

        Parameters
        ----------
        movie_title : str
            Title of the source movie (case-insensitive, strips whitespace).
        n : int
            Number of recommendations to return (1-50).

        Returns
        -------
        list of dicts, each containing:
            title, genres, overview, similarity_score, vote_average,
            vote_count, popularity, release_date, poster_path, imdb_id.

        Special return values
        ---------------------
        - If the title is not found, returns a list containing one dict
          with key 'error' and key 'message'.
        - If n is out of range, raises ValueError.
        """
        # Validate n
        if not isinstance(n, int) or n < 1 or n > 50:
            raise ValueError(
                "n must be an integer between 1 and 50, got: {}".format(n)
            )

        # Resolve title -> row index
        try:
            source_idx, source_row = self._resolve_title(movie_title)
        except KeyError:
            return [{
                "error":   "Movie not found",
                "message": "No movie titled '{}' exists in the dataset. "
                           "Try search_movies() to find similar titles.".format(movie_title),
                "query":   movie_title,
            }]

        t0 = time.time()

        # Retrieve a larger candidate pool (e.g., 500) for re-ranking
        # This ensures high-rated/popular movies can bubble up even if 
        # their baseline TF-IDF similarity is slightly lower.
        pool_size = min(500, len(self.metadata))
        # The NN model stores the sparse matrix internally; we pass a row index
        # to kneighbors via the stored matrix reference.
        # We reconstruct the query vector from the model's internal training data.
        query_vector = self.nn_model._fit_X[source_idx]   # sparse row

        distances, indices = self.nn_model.kneighbors(
            query_vector, n_neighbors=pool_size
        )
        distances = distances.flatten()
        indices   = indices.flatten()

        # Build candidate list
        candidates = []
        input_key = str(source_row["title"]).lower().strip()
        
        # Pre-fetch vocab for explainability
        feature_names = self.vectorizer.get_feature_names_out()
        source_genres = set(str(source_row.get("genres", "")).split())
        query_indices = set(query_vector.indices)

        for dist, idx in zip(distances, indices):
            row = self.metadata.iloc[idx]
            row_key = str(row["title"]).lower().strip()

            # Skip the input movie itself and exact duplicate titles
            if idx == source_idx or row_key == input_key:
                continue

            # 1. Content similarity (cosine similarity) [0, 1]
            content_sim = 1.0 - float(dist)

            # 2. Normalized Rating [0, 1] (vote_average is out of 10)
            try:
                rating = float(row.get("vote_average", 0))
                if pd.isna(rating): rating = 0.0
            except (ValueError, TypeError):
                rating = 0.0
            norm_rating = rating / 10.0

            # 3. Normalized Popularity [0, 1] (scaled by global max popularity)
            try:
                pop = float(row.get("popularity", 0))
                if pd.isna(pop): pop = 0.0
            except (ValueError, TypeError):
                pop = 0.0
            norm_pop = pop / self._max_pop

            # Hybrid Score calculation
            # 70% content + 20% rating + 10% popularity
            final_score = (0.70 * content_sim) + (0.20 * norm_rating) + (0.10 * norm_pop)

            # EXPLAINABILITY MODULE
            rec_genres = set(str(row.get("genres", "")).split())
            matching_genres = list(source_genres & rec_genres)

            rec_vector = self.nn_model._fit_X[idx]
            overlap_indices = list(query_indices & set(rec_vector.indices))
            
            # Map TF-IDF feature indices -> scores in the recommended vector
            score_map = {c: v for c, v in zip(rec_vector.indices, rec_vector.data)}
            overlap_indices.sort(key=lambda x: score_map.get(x, 0), reverse=True)

            top_keywords = []
            lower_genres = {g.lower() for g in (source_genres | rec_genres)}
            for i in overlap_indices:
                word = feature_names[i]
                if word not in lower_genres:
                    top_keywords.append(word)
                    if len(top_keywords) >= 4:
                        break

            # Natural language explanation
            explanation = "Recommended because it shares {} similar genres and themes like '{}' with {}.".format(
                len(matching_genres),
                ", ".join(top_keywords[:3]) if top_keywords else "overlapping tags",
                source_row.get("title", "")
            )

            result_dict = self._row_to_dict(row, content_sim)
            result_dict["final_score"] = final_score
            result_dict["explanation"] = explanation
            result_dict["matching_genres"] = matching_genres
            result_dict["matching_keywords"] = top_keywords
            
            candidates.append(result_dict)

        # Sort all candidates by the new hybrid final score descending
        candidates.sort(key=lambda x: x["final_score"], reverse=True)
        
        # Take the top 'n' results
        results = candidates[:n]

        self._last_query_time = time.time() - t0
        return results

    def search_movies(self, query: str, limit: int = 10) -> list:
        """
        Case-insensitive partial title search.

        Parameters
        ----------
        query : str
            Substring to search for in movie titles.
        limit : int
            Maximum number of matching titles to return.

        Returns
        -------
        list of dicts with keys: title, genres, vote_average, release_date.
        """
        if not isinstance(limit, int) or limit < 1:
            raise ValueError("limit must be a positive integer.")

        pattern = re.escape(str(query).strip().lower())
        results = []
        for idx, row in self.metadata.iterrows():
            t = str(row.get("title", "")).lower()
            if re.search(pattern, t):
                results.append({
                    "title":        row.get("title"),
                    "genres":       row.get("genres"),
                    "vote_average": row.get("vote_average"),
                    "release_date": str(row.get("release_date", ""))[:10],
                })
                if len(results) >= limit:
                    break
        return results

    def get_movie_details(self, movie_title: str) -> dict:
        """
        Return full metadata for a movie title.

        Parameters
        ----------
        movie_title : str
            Exact title (case-insensitive match).

        Returns
        -------
        dict of all metadata columns, or a dict with 'error' key if not found.
        """
        try:
            _, row = self._resolve_title(movie_title)
        except KeyError:
            return {
                "error":   "Movie not found",
                "message": "No movie titled '{}' in the dataset.".format(movie_title),
            }

        return row.to_dict()

    # ---------------------------------------------------------------- #
    #  Introspection helpers                                             #
    # ---------------------------------------------------------------- #

    @property
    def movie_count(self) -> int:
        """Total number of movies in the dataset."""
        return len(self.metadata)

    @property
    def feature_count(self) -> int:
        """Number of TF-IDF features in the vocabulary."""
        return len(self.vectorizer.vocabulary_)

    def __repr__(self) -> str:
        return (
            "MovieRecommender(movies={:,}, features={:,}, "
            "load_time={:.2f}s)".format(
                self.movie_count, self.feature_count, self.load_time
            )
        )
