# -*- coding: utf-8 -*-
import os
import json
import numpy as np
import pandas as pd
import time
from recommendation_engine import MovieRecommender
from semantic_model import SemanticMovieModel

_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJECT = os.path.dirname(_HERE)
WEIGHTS_PATH = os.path.join(_PROJECT, "optimized_weights.json")

class AdaptiveRecommender(MovieRecommender):
    def __init__(self):
        # Initializes underlying models (TFIDF baseline, etc.)
        super().__init__()
        
        self.model_mode = "asmr"
        
        # Load Semantic Model
        self.semantic_model = SemanticMovieModel(self.metadata)
        self.semantic_model.load_or_generate_embeddings()
        
        # Load ASMR Optimized Weights
        self.asmr_weights = [0.60, 0.15, 0.10, 0.05, 0.10] # Default fallback Model F
        if os.path.exists(WEIGHTS_PATH):
            try:
                with open(WEIGHTS_PATH, "r") as f:
                    cfg = json.load(f)
                    self.asmr_weights = cfg.get("optimized_weights", self.asmr_weights)
            except Exception as e:
                print(f"Warning: Failed to load ASMR weights. {e}")
                
    def recommend(self, movie_title: str, n: int = 10) -> list:
        if not isinstance(n, int) or n < 1 or n > 50:
            raise ValueError(f"n must be an integer between 1 and 50, got: {n}")

        try:
            source_idx, source_row = self._resolve_title(movie_title)
        except KeyError:
            return [{
                "error":   "Movie not found",
                "message": f"No movie titled '{movie_title}' exists in the dataset. Try search_movies().",
                "query":   movie_title,
            }]

        t0 = time.time()
        pool_size = min(500, len(self.metadata))
        
        # O(1) Offline Semantic Matrix Dot Product
        all_sims = self.semantic_model.get_similarities(source_idx)
        indices = np.argsort(all_sims)[::-1][:pool_size]

        candidates = []
        input_key = str(source_row["title"]).lower().strip()
        
        source_genres = set(str(source_row.get("genres", "")).split())
        q_t_clean = str(source_row["title"]).lower().strip().replace(":", "").replace("-", "")

        for idx in indices:
            row = self.metadata.iloc[idx]
            row_key = str(row["title"]).lower().strip()

            if idx == source_idx or row_key == input_key:
                continue

            content_sim = float(all_sims[idx])
            
            c_g_set = set(str(row.get("genres", "")).split())
            u_g = source_genres | c_g_set
            g_sim = len(source_genres & c_g_set) / len(u_g) if u_g else 0.0
            
            try:
                rating = float(row.get("vote_average", 0))
                if pd.isna(rating): rating = 0.0
            except:
                rating = 0.0
            norm_rating = rating / 10.0
            
            try:
                pop = float(row.get("popularity", 0))
                if pd.isna(pop): pop = 0.0
            except:
                pop = 0.0
            norm_pop = pop / self._max_pop
            
            c_t_clean = str(row["title"]).lower().strip().replace(":", "").replace("-", "")
            f_sig = 1.0 if (len(q_t_clean) > 3 and len(c_t_clean) > 3 and (q_t_clean in c_t_clean or c_t_clean in q_t_clean)) else 0.0
            
            # ASMR Score calculation
            w = self.asmr_weights
            raw_components = [
                w[0] * content_sim,
                w[1] * g_sim,
                w[2] * norm_rating,
                w[3] * norm_pop,
                w[4] * f_sig
            ]
            final_score = sum(raw_components)
            
            # Contribution Percentages for Explainability
            contrib_pct = [0, 0, 0, 0, 0]
            if final_score > 0:
                contrib_pct = [int((c / final_score) * 100) for c in raw_components]
                
            # Adjust rounding error explicitly on the largest component
            if sum(contrib_pct) > 0 and sum(contrib_pct) != 100:
                max_idx = np.argmax(contrib_pct)
                contrib_pct[max_idx] += (100 - sum(contrib_pct))

            matching_genres = list(source_genres & c_g_set)
            explanation = "Recommended intelligently by Adaptive Semantic Metaheuristics."
            if len(matching_genres) > 0:
                explanation += f" Shared genres: {', '.join(matching_genres[:3])}."

            result_dict = self._row_to_dict(row, 1.0 - content_sim)
            result_dict["final_score"] = final_score
            result_dict["explanation"] = explanation
            result_dict["matching_genres"] = matching_genres
            result_dict["semantic_score"] = content_sim
            result_dict["genre_score"] = g_sim
            result_dict["rating_score"] = norm_rating
            result_dict["popularity_score"] = norm_pop
            result_dict["franchise_score"] = f_sig
            result_dict["asmr_contrib_pct"] = contrib_pct
            
            candidates.append(result_dict)

        candidates.sort(key=lambda x: x["final_score"], reverse=True)
        results = candidates[:n]

        self._last_query_time = time.time() - t0
        return results
