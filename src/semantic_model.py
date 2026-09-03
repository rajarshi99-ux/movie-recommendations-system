# -*- coding: utf-8 -*-
import os
import time
import numpy as np
import pandas as pd
# pyrefly: ignore [missing-import]
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJECT = os.path.dirname(_HERE)
_MODELS_DIR = os.path.join(_PROJECT, "models")
EMBEDDING_CACHE = os.path.join(_MODELS_DIR, "semantic_embeddings.npy")
MODEL_NAME = "all-MiniLM-L6-v2"

class SemanticMovieModel:
    def __init__(self, metadata_df):
        self.metadata = metadata_df.reset_index(drop=True)
        self.embeddings = None
        self.embedding_generation_time_ms = 0
        self.embedding_cache_load_time_ms = 0
        self.model = None

    def _load_model(self):
        if self.model is None:
            self.model = SentenceTransformer(MODEL_NAME)

    def load_or_generate_embeddings(self):
        if os.path.exists(EMBEDDING_CACHE):
            t0 = time.time()
            self.embeddings = np.load(EMBEDDING_CACHE)
            self.embedding_cache_load_time_ms = (time.time() - t0) * 1000
        else:
            self.generate_and_save_embeddings()
            
    def get_text_for_row(self, row):
        overview = str(row.get('overview', ''))
        genres = str(row.get('genres', ''))
        tags = str(row.get('tags', ''))
        if overview == 'nan': overview = ''
        if genres == 'nan': genres = ''
        if tags == 'nan': tags = ''
        return f"{genres}. {tags}. {overview}".strip()

    def generate_and_save_embeddings(self):
        self._load_model()
        t0 = time.time()
        print(f"Generating embeddings for {len(self.metadata)} movies...")
        
        texts = [self.get_text_for_row(r) for _, r in self.metadata.iterrows()]
        
        # We use a dynamic batch size and force normalization for fast cosine similarity via dot product / matrix mult
        self.embeddings = self.model.encode(texts, batch_size=128, show_progress_bar=True, normalize_embeddings=True)
        self.embedding_generation_time_ms = (time.time() - t0) * 1000
        np.save(EMBEDDING_CACHE, self.embeddings)

    def get_similarities(self, row_idx):
        """Returns distance for all movies from row_idx."""
        if self.embeddings is None:
            self.load_or_generate_embeddings()
        
        q_emb = self.embeddings[row_idx].reshape(1, -1)
        sims = np.dot(self.embeddings, q_emb.T).flatten()
        return sims
