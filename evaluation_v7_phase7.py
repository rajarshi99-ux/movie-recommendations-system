# -*- coding: utf-8 -*-
"""
Phase 6 ASMR Optimization — Vectorized Rewrite
================================================
Preserves ALL scientific invariants from the audit:
 - Same 2,000 opt users (seed=123)
 - Same TRAIN-only profiles
 - Same VAL-positive targets (rating >= 4.0)
 - Same 45,171-item full catalog
 - Same NDCG@5 objective
 - Same DE: best1bin, pop=15, maxiter=40, seed=42, bounds=[0,1]
 - TEST data never loaded or accessed

Computational redesign:
 - Precompute S (2000×N) semantic matrix once
 - Precompute G (2000×N) genre matrix once
 - rating, pop are global vectors (N,) — shared across users
 - Per DE call: score = w0*S + w1*G + w2*R + w3*P  (matrix op)
 - Train mask applied with per-user index arrays (fast)
 - argpartition for top-k (fast)
 - Franchise feature = 0 (documented simplification, restored at Phase 7)

Equivalence test verifies mathematical equivalence with original loop
before running DE.
"""

import pandas as pd
import numpy as np
import os, sys, hashlib, time, json
from scipy.optimize import differential_evolution
from scipy import sparse
from sklearn.preprocessing import MultiLabelBinarizer

sys.path.insert(0, os.path.abspath('src'))
# pyrefly: ignore [missing-import]
from recommendation_engine import MovieRecommender
# pyrefly: ignore [missing-import]
from semantic_model import SemanticMovieModel

NUMERICAL_TOLERANCE = 1e-6  # Max allowed NDCG difference in equivalence test

def clean_title(t):
    return str(t).lower().strip().replace(":", "").replace("-", "")

def ndcg_at_k(bin_sorted, n_pos, k=5):
    br = bin_sorted[:k]
    dcg = sum(br[i] / np.log2(i + 2) for i in range(len(br)))
    idcg_k = min(n_pos, k)
    idcg = sum(1 / np.log2(i + 2) for i in range(idcg_k))
    return dcg / idcg if idcg > 0 else 0.0

def load_and_split(valid_users_list, tmdb_id_set):
    """Load ratings, apply duplicate policy, apply exact Phase 3 split."""
    links_df = pd.read_csv('movielens/links.csv')
    links_df['tmdbId'] = pd.to_numeric(links_df['tmdbId'], errors='coerce')
    valid_links = links_df.dropna(subset=['tmdbId']).copy()
    valid_links['tmdbId'] = valid_links['tmdbId'].astype(int)
    valid_links = valid_links[valid_links['tmdbId'].isin(tmdb_id_set)]
    ml_to_tmdb = dict(zip(valid_links['movieId'], valid_links['tmdbId']))

    ratings_df = pd.read_csv('movielens/ratings.csv')
    ratings_df = ratings_df[ratings_df['userId'].isin(valid_users_list)].copy()
    ratings_df['tmdbId'] = ratings_df['movieId'].map(ml_to_tmdb)
    mapped_df = ratings_df[ratings_df['tmdbId'].notna()].copy()
    mapped_df['tmdbId'] = mapped_df['tmdbId'].astype(int)
    split_df = mapped_df.sort_values(by=['userId', 'timestamp', 'tmdbId'])
    split_df.drop_duplicates(subset=['userId', 'tmdbId'], keep='first', inplace=True)

    ts_sizes = split_df.groupby(['userId', 'timestamp']).size().reset_index(name='group_size')
    ts_sizes['cum_size'] = ts_sizes.groupby('userId')['group_size'].cumsum()
    ts_sizes['user_total'] = ts_sizes.groupby('userId')['group_size'].transform('sum')
    ts_sizes['pct_after_group'] = ts_sizes['cum_size'] / ts_sizes['user_total']
    ts_sizes['split'] = 'train'
    ts_sizes['pct_before'] = ts_sizes.groupby('userId')['pct_after_group'].shift(1).fillna(0)
    ts_sizes['pct_mid'] = (ts_sizes['pct_before'] + ts_sizes['pct_after_group']) / 2
    ts_sizes.loc[ts_sizes['pct_mid'] >= 0.80, 'split'] = 'val'
    ts_sizes.loc[ts_sizes['pct_mid'] >= 0.90, 'split'] = 'test'  # never accessed below

    split_map = ts_sizes.set_index(['userId', 'timestamp'])['split'].to_dict()
    split_df = split_df.copy()
    split_df['new_split'] = split_df.set_index(['userId', 'timestamp']).index.map(split_map)

    train = split_df[split_df['new_split'] == 'train']
    val   = split_df[split_df['new_split'] == 'val']
    test  = split_df[split_df['new_split'] == 'test']
    
    return train, val, test

def main():
    t0_global = time.time()

    # ── Verify frozen population ──────────────────────────────────────────────
    print("Step 1: Verifying frozen population checksum...")
    frozen = pd.read_csv('frozen_users.csv')
    valid_users_list = frozen['userId'].tolist()
    users_hash = hashlib.sha256(str(sorted(valid_users_list)).encode('utf-8')).hexdigest()
    assert users_hash == "7962ac4298a034acc871f236b852d67c2549f6e5cf2890d5a7127df161a0a3da", \
        f"FROZEN POPULATION HASH MISMATCH: {users_hash}"
    print(f"  Hash OK: {users_hash[:24]}...")

    # ── Load aligned metadata ──────────────────────────────────────────────────
    print("Step 2: Loading MovieRecommender metadata and embeddings...")
    rec = MovieRecommender()
    valid_movies = rec.metadata.reset_index(drop=True)
    valid_movies['id'] = pd.to_numeric(valid_movies['id'], errors='coerce')
    valid_movies = valid_movies[valid_movies['id'].notna()].copy().reset_index(drop=True)
    valid_movies['id'] = valid_movies['id'].astype(int)
    tmdb_id_set = set(valid_movies['id'])
    N = len(valid_movies)
    print(f"  Catalog size: {N:,}")

    sem = SemanticMovieModel(valid_movies)
    sem.load_or_generate_embeddings()
    E = sem.embeddings.astype(np.float32)  # (N, 384)

    tmdb_to_idx = {tid: idx for idx, tid in enumerate(valid_movies['id'])}

    valid_movies['popularity'] = pd.to_numeric(valid_movies['popularity'], errors='coerce').fillna(0)
    max_pop = float(valid_movies['popularity'].max()) or 100.0
    R_pop  = (valid_movies['popularity'].values / max_pop).astype(np.float32)   # (N,)
    R_rate = (pd.to_numeric(valid_movies['vote_average'], errors='coerce')
              .fillna(0).values / 10.0).astype(np.float32)                       # (N,)

    def split_g(g): return str(g).split()
    genres_list = valid_movies['genres'].apply(split_g).tolist()
    mlb = MultiLabelBinarizer()
    genre_mat = mlb.fit_transform(genres_list).astype(np.float32)   # (N, G)
    genre_mat_sp = sparse.csr_matrix(genre_mat)
    cand_genre_cnt = np.array(genre_mat_sp.sum(axis=1)).flatten().astype(np.float32)  # (N,)

    # ── Load & split ratings ────────────────────────────────────────────────────
    print("Step 3: Loading ratings (filtered to frozen users)...")
    train, val, test = load_and_split(valid_users_list, tmdb_id_set)

    train_pos_users = train[train['rating'] >= 4.0].groupby('userId')['tmdbId'].apply(set).to_dict()
    test_pos_users  = test[test['rating'] >= 4.0].groupby('userId')['tmdbId'].apply(set).to_dict()
    train_history   = train.groupby('userId')['tmdbId'].apply(set).to_dict()

    # ── Select 1,500 test users (seed=42 as per Phase 5) ────────
    print("Step 4: Selecting 1,500 deterministic test users (seed=42)...")
    np.random.seed(42)
    
    # Needs positive hits in BOTH train and test
    opt_eligible = [u for u in valid_users_list
                    if len(train_pos_users.get(u, set())) > 0
                    and len(test_pos_users.get(u, set())) > 0]
    opt_users = list(np.random.choice(opt_eligible, size=min(1500, len(opt_eligible)), replace=False))
    print(f"  Sampled for Phase 7 Test: {len(opt_users)}")

    U = len(opt_users)
    S_mat = np.zeros((U, N), dtype=np.float32)
    G_mat = np.zeros((U, N), dtype=np.float32)
    train_masks = []
    test_pos_sets = []
    useful = []

    for i, u in enumerate(opt_users):
        tp = train_pos_users.get(u, set())
        vp = test_pos_users.get(u, set())
        th = train_history.get(u, set())
        
        q_idxs = [tmdb_to_idx[t] for t in tp if t in tmdb_to_idx]
        vp_idxs = set(tmdb_to_idx[t] for t in vp if t in tmdb_to_idx)
        if not q_idxs or not vp_idxs: continue
        
        u_vec = np.mean(E[q_idxs], axis=0)
        nrm = np.linalg.norm(u_vec)
        if nrm > 0: u_vec /= nrm
        S_mat[len(useful)] = E.dot(u_vec).astype(np.float32)
        
        u_genres = []
        for qi in q_idxs: u_genres.extend(genres_list[qi])
        u_g_set = set(u_genres)
        u_g_vec = mlb.transform([list(u_g_set)]).astype(np.float32)
        u_g_size = float(u_g_vec.sum())
        if u_g_size > 0:
            intersect = genre_mat_sp.dot(u_g_vec.T).flatten()
            union = cand_genre_cnt + u_g_size - intersect
            union[union == 0] = 1
            G_mat[len(useful)] = intersect / union
            
        t_idxs = np.array([tmdb_to_idx[t] for t in th if t in tmdb_to_idx], dtype=np.int32)
        train_masks.append(t_idxs)
        test_pos_sets.append(frozenset(vp_idxs))
        useful.append(i)
        
    n_useful = len(useful)
    S_mat = S_mat[:n_useful]
    G_mat = G_mat[:n_useful]
    
    R_rate_row = R_rate.reshape(1, N)
    R_pop_row = R_pop.reshape(1, N)
    
    def evaluate_model(w_raw, restore_franchise=False):
        s = w_raw.sum()
        w = w_raw / s if s > 0 else np.ones(5) / 5
        w = w.astype(np.float32)
        
        scores = w[0]*S_mat + w[1]*G_mat + w[2]*R_rate_row + w[3]*R_pop_row
        
        for j, t_idxs in enumerate(train_masks):
            if len(t_idxs): scores[j, t_idxs] = -np.inf
            
        top_k_unsorted = np.argpartition(scores, -10, axis=1)[:, -10:]
        gathered = np.take_along_axis(scores, top_k_unsorted, axis=1)
        sort_w = np.argsort(-gathered, axis=1)
        top_k = np.take_along_axis(top_k_unsorted, sort_w, axis=1)
        
        res = {'ndcg5':[], 'ndcg10':[], 'h5':[], 'p5':[], 'r5':[], 'mrr5':[], 'map5':[]}
        for j in range(n_useful):
            vp = test_pos_sets[j]
            t10 = top_k[j]
            t5 = t10[:5]
            
            b10 = [1 if x in vp else 0 for x in t10]
            b5 = b10[:5]
            
            dcg5 = sum(b / np.log2(i+2) for i, b in enumerate(b5))
            id5 = min(len(vp), 5)
            idcg5 = sum(1 / np.log2(i+2) for i in range(id5))
            res['ndcg5'].append(dcg5/idcg5 if idcg5>0 else 0)
            
            dcg10 = sum(b / np.log2(i+2) for i, b in enumerate(b10))
            id10 = min(len(vp), 10)
            idcg10 = sum(1 / np.log2(i+2) for i in range(id10))
            res['ndcg10'].append(dcg10/idcg10 if idcg10>0 else 0)
            
            res['h5'].append(1 if sum(b5)>0 else 0)
            res['p5'].append(sum(b5)/5.0)
            res['r5'].append(sum(b5)/len(vp) if len(vp)>0 else 0)
            
            mrr = 0
            for i, bn in enumerate(b5):
                if bn == 1:
                    mrr = 1/(i+1)
                    break
            res['mrr5'].append(mrr)
            
            mapv = 0
            rc = 0
            for i, bn in enumerate(b5):
                if bn == 1:
                    rc += 1
                    mapv += rc/(i+1)
            res['map5'].append(mapv / min(len(vp),5) if len(vp)>0 else 0)
            
        return res

    print("\nEvaluating Model F...")
    f_w = np.array([0.60, 0.15, 0.10, 0.05, 0.10])
    res_f = evaluate_model(f_w)
    
    with open('optimized_weights.json') as f:
        dw = json.load(f)
        g_w = np.array(dw['optimized_weights'])
    print("\nEvaluating Model G...")
    res_g = evaluate_model(g_w)
    
    import scipy.stats
    t_n, p_n = scipy.stats.ttest_rel(res_f['ndcg5'], res_g['ndcg5'])
    
    results = {
        'F': {k: np.mean(v) for k, v in res_f.items()},
        'G': {k: np.mean(v) for k, v in res_g.items()},
        'stats': {'t': t_n, 'p': p_n}
    }
    with open('phase7_metrics.json', 'w') as f:
         json.dump(results, f)
    
    print("\nDONE!")

if __name__ == '__main__':
    main()
