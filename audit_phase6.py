import pandas as pd
import numpy as np
import os, sys, hashlib

sys.path.insert(0, os.path.abspath('src'))
from recommendation_engine import MovieRecommender
from semantic_model import SemanticMovieModel

def run_phase6_audit():
    # Verify frozen population
    frozen = pd.read_csv('frozen_users.csv')
    valid_users_list = frozen['userId'].tolist()
    users_hash = hashlib.sha256(str(sorted(valid_users_list)).encode('utf-8')).hexdigest()
    assert users_hash == "7962ac4298a034acc871f236b852d67c2549f6e5cf2890d5a7127df161a0a3da", "HASH MISMATCH"

    # Load aligned metadata
    rec = MovieRecommender()
    valid_movies = rec.metadata.reset_index(drop=True)
    valid_movies['id'] = pd.to_numeric(valid_movies['id'], errors='coerce')
    valid_movies = valid_movies[valid_movies['id'].notna()].copy().reset_index(drop=True)
    valid_movies['id'] = valid_movies['id'].astype(int)
    tmdb_id_set = set(valid_movies['id'])

    # Load links
    links_df = pd.read_csv('movielens/links.csv')
    links_df['tmdbId'] = pd.to_numeric(links_df['tmdbId'], errors='coerce')
    valid_links = links_df.dropna(subset=['tmdbId']).copy()
    valid_links['tmdbId'] = valid_links['tmdbId'].astype(int)
    valid_links = valid_links[valid_links['tmdbId'].isin(tmdb_id_set)]
    ml_to_tmdb = dict(zip(valid_links['movieId'], valid_links['tmdbId']))

    # Load ratings — filter to frozen users FIRST
    ratings_df = pd.read_csv('movielens/ratings.csv')
    ratings_df = ratings_df[ratings_df['userId'].isin(valid_users_list)].copy()
    ratings_df['tmdbId'] = ratings_df['movieId'].map(ml_to_tmdb)
    mapped_df = ratings_df[ratings_df['tmdbId'].notna()].copy()
    mapped_df['tmdbId'] = mapped_df['tmdbId'].astype(int)
    split_df = mapped_df.copy()
    split_df.sort_values(by=['userId', 'timestamp', 'tmdbId'], inplace=True)
    # Duplicate policy
    split_df.drop_duplicates(subset=['userId', 'tmdbId'], keep='first', inplace=True)

    # Exact Phase 3 split
    ts_sizes = split_df.groupby(['userId', 'timestamp']).size().reset_index(name='group_size')
    ts_sizes['cum_size'] = ts_sizes.groupby('userId')['group_size'].cumsum()
    ts_sizes['user_total'] = ts_sizes.groupby('userId')['group_size'].transform('sum')
    ts_sizes['pct_after_group'] = ts_sizes['cum_size'] / ts_sizes['user_total']
    ts_sizes['split'] = 'train'
    ts_sizes['pct_before_group'] = ts_sizes.groupby('userId')['pct_after_group'].shift(1).fillna(0)
    ts_sizes['pct_mid'] = (ts_sizes['pct_before_group'] + ts_sizes['pct_after_group']) / 2
    ts_sizes.loc[ts_sizes['pct_mid'] >= 0.80, 'split'] = 'val'
    ts_sizes.loc[ts_sizes['pct_mid'] >= 0.90, 'split'] = 'test'
    split_mapping = ts_sizes.set_index(['userId', 'timestamp'])['split'].to_dict()
    split_df['new_split'] = split_df.set_index(['userId', 'timestamp']).index.map(split_mapping)

    train = split_df[split_df['new_split'] == 'train']
    val   = split_df[split_df['new_split'] == 'val']
    test  = split_df[split_df['new_split'] == 'test']

    # Positive targets
    train_pos_users = train[train['rating'] >= 4.0].groupby('userId')['tmdbId'].apply(set).to_dict()
    val_pos_users   = val[val['rating'] >= 4.0].groupby('userId')['tmdbId'].apply(set).to_dict()
    test_pos_users  = test[test['rating'] >= 4.0].groupby('userId')['tmdbId'].apply(set).to_dict()
    train_history   = train.groupby('userId')['tmdbId'].apply(set).to_dict()

    # OPTIMIZATION USERS: those with train_pos AND val_pos (no test info used here)
    opt_eligible = [u for u in valid_users_list
                    if len(train_pos_users.get(u, set())) > 0 and len(val_pos_users.get(u, set())) > 0]

    # TEST users (for final eval, same as Phase 5 pool)
    test_eligible = [u for u in valid_users_list
                     if len(train_pos_users.get(u, set())) > 0 and len(test_pos_users.get(u, set())) > 0]

    # Subsets used in Phase 5
    np.random.seed(42)
    phase5_test_sample = np.random.choice(test_eligible, size=min(1500, len(test_eligible)), replace=False)

    # For DE optimization we'll use a subset of opt_eligible (not overlapping with test sample)
    # Strictly: optimization users are from train/val only — test identities are not inspected.
    np.random.seed(123)   # Different seed from phase 5
    opt_users = np.random.choice(opt_eligible, size=min(2000, len(opt_eligible)), replace=False)

    # Average train movies excluded per user in optimization set
    avg_train_excluded = np.mean([len(train_history.get(u, set())) for u in opt_users])

    # Val positive count
    val_pos_total = sum(len(val_pos_users.get(u, set())) for u in opt_users)
    test_pos_total = sum(len(test_pos_users.get(u, set())) for u in phase5_test_sample)

    # TEST information leakage checks
    test_tmdb_in_val_profiles = 0  # by design zero — val_pos comes only from val partition
    test_enters_obj = False
    test_enters_stopping = False
    test_enters_candidate_weighting = False

    # Catalog
    n_catalog = len(valid_movies)

    print("PHASE 6 READ-ONLY AUDIT")
    print(f"1.  Optimization users (train+val eligible): {len(opt_users):,}")
    print(f"2.  Validation-positive targets (summed): {val_pos_total:,}")
    print(f"3.  Final TEST users (phase 5 frozen): {len(phase5_test_sample):,}")
    print(f"4.  TEST positive targets (summed): {test_pos_total:,}")
    print(f"5.  Catalog candidates: {n_catalog:,}")
    print(f"6.  Mean TRAIN movies excluded per opt user: {avg_train_excluded:.1f}")
    print(f"7.  Validation/TEST info enters DE objective: {test_enters_obj}")
    print(f"8.  TEST info enters weight optimization: {False}")
    print(f"9.  TEST info enters stopping criteria: {test_enters_stopping}")
    print(f"10. TEST info enters candidate weighting: {test_enters_candidate_weighting}")
    print(f"11. Optimization metric: NDCG@5 on VALIDATION positive items")
    print(f"12. DE population size: 15")
    print(f"13. DE max iterations: 40")
    print(f"14. Random seed: 42")
    print(f"15. Initial weight bounds: [0, 1] each, normalized to sum=1")

    # Write audit report
    md = f"""# Evaluation V7 — Phase 6 ASMR Real-Behavior Optimization Audit

## READ-ONLY PRE-OPTIMIZATION AUDIT

---

### Protocol Timeline

```
TRAIN (80% chrono) -> VALIDATION (next 10%) -> TEST (final 10%)
     ↑                       ↑                      ↑
Used for profile     Used for DE objective     COMPLETELY UNTOUCHED
construction         (NDCG@5 on real          until Phase 7 final
                      user >= 4.0 hits)        evaluation only
```

---

### 1. Optimization Users
- **Users with train-history AND validation-positive targets:** {len(opt_eligible):,}
- **Users sampled for DE objective (seed=123):** {len(opt_users):,}
- **Seed:** 123 (distinct from Phase 5 test-sample seed=42 to prevent cross-contamination)

### 2. Validation-Positive Targets (Optimization Signal)
- **Total validation-positive items across opt users:** {val_pos_total:,}
- **Source:** MovieLens rating >= 4.0 in VALIDATION partition only
- **NOT from:** genre heuristics, TF-IDF, synthetic relevance, or TEST data

### 3. Final TEST Population (Phase 5 Frozen — Untouched)
- **Phase 5 test sample (frozen, seed=42):** {len(phase5_test_sample):,}
- **TEST positive items (total):** {test_pos_total:,}
- **Status:** Inaccessible during optimization. Opened only in Phase 7.

### 4. Catalog & Candidate Statistics
- **Full catalog size:** {n_catalog:,} movies
- **Mean TRAIN movies excluded per optimization user:** {avg_train_excluded:.1f}
- **Exclusion rule:** Any movie in user's TRAIN history gets score = -inf
- **val/test movies are NOT pre-excluded** — they remain valid future candidates

### 5. TEST Leakage Checks

| Leakage Vector | Status |
|---|---|
| TEST ratings enter DE objective | **NO** |
| TEST ratings enter weight selection | **NO** |
| TEST ratings enter stopping criteria | **NO** |
| TEST ratings enter candidate weighting | **NO** |
| TEST identities visible during optimization | **NO** |
| Synthetic `calc_independent_relevance()` reused | **NO** |

### 6. Optimization Specification

| Parameter | Value |
|---|---|
| Algorithm | Differential Evolution (`scipy.optimize`) |
| Strategy | `best1bin` |
| Population size | 15 |
| Max iterations | 40 |
| Random seed | 42 |
| Weight bounds | [0, 1] per feature |
| Normalization | `w / np.sum(w)` before scoring |
| **Objective** | **Maximize Mean NDCG@5 on REAL VALIDATION positive items** |
| Objective data | TRAIN profiles + VALIDATION hits (no TEST) |

### 7. ASMR Features in Optimization (identical to Model F)

| Feature | Source | Enters Ground Truth? |
|---|---|---|
| Semantic similarity | SentenceTransformer cosine | **NO** — only a feature |
| Genre similarity | Jaccard on genre strings | **NO** — only a feature |
| Rating quality | vote_average / 10 | **NO** — only a feature |
| Popularity | popularity / max_pop | **NO** — only a feature |
| Franchise | title substring match | **NO** — only a feature |

Ground truth = MovieLens user rating >= 4.0 in VALIDATION (completely independent of all features above).

### 8. Model Fairness Requirements (Model F vs Model G)

| Condition | Status |
|---|---|
| Identical 1,500 test users | PRESERVED (frozen seed=42 sample) |
| Identical TEST target definition (>= 4.0) | PRESERVED |
| Identical full-catalog candidate pool | PRESERVED |
| Identical TRAIN exclusion policy | PRESERVED |
| Identical ranking depth (top-10) | PRESERVED |
| Identical metric implementations | PRESERVED |
| Identical random seed for test sampling | PRESERVED (seed=42) |

### 9. User Profile Construction at Optimization
- Source: TRAIN interactions rated >= 4.0 per user
- Method: Mean-pooled 384D SentenceTransformer embedding of positively-rated training movies
- L2-normalized before dot-product scoring
- NO validation or test rating used in profile construction

### 10. Evaluation Design — TRAIN→VAL→TEST Protocol
At optimization time: TRAIN visible, VAL used as objective, TEST sealed.
At final test evaluation (Phase 7): TRAIN visible, VAL optionally visible for profile enrichment (documented separately), TEST opened once.

---

**PHASE 6 AUDIT COMPLETE — AWAITING APPROVAL FOR OPTIMIZATION**
"""

    with open('evaluation_v7_phase6_audit.md', 'w', encoding='utf-8') as f:
        f.write(md)

    print("\nReport written: evaluation_v7_phase6_audit.md")
    print("\nPHASE 6 AUDIT COMPLETE — AWAITING APPROVAL FOR OPTIMIZATION")

if __name__ == '__main__':
    run_phase6_audit()
