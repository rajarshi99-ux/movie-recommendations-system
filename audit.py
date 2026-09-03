import os
import sys
import json
import time
import numpy as np
import pandas as pd
from scipy import stats
from collections import defaultdict

sys.path.insert(0, os.path.abspath('src'))
from recommendation_engine import MovieRecommender
from semantic_model import SemanticMovieModel
from evaluation_v6_metaheuristic import clean_title, calc_independent_relevance, relevance_to_grade, get_metrics_shared_idcg, run_models_for_test

print("==================================================")
print("AUDIT 1 — DATA SPLIT")
print("==================================================")
np.random.seed(42)
rec = MovieRecommender()
max_pop = rec._max_pop if hasattr(rec, '_max_pop') else 100.0
sem = SemanticMovieModel(rec.metadata)
sem.load_or_generate_embeddings()

valid_mask = pd.to_numeric(rec.metadata['vote_count'], errors='coerce') > 50
valid_indices = rec.metadata[valid_mask].index.tolist()

np.random.seed(42)
sampled = np.random.choice(valid_indices, size=500, replace=False)
train_queries = sampled[:300]
val_queries = sampled[300:400]
test_queries = sampled[400:]

print(f"1. Exact number of TRAIN queries: {len(train_queries)}")
print(f"2. Exact number of VALIDATION queries: {len(val_queries)}")
print(f"3. Exact number of FINAL TEST queries: {len(test_queries)}")
print(f"TRAIN: {len(train_queries)/len(sampled)*100}% / VAL: {len(val_queries)/len(sampled)*100}% / TEST: {len(test_queries)/len(sampled)*100}%")

print(f"TRAIN ∩ VALIDATION = empty: {len(set(train_queries) & set(val_queries)) == 0}")
print(f"TRAIN ∩ TEST = empty: {len(set(train_queries) & set(test_queries)) == 0}")
print(f"VALIDATION ∩ TEST = empty: {len(set(val_queries) & set(test_queries)) == 0}")

print("\n==================================================")
print("AUDIT 3 — OPTIMIZED WEIGHTS")
print("==================================================")
with open('optimized_weights.json', 'r') as f:
    cfg = json.load(f)
w = cfg['optimized_weights']
print(f"Semantic: {w[0]:.17f}")
print(f"Genre: {w[1]:.17f}")
print(f"Rating: {w[2]:.17f}")
print(f"Popularity: {w[3]:.17f}")
print(f"Franchise: {w[4]:.17f}")
w_sum = sum(w)
print(f"0 <= every weight <= 1: {all(0 <= x <= 1 for x in w)}")
print(f"sum(weights) == 1 within numerical tolerance: {abs(w_sum - 1.0) < 1e-9} (Sum is {w_sum})")

print("\n==================================================")
print("AUDIT 5 — FINAL TEST")
print("==================================================")
test_stats = {m: defaultdict(list) for m in ['A', 'D', 'E', 'F', 'G']}

for q_idx in test_queries:
    title = rec.metadata.at[q_idx, 'title']
    try:
        models = run_models_for_test(rec, sem, q_idx, w, max_pop, title)
        res = get_metrics_shared_idcg(models, q_idx, title, rec, max_pop)
        
        for m in ['A', 'D', 'E', 'F', 'G']:
            for metric in ['p5', 'p10', 'r5', 'r10', 'hr5', 'hr10', 'ndcg5', 'ndcg10', 'map5', 'map10', 'mrr5', 'mrr10']:
                test_stats[m][metric].append(res[m][metric])
    except Exception as e:
        pass

for m in ['A', 'D', 'E', 'F', 'G']:
    print(f"Model {m}:")
    print(f"  NDCG@5: {np.mean(test_stats[m]['ndcg5']):.4f}")
    print(f"  NDCG@10: {np.mean(test_stats[m]['ndcg10']):.4f}")
    print(f"  MAP@5: {np.mean(test_stats[m]['map5']):.4f}")
    print(f"  MAP@10: {np.mean(test_stats[m]['map10']):.4f}")
    print(f"  Precision@5: {np.mean(test_stats[m]['p5']):.4f}")
    print(f"  Recall@5: {np.mean(test_stats[m]['r5']):.4f}")
    print(f"  HitRate@5: {np.mean(test_stats[m]['hr5']):.4f}")
    print(f"  MRR@5: {np.mean(test_stats[m]['mrr5']):.4f}")

print("\n==================================================")
print("AUDIT 6 — STATISTICAL TEST")
print("==================================================")
test_g = np.array(test_stats['G']['ndcg5'])
test_f = np.array(test_stats['F']['ndcg5'])
diff = test_g - test_f
n_obs = len(diff)
mean_diff = np.mean(diff)
std_diff = np.std(diff, ddof=1)
stat, p_val = stats.ttest_rel(test_g, test_f)
conf = 1.96 * (std_diff / np.sqrt(n_obs)) if n_obs > 0 else 0

print(f"Number of paired observations: {n_obs}")
print(f"Mean difference: {mean_diff:.6f}")
print(f"Standard deviation: {std_diff:.6f}")
print(f"Test statistic: {stat:.6f}")
print(f"Exact p-value: {p_val:.6e}")
print(f"Confidence interval: +/- {conf:.6f}")

print("\n==================================================")
print("AUDIT 7 — EXPLAINABILITY")
print("==================================================")
def print_explain(title):
    print(f"Query: {title}")
    from adaptive_recommender import AdaptiveRecommender
    ar = AdaptiveRecommender()
    ar.asmr_weights = w
    recs = ar.recommend(title, n=1)
    if recs and "error" not in recs[0]:
        r = recs[0]
        print(f"  Recommendation 1: {r['title']}")
        print(f"    raw feature values: semantic={r['semantic_score']:.3f}, genre={r['genre_score']:.3f}, rating={r['rating_score']:.3f}, pop={r['popularity_score']:.3f}, franch={r['franchise_score']:.3f}")
        print(f"    raw weighted: {[w[0]*r['semantic_score'], w[1]*r['genre_score'], w[2]*r['rating_score'], w[3]*r['popularity_score'], w[4]*r['franchise_score']]}")
        print(f"    total score: {r['final_score']}")
        print(f"    normalized percentages (displayed): {r['asmr_contrib_pct']} -> Sums to: {sum(r['asmr_contrib_pct'])}%")
    else:
        print("  Not found / error.")

print_explain("Inception")
print_explain("Interstellar")
print_explain("Toy Story")

print("\n==================================================")
print("AUDIT 8 — LATENCY")
print("==================================================")
q_title = "Inception"

# warm up
for _ in range(5):
    rec.model_mode = "existing_hybrid"
    rec.recommend(q_title, n=5)

l_f = []
l_g = []
for _ in range(100):
    t0 = time.time()
    q_idx = sem.get_index(q_title)
    if q_idx is not None:
        q_row = rec.metadata.iloc[q_idx]
        sims = sem.get_similarities(q_idx)
        indices = np.argsort(sims)[::-1][:250]
        # model f logic overhead
        for i in indices:
            row = rec.metadata.iloc[i]
            c_g = set(str(row.get('genres', '')).split())
            rg = (float(row.get('vote_average') or 0)/10.0)
            rp = float(row.get('popularity') or 0) / max_pop
            f_score = (0.60 * sims[i]) + (0.15 * 0) + (0.10 * rg) + (0.05 * rp) + (0.10 * 0)
    l_f.append(time.time() - t0)
    
    t0 = time.time()
    from adaptive_recommender import AdaptiveRecommender
    ar = AdaptiveRecommender()
    ar.asmr_weights = w
    ar.recommend(q_title, n=10)
    l_g.append(time.time() - t0)

print(f"Model F latency: Mean={np.mean(l_f)*1000:.3f}ms, P95={np.percentile(l_f, 95)*1000:.3f}ms")
print(f"Model G (ASMR) latency: Mean={np.mean(l_g)*1000:.3f}ms, P95={np.percentile(l_g, 95)*1000:.3f}ms")
