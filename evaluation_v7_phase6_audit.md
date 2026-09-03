# Evaluation V7 — Phase 6 ASMR Real-Behavior Optimization Audit

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
- **Users with train-history AND validation-positive targets:** 13,544
- **Users sampled for DE objective (seed=123):** 2,000
- **Seed:** 123 (distinct from Phase 5 test-sample seed=42 to prevent cross-contamination)

### 2. Validation-Positive Targets (Optimization Signal)
- **Total validation-positive items across opt users:** 15,303
- **Source:** MovieLens rating >= 4.0 in VALIDATION partition only
- **NOT from:** genre heuristics, TF-IDF, synthetic relevance, or TEST data

### 3. Final TEST Population (Phase 5 Frozen — Untouched)
- **Phase 5 test sample (frozen, seed=42):** 1,500
- **TEST positive items (total):** 10,393
- **Status:** Inaccessible during optimization. Opened only in Phase 7.

### 4. Catalog & Candidate Statistics
- **Full catalog size:** 45,171 movies
- **Mean TRAIN movies excluded per optimization user:** 122.6
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
