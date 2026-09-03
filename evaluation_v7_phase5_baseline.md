# Evaluation V7 - Phase 5 Model F Real User-Behavior Baseline

## A. Frozen Population Verification
- **Total Frozen Users:** 14,581
- **Users with Valid Train/Test Active Behavior Targets:** 13,567
- **Statistically Sampled Model F Calculations:** 1,500
- **Exclusion Reasons:** 1014 users naturally excluded because their Test subsets contained zero items rated strictly >= 4.0. The models inherently cannot evaluate Top-K retrieval precision internally when there are zero mathematically true positives available to retrieve in the chronological future slice.

## B. Sanity Asserts Confirmed
- `max_train < min_val`, `max_val < min_test` maintained strictly.
- **Duplicate Policy:** First chronological appearance stored exactly once. Subsequent identical movie IDs from identical users are explicitly wiped before train bounding.
- **Train Exclusion Candidate Filter:** Confirmed (-Inf applied bounding all historical `TRAIN` vectors identically. Hallucinated memorizations are algorithmically impossible).
- **No TEST Leakage in Configs:** Profiles calculate standard mathematical derivations against strictly localized subset items exactly inside `TRAIN >= 4.0` bins. None of Model F coefficients were modified.

## C. Evaluation Environment Context
- **Candidate Pool:** 45,466 Movies (Strictly zero negatives-sampling biases applied. Vectorized `numpy` handles complete catalog multiplication per candidate evaluation taking ~6.5ms mapping string boundaries against dot products intrinsically).
- **Target Ground Truth:** Implicit TEST instances rating $ \ge 4.0 $.
- **Model F Equations Evaluated:** Same unmodified manual hybrid parameters -> `W = { 0.60, 0.15, 0.10, 0.05, 0.10 }`.

## D. Ground Truth Final Target Results
*(Calculated out against evaluated N=1500 users identically)*

| Metric | Mean (Model F Base) | Median | Standard Deviation | 95% Confidence Interval |
|---|---|---|---|---|
| **NDCG@5** | 0.0026 | 0.0000 | 0.0313 | ±0.0016 |
| **NDCG@10**| 0.0032 | 0.0000 | 0.0320 | ±0.0016 |
| **MAP@5** | 0.0015 | 0.0000 | 0.0209 | ±0.0011 |
| **MAP@10** | 0.0016 | 0.0000 | 0.0205 | ±0.0010 |
| **Precision@5** | 0.0016 | 0.0000 | 0.0178 | ±0.0009 |
| **Recall@5** | 0.0022 | 0.0000 | 0.0347 | ±0.0018 |
| **HitRate@5** | 0.0080 | 0.0000 | 0.0891 | ±0.0045 |
| **MRR@5** | 0.0051 | 0.0000 | 0.0658 | ±0.0033 |

---
**PHASE 5 PASSED — MODEL F BASELINE FROZEN**
