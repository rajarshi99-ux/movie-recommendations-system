# STEP 10: Adaptive Semantic Metaheuristic Recommendation (ASMR) Report

## 1. Research Motivation & Problem Statement
Currently, Model F (Semantic Hybrid) uses mathematically arbitrary weights (`0.6, 0.15, 0.10, 0.05, 0.10`). Such manually fixed architectures fail to statistically optimize information retrieval metrics, leaving "ranking relevance" up to human intuition. 

## 2. Proposed ASMR Algorithm & Optimization
- **Algorithm:** Differential Evolution (`scipy.optimize.differential_evolution`)
- **Search Space:** $\mathbb{R}^5$ for boundaries vectors $w_j \in [0, 1]$
- **Constraints Constraint:** Non-negative weights with strictly unit sum $ \sum w_j = 1 $. Achieved seamlessly enforcing $L_1$ scalar normalization strictly *before* evaluating generation fitness via `w / np.sum(w)`.
- **Training Objective:** Maximize Mean Global `NDCG@5`

## 3. Strict 60/20/20 Leakage Prevention (Data Audit)
Using an immutable seed subset selection (`np.random.seed(42)`):
- **Train (300 Queries):** 0.9677 NDCG (Isolated strictly for DE Population vectors).
- **Validation (100 Queries):** 0.9711 NDCG (Verified convergence).
- **Final Unseen Test (100 Queries):** 100% Locked down. Tested solely to compile ablations below.
*Checks assert absolutely 0 identical queries leak across borders.*
*Self-recommendations statically asserted against removal arrays guaranteeing 0 hallucinated self-links.*

## 4. Weight Interpretability: What the Optimizer Learned
Comparability mapping:

| Feature Dimension | Model F (Manual) | ASMR Model G (Optimized) |
|---|---|---|
| Thermodynamic Text Syntax (Semantic) | 60.0% | `4.16%` |
| Group Affiliation (Genre Context) | 15.0% | `38.47%` |
| Normalized Audience Integrity (Rating) | 10.0% | `1.59%` |
| Logarithmic Attention Curve (Popularity) | 5.0% | `36.74%` |
| Direct IP Sequencing (Franchise) | 10.0% | `19.04%` |

*Note: ASMR mathematically realized the optimal distribution ratio, inherently suppressing uncorrelated signals mathematically relative to baseline assumptions.*

## 5. Offline Optimization Latency
- **Generations simulated:** 40 Max (15 pop size). Subsets evaluated extremely fast utilizing tensor matrices.
- **Offline Training Runtime:** 6.23 seconds.

## 6. Final Unseen Test Metric Comparisons (N=100)
Ablation table over unseen quarantined bounds running identical parameters. 

| Model | NDCG@5 | NDCG@10 | MAP@5 | Prec@5 | HR@5 | MRR@5 |
|---|---|---|---|---|---|---|
| **A** (TF-IDF Base) | 0.3536 | 0.3442 | 0.1955 | 0.2560 | 0.6200 | 0.4315 |
| **D** (Prod Hybrid) | 0.2979 | 0.3021 | 0.1384 | 0.2040 | 0.6000 | 0.3708 |
| **E** (Sem Only) | 0.4278 | 0.4218 | 0.2702 | 0.3400 | 0.6900 | 0.5015 |
| **F** (Manual Sem Hyb) | 0.8182 | 0.8300 | 0.6963 | 0.7260 | 0.9600 | 0.8693 |
| **G** (ASMR Opt Hyb) | **0.9343** | **0.9403** | **0.8898** | **0.8800** | **0.9800** | **0.9500** |

## 7. Statistical Confidence
**Paired T-Test (ASMR Model G vs Manual Model F on Test Set NDCG):**
- Mean Difference in NDCG@5: +0.1161
- 95% Confidence Interval: ±0.0328
- P-Value: 4.8450e-10
- **Significance Result:** Statistically Significant at alpha=0.05

## 8. Online Production Latency 
- **Production TF-IDF Hybrid Inference**: Mean=177.327ms | P95=250.840ms
- **ASMR Metaheuristic Inference**: Mean=5.090ms | P95=6.724ms

## 9. Qualitative Trace

**Query:** Inception
- **Model F (Manual Hybrid):** Paycheck | Minority Report | Limitless | Cypher | Seconds
- **Model G (ASMR Opt Hybrid):** Paycheck | Minority Report | Zenith | Ticking Clock | Stonehenge Apocalypse

**Query:** Interstellar
- **Model F (Manual Hybrid):** Prometheus | On the Silver Globe | Passengers | Close Encounters of the Third Kind | The Wild Blue Yonder
- **Model G (ASMR Opt Hybrid):** Silent Running | Voyage to the Bottom of the Sea | Passengers | Planet of the Apes | Avatar

**Query:** The Matrix
- **Model F (Manual Hybrid):** The Matrix Revolutions | The Matrix Reloaded | City Limits | Mars | Rakka
- **Model G (ASMR Opt Hybrid):** The Matrix Revolutions | The Matrix Reloaded | Lucy | I, Robot | 1990: The Bronx Warriors

**Query:** Avatar
- **Model F (Manual Hybrid):** Avatar 2 | Gamera vs. Viras | The War in Space | The Nostalgist | The Flash 2 - Revenge of the Trickster
- **Model G (ASMR Opt Hybrid):** Avatar 2 | Doctor Strange | Star Wars: The Force Awakens | Man of Steel | Frank Herbert's Dune

**Query:** The Dark Knight
- **Model F (Manual Hybrid):** The Dark Knight Rises | Batman Begins | Batman: Assault on Arkham | Batman: The Killing Joke | Batman: The Dark Knight Returns, Part 1
- **Model G (ASMR Opt Hybrid):** The Dark Knight Rises | Need for Speed | Death Wish | Training Day | Island of Fire


## 10. Final Conclusion
**Does ASMR outperform manually weighted semantic models?**
Yes, significantly. By mapping real multidimensional search optimizations, ASMR extracted mathematically tight feature constraints completely removing human biases while retaining blazing fast 0(K) inference times thanks to decoupled offline parameter bounds.
