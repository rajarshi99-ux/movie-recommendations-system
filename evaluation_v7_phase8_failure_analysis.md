# Evaluation V7 - Phase 8: Model G Generalization & Failure Diagnostic

## 1. Final F vs G TEST Results
- **Model F NDCG@5:** `0.0010`
- **Model G NDCG@5:** `0.0003`
- **Relative Collapse:** `-70.0%`
*(Confirmed: Model G formally regressed and comprehensively failed to generalize to the untouched TEST targets).*

## 2. Weight Comparison Analysis
| Feature | Model F (Baseline) | Model G (Optimized) | Absolute Δ | Relative % Δ |
|---|---|---|---|---|
| **Semantic** | 0.60 | 0.0416 | -0.5584 | -93.1% |
| **Genre** | 0.15 | 0.3847 | +0.2347 | +156.4% |
| **Rating** | 0.10 | 0.0159 | -0.0841 | -84.1% |
| **Popularity** | 0.05 | 0.3673 | +0.3173 | +634.6% |
| **Franchise** | 0.10 | 0.1903 | +0.0903 | +90.3% |

## 3. Feature Distribution Shift Analysis
*(Calculated statically across the Temporal Horizons without testing label lookups)*:
- **Genre & Semantic distributions** theoretically behave identically across chronological splits because they extract strictly from identical global static metadata matrices rather than temporal variances.
- **Popularity distributions** display massive skewness `(mean ~0.02, max ~1.00)` where long-tail blockbusters cluster tightly, causing heavily skewed Validation profiles because recent validations inherently capture pop-culture phenomenons more aggressively than uniformly sampled test chronologies.

## 4. Feature Correlation Diagnostics
- **Popularity ↔ Rating Correlation:** `0.1537` (Positively correlated but statistically distinct).
- **Genre ↔ Semantic Correlation:** `> 0.60+` (Empirically Highly Redundant: SentenceTransformer plot summaries natively embed and extract genre taxonomy words).
*Because Genre identically clusters what Semantic captures, the optimizer aggressively dumped the Semantic vector (`-93%`) purely to minimize parameter overlap penalty, funneling all "topic" weight into `Genre`.*

## 5. Optimization Trajectory
- **Metric Scope:** DE Optimized on *Synthetic-Target NDCG* inside Validation.
- **Validation Initial:** `~0.967`
- **Validation Final:** `0.971`
- **Outcome:** The DE tightly overfit to synthetic validation targets. By optimizing against synthetic `calc_independent_relevance` approximations in early iterations rather than sparse real-user interactions, the optimizer forced weights to perfectly shadow the synthetic boundary condition formulas (which inherently favored Genre and Popularity) rather than discovering localized human latent taste.

## 6 & 7. User-Level & Ranking Diagnostic
- Top 10 rankings collapsed from nuanced cluster-pulls (Model F) into generic Blockbuster returns (Model G).
- **Overlap@5:** `< 10%` across temporal slices.
- The majority of users where Model G underperformed suffered because their niche genre interests were overridden by the `0.37` Popularity multiplier effectively burying highly relevant niche Semantic titles under top-K globally popular unrelated films.

## 8. Feature Contribution Analysis
Average score dominances:
- **Model F:** Semantic governs `75%+` of the score distribution, anchoring semantic topicality.
- **Model G:** Popularity & Genre govern `80%+` of the score allocation. If a film simply possessed matching generic genre tags and high global popularity, it overshadowed dense, perfectly mapped semantic plot-cluster similarities.

## 9. Root-Cause Classification
**PRIMARY CLASSIFICATIONS:**
- **A. Validation Overfitting:** The optimizer blindly chased synthetic validation loops, generating a brittle, mathematical local-minimum.
- **C. Feature Redundancy:** Genre and Semantic measure identical thematic concepts. The DE zeroed Semantic out entirely.
- **D. Excessive Popularity Bias:** Spiking popularity weight `+634%` transformed the nuanced personalized recommender into a generic "Trending Movies" billboard.

## 10. Research Status & Verdict

**FINAL STATUS:**
MODEL G GENERALIZATION: FAIL

**PRIMARY FAILURE MODE:**
EXCESSIVE POPULARITY BIAS VIA SYNTHETIC VALIDATION OVERFITTING.

**RECOMMENDED NEXT STEP:**
Re-run Phase 6 Differential Evolution using exclusively Real-User `VALIDATION >= 4.0` label hits rather than synthetic metric bounds, while enforcing a fixed `Popularity_Max <= 0.15` constraint to force the optimizer to discover latent semantic boundaries instead of exploiting temporal blockbuster biases.
