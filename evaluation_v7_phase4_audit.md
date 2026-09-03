# Evaluation V7 - Phase 4 Behavioral Protocol Audit

## A. Frozen Population Verification
- **Hash Checked:** `7962ac4298a034acc871f236b852d67c2549f6e5cf2890d5a7127df161a0a3da` -> **SUCCESS**. The identical 14,581 array is validated.

## B. Interaction Mapping Statistics
- **Total Unique Users Evaluated:** 14,581 (100.0% coverage of frozen pool).
- **Unique Processed TMDB Movies:** 17,019
- **TRAIN Interactions:** 1,706,917
- **VALIDATION Interactions:** 214,093
- **TEST Interactions:** 213,389
*(Note: Mapping maintains structural 80/10/10 timeline splits mathematically.)*

## C. Training History Extraction
- **Total Ratings In Train Phase:** 1,706,917
- **Average Train Ratings per User:** 117.1
- **Mean Historical Rating Score:** 3.61/5.0
- **Total Historical Positive Ratings (>=4):** 863,198
- **Average Positive Preference Density:** 54.36% of user history yields a $\ge 4$ rating.
- **Median Training Activity Span:** 1,858 seconds *(No future info accessed)*.

## D/F/G. Evaluability & Ground Truth Target Mechanics
- **Ground Truth Target Vector:** Only items rated strictly $\ge 4.0$ uniquely in future blocks apply as independent positive markers. Unrated or low-rated items process natively as Negative components in ranking DCG mathematics.
- **Candidate Eligibility Rule:** For testing a target block (e.g. TEST), we exclude *all* TRAIN interactions globally per user to prevent hallucinating success by re-recommending known items.
- **Full Catalog Feasibility:** YES. The engine computes semantic/genre dot products natively in vectorized formats. A 45,000 item dot product on a 384D user-preference vector takes <5 milliseconds per user. Full catalog ranking ensures zero negative-sampling bias.

## E. Repeated-Interaction Anomaly Audit
- **Identified repeat rating entries (same user, same movie):** 4 interactions affecting 2 distinct User+Movie combinations globally.
- **Cross-Partition Repeats:** 1 boundaries violated where a user re-rated a movie initially present in Training later on in Validation or Test.
- **Resolution Policy Locked:** Keep ONLY the temporally FIRST interaction functionally active. Drop subsequent repeats (especially if they leak from TRAIN into TEST as positive signals). Realistically, recommendations shouldn't re-recommend items the user already rated, regardless of if they eventually rate it again.

## H. Cold-Start Missing Information Restrictions
- **Users completely missing Train History:** 0
- **Users missing valid Positive ($\ge 4$) Validation targets:** 1031
- **Users missing valid Positive ($\ge 4$) Test targets:** 1007
- **Calculated Metric Eliqibility (Final Test Evaluators):** 13,574 users correctly possess mathematical prerequisites (both a train anchor and at least one positive test target) to compute final Mean Average Precision.

## I. Temporal Sanity Gates
- `max_train < min_val` isolation passed? **TRUE (0 Leakages asserted)**
- `max_val < min_test` isolation passed? **TRUE (0 Leakages asserted)**

## J. Profile Construction Strategy Proposal

**Proposed User Profile Construction (Strictly from TRAIN Interactions):**
1. **Semantic Profile:** For every movie rated $\ge 4.0$ in the user's TRAIN set, extract its `SentenceTransformer` 384D vector. Compute the mean pool of these vectors to form a single continuous 384D `user_semantic_core`. (Inference becomes the cosine/dot product of this core against the 45,000 candidate dataset vectors).
2. **Genre Profile:** Extract all assigned genres from the user's TRAIN $\ge 4.0$ items, build a flattened frequency distribution set. Compare via weighted subset targeting during the Genre-hybrid iteration.
3. **Pessimistic Anchoring (Optional):** Subtract semantic properties of heavily penalized TRAIN items (Rating $\le 2.0$) from the `user_semantic_core`.
*Note: Validations and test candidates dynamically filter out anything inside the user's total TRAIN history mapping.*

