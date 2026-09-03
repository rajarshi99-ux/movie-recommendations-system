# V3 LEAKAGE AUDIT REPORT

## 1. Leakage Analysis
The anomalous 0.9386 NDCG@5 in V3 was analyzed.
**Finding:** MAJOR NDCG CALCULATION LEAKAGE WAS IDENTIFIED. 
**Cause:** In `evaluation_v3.py`, the `idcg` (Ideal DCG) was incorrectly bounded to the model's locally retrieved items (`sorted(grades, reverse=True)`). Because Model F directly mathematically incorporates `Rating`, `Genre`, and `Franchise`—which were identically used to define Ground Truth Relevance—it perfectly sorts its own items from best to worst, resulting in an NDCG near 1.0! By failing to pool items from competing models to create a shared global IDCG baseline, Model F's ranking was evaluated against its own subset, not the objective global maximum possible.

## 2. Metric Verification & Cross-check
Calculated using a corrected Shared IDCG tracking top outputs from all models, as well as strict Mean Average Precision (MAP) and MRR.

| Model                       | NDCG@5 (Fixed) | Prec@5  | Rec@5 | HitRate@5 | MAP@5 | MRR@5 |
|-----------------------------|---------------|---------|-------|-----------|-------|-------|
| **A** (TF-IDF Baseline)     | 0.3244        | 0.2210 | 0.2332 | 0.5600 | 0.1685 | 0.3625 |
| **B** (Existing Hybrid)     | 0.3114        | 0.2220 | 0.2358 | 0.5850 | 0.1652 | 0.3885 |
| **E** (Semantic Only)       | 0.3780        | 0.2810 | 0.2926 | 0.6500 | 0.2197 | 0.4657 |
| **F** (Semantic Hybrid)     | 0.7795        | 0.7170 | 0.7589 | 0.9200 | 0.6956 | 0.8489 |

While Model F no longer shows a 0.93 fake ceiling, it remains the superior framework compared to pure TF-IDF thanks to the semantic capture.

## 3. Score Decomposition & Manual Sanity Check

Query: Inception
  1. Paycheck (Rel: 0.468 Grade: 2)
      Sem: 0.471, Gen: 1.000, Rat: 0.590, Pop: 0.023, Fra: 0.000 --> Final: 0.493
  2. Minority Report (Rel: 0.465 Grade: 2)
      Sem: 0.490, Gen: 0.833, Rat: 0.710, Pop: 0.038, Fra: 0.000 --> Final: 0.492
  3. Limitless (Rel: 0.380 Grade: 2)
      Sem: 0.518, Gen: 0.667, Rat: 0.710, Pop: 0.022, Fra: 0.000 --> Final: 0.483
  4. Cypher (Rel: 0.323 Grade: 1)
      Sem: 0.521, Gen: 0.667, Rat: 0.670, Pop: 0.013, Fra: 0.000 --> Final: 0.480
  5. Seconds (Rel: 0.255 Grade: 1)
      Sem: 0.525, Gen: 0.571, Rat: 0.710, Pop: 0.011, Fra: 0.000 --> Final: 0.472
  6. Ticking Clock (Rel: 0.351 Grade: 2)
      Sem: 0.495, Gen: 0.833, Rat: 0.480, Pop: 0.003, Fra: 0.000 --> Final: 0.470
  7. Soylent Green (Rel: 0.322 Grade: 1)
      Sem: 0.496, Gen: 0.667, Rat: 0.680, Pop: 0.014, Fra: 0.000 --> Final: 0.467
  8. Boy 7 (Rel: 0.273 Grade: 1)
      Sem: 0.510, Gen: 0.667, Rat: 0.550, Pop: 0.012, Fra: 0.000 --> Final: 0.462
  9. Unknown (Rel: 0.288 Grade: 1)
      Sem: 0.533, Gen: 0.500, Rat: 0.650, Pop: 0.026, Fra: 0.000 --> Final: 0.461
  10. iBoy (Rel: 0.296 Grade: 1)
      Sem: 0.522, Gen: 0.571, Rat: 0.600, Pop: 0.015, Fra: 0.000 --> Final: 0.460

Query: Interstellar
  1. Prometheus (Rel: 0.345 Grade: 1)
      Sem: 0.698, Gen: 0.600, Rat: 0.630, Pop: 0.030, Fra: 0.000 --> Final: 0.573
  2. On the Silver Globe (Rel: 0.261 Grade: 1)
      Sem: 0.643, Gen: 0.600, Rat: 0.850, Pop: 0.004, Fra: 0.000 --> Final: 0.561
  3. Passengers (Rel: 0.422 Grade: 2)
      Sem: 0.593, Gen: 0.800, Rat: 0.670, Pop: 0.037, Fra: 0.000 --> Final: 0.545
  4. Close Encounters of the Third Kind (Rel: 0.404 Grade: 2)
      Sem: 0.587, Gen: 0.750, Rat: 0.720, Pop: 0.019, Fra: 0.000 --> Final: 0.538
  5. The Wild Blue Yonder (Rel: 0.316 Grade: 1)
      Sem: 0.577, Gen: 0.750, Rat: 0.570, Pop: 0.002, Fra: 0.000 --> Final: 0.516
  6. Voyage to the Bottom of the Sea (Rel: 0.418 Grade: 2)
      Sem: 0.513, Gen: 1.000, Rat: 0.540, Pop: 0.006, Fra: 0.000 --> Final: 0.512
  7. Judas Kiss (Rel: 0.316 Grade: 1)
      Sem: 0.564, Gen: 0.750, Rat: 0.570, Pop: 0.002, Fra: 0.000 --> Final: 0.508
  8. Des fleurs pour Algernon (Rel: 0.310 Grade: 1)
      Sem: 0.536, Gen: 0.750, Rat: 0.710, Pop: 0.002, Fra: 0.000 --> Final: 0.505
  9. Lipton Cockton in the Shadows of Sodoma (Rel: 0.253 Grade: 1)
      Sem: 0.586, Gen: 0.600, Rat: 0.600, Pop: 0.000, Fra: 0.000 --> Final: 0.502
  10. Star Trek Beyond (Rel: 0.346 Grade: 1)
      Sem: 0.571, Gen: 0.600, Rat: 0.660, Pop: 0.038, Fra: 0.000 --> Final: 0.500

Query: The Matrix
  1. The Matrix Revolutions (Rel: 0.482 Grade: 2)
      Sem: 0.622, Gen: 0.600, Rat: 0.640, Pop: 0.028, Fra: 1.000 --> Final: 0.629
  2. The Matrix Reloaded (Rel: 0.483 Grade: 2)
      Sem: 0.591, Gen: 0.600, Rat: 0.670, Pop: 0.030, Fra: 1.000 --> Final: 0.613
  3. City Limits (Rel: 0.385 Grade: 2)
      Sem: 0.684, Gen: 1.000, Rat: 0.280, Pop: 0.001, Fra: 0.000 --> Final: 0.588
  4. Mars (Rel: 0.145 Grade: 0)
      Sem: 0.631, Gen: 1.000, Rat: 0.490, Pop: 0.000, Fra: 0.000 --> Final: 0.578
  5. Rakka (Rel: 0.413 Grade: 2)
      Sem: 0.568, Gen: 1.000, Rat: 0.740, Pop: 0.005, Fra: 0.000 --> Final: 0.565
  6. Cyberjack (Rel: 0.299 Grade: 1)
      Sem: 0.631, Gen: 0.750, Rat: 0.730, Pop: 0.000, Fra: 0.000 --> Final: 0.564
  7. 1990: The Bronx Warriors (Rel: 0.388 Grade: 2)
      Sem: 0.608, Gen: 1.000, Rat: 0.420, Pop: 0.015, Fra: 0.000 --> Final: 0.558
  8. Hands of Steel (Rel: 0.403 Grade: 2)
      Sem: 0.597, Gen: 1.000, Rat: 0.490, Pop: 0.002, Fra: 0.000 --> Final: 0.557
  9. The Gene Generation (Rel: 0.380 Grade: 2)
      Sem: 0.593, Gen: 1.000, Rat: 0.470, Pop: 0.007, Fra: 0.000 --> Final: 0.553
  10. Sons of Liberty (Rel: 0.390 Grade: 2)
      Sem: 0.604, Gen: 1.000, Rat: 0.330, Pop: 0.001, Fra: 0.000 --> Final: 0.545

Query: The Dark Knight
  1. The Dark Knight Rises (Rel: 0.658 Grade: 4)
      Sem: 0.778, Gen: 1.000, Rat: 0.760, Pop: 0.038, Fra: 1.000 --> Final: 0.795
  2. Batman Begins (Rel: 0.408 Grade: 2)
      Sem: 0.703, Gen: 0.750, Rat: 0.750, Pop: 0.052, Fra: 0.000 --> Final: 0.612
  3. Batman: Assault on Arkham (Rel: 0.306 Grade: 1)
      Sem: 0.708, Gen: 0.600, Rat: 0.730, Pop: 0.011, Fra: 0.000 --> Final: 0.588
  4. Batman: The Killing Joke (Rel: 0.329 Grade: 1)
      Sem: 0.726, Gen: 0.600, Rat: 0.620, Pop: 0.012, Fra: 0.000 --> Final: 0.588
  5. Batman: The Dark Knight Returns, Part 1 (Rel: 0.339 Grade: 1)
      Sem: 0.597, Gen: 0.200, Rat: 0.770, Pop: 0.022, Fra: 1.000 --> Final: 0.566
  6. Batman Unmasked: The Psychology of the Dark Knight (Rel: 0.163 Grade: 0)
      Sem: 0.637, Gen: 0.000, Rat: 0.800, Pop: 0.002, Fra: 1.000 --> Final: 0.562
  7. Where the Sidewalk Ends (Rel: 0.376 Grade: 2)
      Sem: 0.574, Gen: 1.000, Rat: 0.650, Pop: 0.005, Fra: 0.000 --> Final: 0.560
  8. The Seven-Ups (Rel: 0.376 Grade: 2)
      Sem: 0.560, Gen: 1.000, Rat: 0.650, Pop: 0.004, Fra: 0.000 --> Final: 0.551
  9. Death Wish (Rel: 0.414 Grade: 2)
      Sem: 0.542, Gen: 1.000, Rat: 0.700, Pop: 0.024, Fra: 0.000 --> Final: 0.547
  10. Brick Mansions (Rel: 0.363 Grade: 2)
      Sem: 0.627, Gen: 0.750, Rat: 0.570, Pop: 0.014, Fra: 0.000 --> Final: 0.546

Query: Avatar
  1. Avatar 2 (Rel: 0.601 Grade: 3)
      Sem: 0.478, Gen: 1.000, Rat: 0.000, Pop: 0.011, Fra: 1.000 --> Final: 0.537
  2. Gamera vs. Viras (Rel: 0.307 Grade: 1)
      Sem: 0.605, Gen: 0.800, Rat: 0.500, Pop: 0.002, Fra: 0.000 --> Final: 0.533
  3. The War in Space (Rel: 0.326 Grade: 1)
      Sem: 0.567, Gen: 0.800, Rat: 0.700, Pop: 0.001, Fra: 0.000 --> Final: 0.530
  4. The Nostalgist (Rel: 0.395 Grade: 2)
      Sem: 0.501, Gen: 1.000, Rat: 0.760, Pop: 0.001, Fra: 0.000 --> Final: 0.527
  5. The Flash 2 - Revenge of the Trickster (Rel: 0.350 Grade: 2)
      Sem: 0.557, Gen: 0.800, Rat: 0.660, Pop: 0.003, Fra: 0.000 --> Final: 0.520
  6. Fantastic Four (Rel: 0.479 Grade: 2)
      Sem: 0.584, Gen: 0.800, Rat: 0.440, Pop: 0.032, Fra: 0.000 --> Final: 0.516
  7. Star Trek Beyond (Rel: 0.424 Grade: 2)
      Sem: 0.546, Gen: 0.800, Rat: 0.660, Pop: 0.038, Fra: 0.000 --> Final: 0.515
  8. A Trip to the Moon (Rel: 0.414 Grade: 2)
      Sem: 0.526, Gen: 0.800, Rat: 0.790, Pop: 0.012, Fra: 0.000 --> Final: 0.515
  9. Damnation Alley (Rel: 0.316 Grade: 1)
      Sem: 0.574, Gen: 0.800, Rat: 0.500, Pop: 0.004, Fra: 0.000 --> Final: 0.515
  10. Star Wars: The Force Awakens (Rel: 0.514 Grade: 3)
      Sem: 0.475, Gen: 1.000, Rat: 0.750, Pop: 0.058, Fra: 0.000 --> Final: 0.513

Query: Toy Story
  1. Toy Story 3 (Rel: 0.658 Grade: 4)
      Sem: 0.877, Gen: 1.000, Rat: 0.760, Pop: 0.031, Fra: 1.000 --> Final: 0.854
  2. Toy Story 2 (Rel: 0.667 Grade: 4)
      Sem: 0.805, Gen: 1.000, Rat: 0.730, Pop: 0.032, Fra: 1.000 --> Final: 0.808
  3. Toy Story of Terror! (Rel: 0.584 Grade: 3)
      Sem: 0.579, Gen: 1.000, Rat: 0.730, Pop: 0.001, Fra: 1.000 --> Final: 0.670
  4. Superstar Goofy (Rel: 0.399 Grade: 2)
      Sem: 0.574, Gen: 1.000, Rat: 0.480, Pop: 0.001, Fra: 0.000 --> Final: 0.542
  5. The Bear That Wasn't (Rel: 0.370 Grade: 2)
      Sem: 0.520, Gen: 1.000, Rat: 0.700, Pop: 0.000, Fra: 0.000 --> Final: 0.532
  6. Mickey's Once Upon a Christmas (Rel: 0.388 Grade: 2)
      Sem: 0.516, Gen: 1.000, Rat: 0.670, Pop: 0.012, Fra: 0.000 --> Final: 0.527
  7. An Extremely Goofy Movie (Rel: 0.395 Grade: 2)
      Sem: 0.525, Gen: 1.000, Rat: 0.600, Pop: 0.019, Fra: 0.000 --> Final: 0.526
  8. One Froggy Evening (Rel: 0.387 Grade: 2)
      Sem: 0.496, Gen: 1.000, Rat: 0.780, Pop: 0.004, Fra: 0.000 --> Final: 0.526
  9. Scooby-Doo! Camp Scare (Rel: 0.390 Grade: 2)
      Sem: 0.489, Gen: 1.000, Rat: 0.730, Pop: 0.004, Fra: 0.000 --> Final: 0.517
  10. Happiness Is a Warm Blanket, Charlie Brown (Rel: 0.398 Grade: 2)
      Sem: 0.496, Gen: 1.000, Rat: 0.650, Pop: 0.003, Fra: 0.000 --> Final: 0.512

Query: The Lord of the Rings
  1. The Lord of the Rings: The Fellowship of the Ring (Rel: 0.440 Grade: 2)
      Sem: 0.616, Gen: 0.400, Rat: 0.800, Pop: 0.059, Fra: 1.000 --> Final: 0.613
  2. The Lord of the Rings: The Two Towers (Rel: 0.463 Grade: 2)
      Sem: 0.545, Gen: 0.400, Rat: 0.800, Pop: 0.054, Fra: 1.000 --> Final: 0.570
  3. The Lord of the Rings: The Return of the King (Rel: 0.448 Grade: 2)
      Sem: 0.507, Gen: 0.400, Rat: 0.810, Pop: 0.054, Fra: 1.000 --> Final: 0.548
  4. The Adventures of Prince Achmed (Rel: 0.391 Grade: 2)
      Sem: 0.512, Gen: 1.000, Rat: 0.780, Pop: 0.003, Fra: 0.000 --> Final: 0.535
  5. The Return of the King (Rel: 0.177 Grade: 0)
      Sem: 0.699, Gen: 0.400, Rat: 0.510, Pop: 0.003, Fra: 0.000 --> Final: 0.531
  6. Children Who Chase Lost Voices (Rel: 0.408 Grade: 2)
      Sem: 0.499, Gen: 1.000, Rat: 0.710, Pop: 0.010, Fra: 0.000 --> Final: 0.521
  7. The Hunt for Gollum (Rel: 0.194 Grade: 0)
      Sem: 0.637, Gen: 0.400, Rat: 0.630, Pop: 0.006, Fra: 0.000 --> Final: 0.506
  8. The Beautiful Story (Rel: 0.282 Grade: 1)
      Sem: 0.509, Gen: 0.750, Rat: 0.800, Pop: 0.001, Fra: 0.000 --> Final: 0.498
  9. The Hobbit: The Desolation of Smaug (Rel: 0.337 Grade: 1)
      Sem: 0.541, Gen: 0.500, Rat: 0.760, Pop: 0.038, Fra: 0.000 --> Final: 0.478
  10. Spring and Chaos (Rel: 0.280 Grade: 1)
      Sem: 0.473, Gen: 0.750, Rat: 0.800, Pop: 0.000, Fra: 0.000 --> Final: 0.476

Query: Titanic
  1. Titanic 2 (Rel: 0.245 Grade: 1)
      Sem: 0.550, Gen: 0.200, Rat: 0.340, Pop: 0.008, Fra: 1.000 --> Final: 0.494
  2. Raise the Titanic (Rel: 0.346 Grade: 1)
      Sem: 0.443, Gen: 0.500, Rat: 0.520, Pop: 0.004, Fra: 1.000 --> Final: 0.493
  3. Titanic at 100: Mystery Solved (Rel: 0.157 Grade: 0)
      Sem: 0.511, Gen: 0.000, Rat: 0.800, Pop: 0.000, Fra: 1.000 --> Final: 0.487
  4. The Chambermaid on the Titanic (Rel: 0.273 Grade: 1)
      Sem: 0.464, Gen: 0.333, Rat: 0.530, Pop: 0.001, Fra: 1.000 --> Final: 0.482
  5. Titanica (Rel: 0.171 Grade: 0)
      Sem: 0.522, Gen: 0.000, Rat: 0.600, Pop: 0.002, Fra: 1.000 --> Final: 0.473
  6. Living 'til the End (Rel: 0.267 Grade: 1)
      Sem: 0.468, Gen: 0.667, Rat: 0.700, Pop: 0.000, Fra: 0.000 --> Final: 0.451
  7. Titanic: The Final Word with James Cameron (Rel: 0.159 Grade: 0)
      Sem: 0.462, Gen: 0.000, Rat: 0.660, Pop: 0.001, Fra: 1.000 --> Final: 0.443
  8. Niagara (Rel: 0.203 Grade: 1)
      Sem: 0.493, Gen: 0.500, Rat: 0.670, Pop: 0.027, Fra: 0.000 --> Final: 0.439
  9. Swept from the Sea (Rel: 0.261 Grade: 1)
      Sem: 0.449, Gen: 0.667, Rat: 0.650, Pop: 0.018, Fra: 0.000 --> Final: 0.435
  10. Closing the Ring (Rel: 0.253 Grade: 1)
      Sem: 0.452, Gen: 0.667, Rat: 0.630, Pop: 0.019, Fra: 0.000 --> Final: 0.435


