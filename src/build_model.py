# -*- coding: utf-8 -*-
"""
=============================================================
  Movie Recommendation System  -  Step 3: Build Model
=============================================================
Loads dataset/movies_cleaned.csv, converts the 'tags' column
into a sparse TF-IDF matrix, fits a NearestNeighbors model
on it, and serialises all artefacts into models/.

Memory strategy:
  - The TF-IDF matrix stays sparse (CSR format).
  - NearestNeighbors with metric="cosine" + algorithm="brute"
    works directly on sparse matrices and computes cosine
    distances row-by-row at query time.
  - NO full N x N similarity matrix is ever computed or stored.

Outputs:
  models/tfidf_vectorizer.pkl
  models/movie_neighbors.pkl
  models/movie_metadata.pkl
  src/model_report.txt
"""

import os
import sys
import re
import time
import textwrap
import pickle

import pandas as pd
import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

# ------------------------------------------------------------------ #
#  Paths                                                               #
# ------------------------------------------------------------------ #
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

CLEANED_CSV  = os.path.join(PROJECT_ROOT, "dataset", "movies_cleaned.csv")
MODELS_DIR   = os.path.join(PROJECT_ROOT, "models")
REPORT_PATH  = os.path.join(SCRIPT_DIR,  "model_report.txt")

TFIDF_PATH   = os.path.join(MODELS_DIR, "tfidf_vectorizer.pkl")
NN_PATH      = os.path.join(MODELS_DIR, "movie_neighbors.pkl")
META_PATH    = os.path.join(MODELS_DIR, "movie_metadata.pkl")

os.makedirs(MODELS_DIR, exist_ok=True)

SEP = "=" * 70

def section(title):
    print("\n" + SEP)
    print("  " + title)
    print(SEP)

def fmt_bytes(nbytes):
    for unit in ("B", "KB", "MB", "GB"):
        if nbytes < 1024:
            return "{:.2f} {}".format(nbytes, unit)
        nbytes /= 1024
    return "{:.2f} TB".format(nbytes)

# ------------------------------------------------------------------ #
#  1. Load cleaned dataset                                             #
# ------------------------------------------------------------------ #
section("1. LOADING CLEANED DATASET")

if not os.path.exists(CLEANED_CSV):
    print("[ERROR] {} not found. Run preprocess_data.py first.".format(CLEANED_CSV))
    sys.exit(1)

df = pd.read_csv(CLEANED_CSV, low_memory=False)
print("   Loaded : " + CLEANED_CSV)
print("   Shape  : {:,} rows x {} columns".format(*df.shape))

# ------------------------------------------------------------------ #
#  2. Prepare the 'tags' text column                                   #
# ------------------------------------------------------------------ #
section("2. PREPARING 'TAGS' TEXT COLUMN")

# Ensure tags column exists
if "tags" not in df.columns:
    print("[ERROR] 'tags' column missing. Re-run preprocess_data.py.")
    sys.exit(1)

def clean_tags(text):
    """Lowercase, normalise whitespace, strip."""
    if pd.isna(text):
        return ""
    s = str(text).lower().strip()
    s = re.sub(r"\s+", " ", s)
    return s

df["tags"] = df["tags"].apply(clean_tags)

# Drop any remaining rows where tags are completely empty
before = len(df)
df = df[df["tags"] != ""].reset_index(drop=True)
dropped = before - len(df)
print("   Tags normalised (lowercase, whitespace cleaned).")
print("   Rows with empty tags after cleaning : {:,}".format(dropped))
print("   Movies available for modelling      : {:,}".format(len(df)))

# ------------------------------------------------------------------ #
#  3. TF-IDF Vectorisation                                             #
# ------------------------------------------------------------------ #
section("3. TF-IDF VECTORISATION")

TFIDF_CONFIG = dict(
    stop_words="english",
    ngram_range=(1, 2),
    min_df=2,
    max_features=20_000,
    sublinear_tf=True,
)

print("   Configuration:")
for k, v in TFIDF_CONFIG.items():
    print("     {:<15} : {}".format(k, v))

t0 = time.time()
vectorizer = TfidfVectorizer(**TFIDF_CONFIG)
tfidf_matrix = vectorizer.fit_transform(df["tags"])
elapsed = time.time() - t0

n_movies, n_features = tfidf_matrix.shape
nnz        = tfidf_matrix.nnz
max_nnz    = n_movies * n_features
density    = nnz / max_nnz * 100
mem_sparse = (
    tfidf_matrix.data.nbytes
    + tfidf_matrix.indices.nbytes
    + tfidf_matrix.indptr.nbytes
)
mem_dense  = n_movies * n_features * 8   # float64 bytes for a hypothetical dense matrix

print()
print("   Results:")
print("   Number of movies          : {:,}".format(n_movies))
print("   Number of TF-IDF features : {:,}".format(n_features))
print("   Sparse matrix shape       : {} x {}".format(n_movies, n_features))
print("   Non-zero elements         : {:,}".format(nnz))
print("   Matrix density            : {:.4f}%".format(density))
print("   Sparse matrix memory      : {}".format(fmt_bytes(mem_sparse)))
print("   Dense equivalent would be : {}".format(fmt_bytes(mem_dense)))
print("   Vectorisation time        : {:.2f}s".format(elapsed))

# ------------------------------------------------------------------ #
#  4. Build NearestNeighbors model                                     #
# ------------------------------------------------------------------ #
section("4. BUILDING NEARESTNEIGHBORS MODEL")

NN_CONFIG = dict(
    metric="cosine",
    algorithm="brute",
    n_neighbors=11,     # 11 so we can drop self-match and return 10
)

print("   Configuration:")
for k, v in NN_CONFIG.items():
    print("     {:<12} : {}".format(k, v))

t0 = time.time()
nn_model = NearestNeighbors(**NN_CONFIG)
nn_model.fit(tfidf_matrix)
elapsed_fit = time.time() - t0

print()
print("   NearestNeighbors fitted successfully.")
print("   Fit time : {:.2f}s".format(elapsed_fit))
print()
print("   Design note:")
print("     algorithm='brute'  computes distances on-the-fly at query time.")
print("     No N x N similarity matrix is ever computed or held in memory.")
print("     Memory peak = O(N * F) for the sparse matrix, NOT O(N^2).")
dense_sim_size = fmt_bytes(n_movies * n_movies * 8)
print("     A full dense cosine matrix would require ~{}.".format(dense_sim_size))
print("     The sparse approach uses only {}.".format(fmt_bytes(mem_sparse)))

# ------------------------------------------------------------------ #
#  5. Build and save movie metadata                                    #
# ------------------------------------------------------------------ #
section("5. BUILDING MOVIE METADATA")

META_COLS = [
    "id", "title", "genres", "overview",
    "popularity", "vote_average", "vote_count",
    "release_date", "poster_path", "imdb_id",
]
existing_meta_cols = [c for c in META_COLS if c in df.columns]
movie_metadata = df[existing_meta_cols].reset_index(drop=True)

print("   Metadata columns   : " + ", ".join(existing_meta_cols))
print("   Metadata shape     : {:,} rows x {} columns".format(*movie_metadata.shape))

# ------------------------------------------------------------------ #
#  6. Save all artefacts                                               #
# ------------------------------------------------------------------ #
section("6. SAVING ARTEFACTS TO models/")

# 6a. TF-IDF vectorizer
joblib.dump(vectorizer, TFIDF_PATH)
tfidf_size = os.path.getsize(TFIDF_PATH)
print("   Saved tfidf_vectorizer.pkl  ({})".format(fmt_bytes(tfidf_size)))

# 6b. NearestNeighbors model
joblib.dump(nn_model, NN_PATH)
nn_size = os.path.getsize(NN_PATH)
print("   Saved movie_neighbors.pkl   ({})".format(fmt_bytes(nn_size)))

# 6c. Metadata DataFrame
joblib.dump(movie_metadata, META_PATH)
meta_size = os.path.getsize(META_PATH)
print("   Saved movie_metadata.pkl    ({})".format(fmt_bytes(meta_size)))

# ------------------------------------------------------------------ #
#  7. Verification: all files exist                                    #
# ------------------------------------------------------------------ #
section("7. FILE VERIFICATION")

files_to_check = {
    "models/tfidf_vectorizer.pkl": TFIDF_PATH,
    "models/movie_neighbors.pkl" : NN_PATH,
    "models/movie_metadata.pkl"  : META_PATH,
}
all_exist = True
for label, path in files_to_check.items():
    exists = os.path.exists(path)
    status = "[PASS]" if exists else "[FAIL]"
    if not exists:
        all_exist = False
    print("   {} {}  ({})".format(
        status, label,
        fmt_bytes(os.path.getsize(path)) if exists else "MISSING"
    ))

print()
print("   All files present : {}".format("YES" if all_exist else "NO - CHECK ERRORS"))

# ------------------------------------------------------------------ #
#  8. Sanity test: 10 nearest neighbours for a test movie             #
# ------------------------------------------------------------------ #
section("8. SANITY TEST - NEAREST NEIGHBOR LOOKUP")

# Prefer "Toy Story"; fall back to first movie in dataset
TEST_TITLES = ["Toy Story", "The Dark Knight", "Inception", "Avatar"]
test_title  = None
test_idx    = None

for candidate in TEST_TITLES:
    matches = df[df["title"].str.lower() == candidate.lower()]
    if not matches.empty:
        test_title = candidate
        test_idx   = matches.index[0]
        break

if test_idx is None:
    test_title = df["title"].iloc[0]
    test_idx   = 0

print()
print("   Input movie : {} (dataset index {})".format(test_title, test_idx))
print()

# Get the TF-IDF vector for the test movie (sparse row)
test_vector     = tfidf_matrix[test_idx]

# Query the model (n_neighbors=11 to include the movie itself)
t0 = time.time()
distances, indices = nn_model.kneighbors(test_vector, n_neighbors=11)
query_time = time.time() - t0

distances = distances.flatten()
indices   = indices.flatten()

print("   Query time : {:.4f}s".format(query_time))
print()
print("   Recommended movies:")
print("   {:<4} {:<45} {:<20} {:>10} {:>8}".format(
    "#", "Title", "Genres", "Vote Avg", "Cos Dist"))
print("   " + "-" * 95)

rec_count = 0
for dist, idx in zip(distances, indices):
    if idx == test_idx:        # skip self-match
        continue
    row = movie_metadata.iloc[idx]
    genre_str = str(row.get("genres", ""))[:18]
    vote_avg  = row.get("vote_average", "N/A")
    rec_count += 1
    print("   {:<4} {:<45} {:<20} {:>10} {:>8.4f}".format(
        rec_count,
        str(row["title"])[:44],
        genre_str,
        str(vote_avg),
        dist
    ))
    if rec_count >= 10:
        break

# ------------------------------------------------------------------ #
#  9. Write model report                                               #
# ------------------------------------------------------------------ #
section("9. WRITING MODEL REPORT")

report = textwrap.dedent("""\
======================================================================
  MOVIE RECOMMENDATION SYSTEM - MODEL REPORT
  Generated : 2026-08-31
======================================================================

----------------------------------------------------------------------
 DATASET
----------------------------------------------------------------------
  Source          : dataset/movies_cleaned.csv
  Number of movies: {n_movies:,}
  Text feature    : 'tags' column (genres + overview, lowercased)

----------------------------------------------------------------------
 TF-IDF CONFIGURATION
----------------------------------------------------------------------
  stop_words  : english
  ngram_range : (1, 2)   -- unigrams and bigrams
  min_df      : 2        -- ignore terms in fewer than 2 documents
  max_features: 20,000   -- vocabulary capped for memory efficiency
  sublinear_tf: True     -- apply log(1+tf) to dampen term frequency

  Number of features generated : {n_features:,}
  TF-IDF matrix shape          : {n_movies:,} x {n_features:,}
  Non-zero elements             : {nnz:,}
  Matrix density                : {density:.4f}%
  Sparse matrix memory          : {mem_sparse}
  Dense equivalent memory       : {mem_dense}

----------------------------------------------------------------------
 WHY SPARSE REPRESENTATION
----------------------------------------------------------------------
  Most movies share very few of the 20,000 vocabulary terms.
  Storing only non-zero values (CSR format) cuts memory from
  {mem_dense} (dense float64) down to {mem_sparse}.
  The TF-IDF matrix is {sparsity_ratio:.0f}x more memory-efficient in
  sparse form.

----------------------------------------------------------------------
 WHY NO FULL COSINE SIMILARITY MATRIX
----------------------------------------------------------------------
  A full N x N float64 similarity matrix for {n_movies:,} movies
  would require approximately {dense_sim_size}, which is
  infeasible for most laptops and unnecessary.

  Instead, NearestNeighbors (brute-force, cosine metric) computes
  distances only at query time for a single input vector.
  Memory cost is O(N * F) for the sparse index, NOT O(N^2).

----------------------------------------------------------------------
 NEARESTNEIGHBORS CONFIGURATION
----------------------------------------------------------------------
  metric      : cosine
  algorithm   : brute  -- exact, no approximation; works on sparse
  n_neighbors : 11     -- 11 retrieved; self-match dropped -> 10 recs

----------------------------------------------------------------------
 SAVED MODEL FILES
----------------------------------------------------------------------
  models/tfidf_vectorizer.pkl  -- fitted TfidfVectorizer ({tfidf_size})
  models/movie_neighbors.pkl   -- fitted NearestNeighbors ({nn_size})
  models/movie_metadata.pkl    -- DataFrame with movie metadata ({meta_size})

----------------------------------------------------------------------
 METADATA COLUMNS PRESERVED
----------------------------------------------------------------------
  id, title, genres, overview, popularity,
  vote_average, vote_count, release_date, poster_path, imdb_id

======================================================================
 END OF REPORT
======================================================================
""").format(
    n_movies=n_movies,
    n_features=n_features,
    nnz=nnz,
    density=density,
    mem_sparse=fmt_bytes(mem_sparse),
    mem_dense=fmt_bytes(mem_dense),
    dense_sim_size=dense_sim_size,
    sparsity_ratio=mem_dense / mem_sparse if mem_sparse > 0 else 0,
    tfidf_size=fmt_bytes(tfidf_size),
    nn_size=fmt_bytes(nn_size),
    meta_size=fmt_bytes(meta_size),
)

with open(REPORT_PATH, "w", encoding="utf-8") as f:
    f.write(report)
print("   Report saved : " + REPORT_PATH)

# ------------------------------------------------------------------ #
#  10. Final summary                                                   #
# ------------------------------------------------------------------ #
section("BUILD COMPLETE - SUMMARY")
print("   Movies used         : {:,}".format(n_movies))
print("   TF-IDF features     : {:,}".format(n_features))
print("   Matrix shape        : {:,} x {:,}".format(n_movies, n_features))
print("   Matrix is sparse    : YES (CSR format, density {:.4f}%)".format(density))
print("   Dense matrix built  : NO  (never computed or stored)")
print("   Dense would be      : {}".format(fmt_bytes(mem_dense)))
print("   Sparse uses only    : {}".format(fmt_bytes(mem_sparse)))
print()
print("   Saved artefacts:")
print("     " + TFIDF_PATH)
print("     " + NN_PATH)
print("     " + META_PATH)
print()
print("   Ready for Step 4 -> Recommendation engine + Streamlit app.\n")
