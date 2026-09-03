# -*- coding: utf-8 -*-
"""
=============================================================
  Movie Recommendation System  -  Step 1: Dataset Analysis
=============================================================
Reads movies.csv safely and prints a comprehensive report
covering shape, column types, missing values, duplicates,
sample rows, and recommendations for the first model.

DO NOT modify the original dataset - this script is read-only.
"""

import os
import sys
import pandas as pd

# ------------------------------------------------------------------ #
#  Paths                                                               #
# ------------------------------------------------------------------ #

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

# The file on disk has a double extension; try both names gracefully
CSV_CANDIDATES = [
    os.path.join(PROJECT_ROOT, "movies.csv.csv"),
    os.path.join(PROJECT_ROOT, "movies.csv"),
]

CSV_PATH = None
for candidate in CSV_CANDIDATES:
    if os.path.exists(candidate):
        CSV_PATH = candidate
        break

if CSV_PATH is None:
    print("[ERROR] Could not locate movies.csv or movies.csv.csv in the project root.")
    sys.exit(1)

# ------------------------------------------------------------------ #
#  Helpers                                                             #
# ------------------------------------------------------------------ #

SEP = "=" * 70

def section(title):
    print("\n" + SEP)
    print("  " + title)
    print(SEP)

# ------------------------------------------------------------------ #
#  Load                                                                #
# ------------------------------------------------------------------ #

section("1. LOADING DATASET")
print("   File  : " + CSV_PATH)
print("   Size  : {:.2f} MB".format(os.path.getsize(CSV_PATH) / 1_048_576))

try:
    df = pd.read_csv(CSV_PATH, low_memory=False)
    print("   Status: Loaded successfully [OK]")
except Exception as exc:
    print("[ERROR] Failed to load CSV: " + str(exc))
    sys.exit(1)

# ------------------------------------------------------------------ #
#  Shape                                                               #
# ------------------------------------------------------------------ #

section("2. DATASET SHAPE (rows x columns)")
rows, cols = df.shape
print("   Rows    : {:,}".format(rows))
print("   Columns : {}".format(cols))

# ------------------------------------------------------------------ #
#  Column Names                                                        #
# ------------------------------------------------------------------ #

section("3. ALL COLUMN NAMES")
for i, col in enumerate(df.columns, 1):
    print("   {:>3}. {}".format(i, col))

# ------------------------------------------------------------------ #
#  Data Types                                                          #
# ------------------------------------------------------------------ #

section("4. DATA TYPES PER COLUMN")
print("   {:<35} {}".format("Column", "Dtype"))
print("   {:<35} {}".format("-" * 35, "-" * 15))
for col, dtype in df.dtypes.items():
    print("   {:<35} {}".format(col, dtype))

# ------------------------------------------------------------------ #
#  Missing Values                                                      #
# ------------------------------------------------------------------ #

section("5. MISSING VALUES")
missing = df.isnull().sum()
missing_pct = (missing / len(df) * 100).round(2)
missing_df = pd.DataFrame({
    "Missing Count": missing,
    "Missing %": missing_pct
}).sort_values("Missing %", ascending=False)

has_missing = missing_df[missing_df["Missing Count"] > 0]
if has_missing.empty:
    print("   No missing values found in any column! [OK]")
else:
    print("   {:<35} {:>14} {:>10}".format("Column", "Missing Count", "Missing %"))
    print("   {:<35} {:>14} {:>10}".format("-" * 35, "-" * 14, "-" * 10))
    for col, row in has_missing.iterrows():
        flag = "  <- HIGH" if row["Missing %"] > 40 else ""
        print("   {:<35} {:>14,} {:>9.2f}%{}".format(
            col, int(row["Missing Count"]), row["Missing %"], flag))

print("\n   Columns with zero missing values : {} / {}".format(
    (missing == 0).sum(), cols))

# ------------------------------------------------------------------ #
#  Duplicates                                                          #
# ------------------------------------------------------------------ #

section("6. DUPLICATE ROWS")
dup_count = df.duplicated().sum()
print("   Exact duplicate rows : {:,}".format(dup_count))

if "id" in df.columns:
    id_dups = df["id"].duplicated().sum()
    print("   Duplicate movie IDs  : {:,}".format(id_dups))

if "title" in df.columns:
    title_dups = df["title"].duplicated(keep=False).sum()
    print("   Rows sharing a title : {:,}  (includes remakes / different releases)".format(
        title_dups))

# ------------------------------------------------------------------ #
#  Sample Records                                                      #
# ------------------------------------------------------------------ #

section("7. 5 SAMPLE RECORDS  (transposed for readability)")
sample = df.sample(5, random_state=42) if len(df) >= 5 else df
pd.set_option("display.max_colwidth", 60)
pd.set_option("display.width", 120)
print(sample.T.to_string())

# ------------------------------------------------------------------ #
#  Column Usefulness Audit                                             #
# ------------------------------------------------------------------ #

section("8. COLUMN USEFULNESS AUDIT FOR RECOMMENDATION SYSTEM")

PRIORITY_COLUMNS = {
    "title":        "Movie name - mandatory for display & lookup",
    "genres":       "Genre tags - key content signal",
    "overview":     "Plot synopsis - rich text for TF-IDF / embeddings",
    "vote_average": "User rating - quality filter & score weighting",
    "vote_count":   "Number of votes - confidence / popularity weight",
    "popularity":   "TMDB popularity score - trend signal",
    "release_date": "Release year / era - temporal filtering",
    "poster_path":  "Poster image URL - UI display",
    "id":           "Unique movie identifier",
    "cast":         "Actor names - collaborative content signal",
    "crew":         "Director & crew - content signal",
    "keywords":     "Curated keywords - precision content tags",
    "tagline":      "Short marketing phrase - supplementary text",
    "runtime":      "Film duration - preference filtering",
    "original_language": "Language - diversity / language filter",
    "production_companies": "Studio - additional content signal",
    "spoken_languages":     "Language info",
    "adult":        "Adult content flag - safety filter",
    "status":       "Released / Rumored etc - quality filter",
}

found, missing_priority = [], []
for col, desc in PRIORITY_COLUMNS.items():
    if col in df.columns:
        null_pct = df[col].isnull().mean() * 100
        found.append((col, desc, null_pct))
    else:
        missing_priority.append(col)

print("\n   {:<28} {:>10}   {}".format("Column", "Missing %", "Description"))
print("   {:<28} {:>10}   {}".format("-" * 28, "-" * 10, "-" * 40))
for col, desc, np_ in found:
    status = "[OK]" if np_ < 30 else "[!! HIGH MISS]"
    print("   {:<28} {:>9.1f}%  [{}] {}".format(col, np_, status, desc))

if missing_priority:
    print("\n   Columns NOT present in this dataset: {}".format(missing_priority))

# ------------------------------------------------------------------ #
#  Recommendations                                                     #
# ------------------------------------------------------------------ #

section("9. RECOMMENDED COLUMNS FOR FIRST RECOMMENDATION MODEL")
print("""
   CONTENT-BASED FILTERING MODEL  (Phase 1)
   -----------------------------------------
   Primary Feature Columns (build the 'tags' corpus):
     * title          - display & search index
     * genres         - strongest content signal
     * overview       - TF-IDF / sentence-embedding source
     * keywords       - curated content descriptors (if present)
     * cast / crew    - top-3 actors + director (if present)

   Metadata / Filtering Columns:
     * vote_average   - weight / rank results; filter low-quality
     * vote_count     - minimum vote threshold (e.g. > 50 votes)
     * popularity     - tie-breaking / trending boost
     * release_date   - year extraction for era filtering
     * poster_path    - Streamlit UI image display

   Columns to SKIP in Phase 1:
     * adult, status, spoken_languages, production_companies
       (low signal or high missing rate)
     * Any column with > 70 % missing values

   Suggested Pipeline:
     1. Drop rows where BOTH overview AND genres are null
     2. Fill remaining nulls with empty strings
     3. Combine genres + overview + keywords + cast + director
        into a single 'tags' text column
     4. TF-IDF vectorise 'tags'  (max_features=10,000)
     5. Cosine similarity matrix  ->  top-N nearest neighbours
""")

section("ANALYSIS COMPLETE")
print("   Dataset has {:,} movies across {} columns.".format(rows, cols))
print("   Script finished without modifying the source file.\n")
