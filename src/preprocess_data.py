# -*- coding: utf-8 -*-
"""
=============================================================
  Movie Recommendation System  -  Step 2: Preprocessing
=============================================================
Cleans and feature-engineers movies.csv into movies_cleaned.csv.

- Removes exact duplicate rows
- Handles duplicate movie IDs (keeps most-informative record)
- Filters rows where neither genres nor overview is available
- Parses JSON-like genres into plain names
- Cleans overview and title
- Creates 'tags' column = genres + overview
- Converts numeric / date columns
- Preserves all useful metadata
- Does NOT modify the original movies.csv
- Writes dataset/movies_cleaned.csv
- Writes src/preprocessing_report.txt
"""

import os
import sys
import ast
import re
import textwrap
import pandas as pd

# ------------------------------------------------------------------ #
#  Paths                                                               #
# ------------------------------------------------------------------ #
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

CSV_CANDIDATES = [
    os.path.join(PROJECT_ROOT, "movies.csv.csv"),
    os.path.join(PROJECT_ROOT, "movies.csv"),
]
CSV_PATH = next((p for p in CSV_CANDIDATES if os.path.exists(p)), None)
if CSV_PATH is None:
    print("[ERROR] Cannot find movies.csv in the project root.")
    sys.exit(1)

CLEANED_DIR  = os.path.join(PROJECT_ROOT, "dataset")
CLEANED_PATH = os.path.join(CLEANED_DIR, "movies_cleaned.csv")
REPORT_PATH  = os.path.join(SCRIPT_DIR,  "preprocessing_report.txt")

os.makedirs(CLEANED_DIR, exist_ok=True)

SEP = "=" * 70

def section(title):
    print("\n" + SEP)
    print("  " + title)
    print(SEP)

# ------------------------------------------------------------------ #
#  1. Load                                                             #
# ------------------------------------------------------------------ #
section("1. LOADING RAW DATA")
print("   Source : " + CSV_PATH)
df_raw = pd.read_csv(CSV_PATH, low_memory=False)
original_row_count = len(df_raw)
print("   Rows   : {:,}".format(original_row_count))
print("   Cols   : {}".format(df_raw.shape[1]))

# Work on a copy so the original DataFrame is untouched in memory.
df = df_raw.copy()

# ------------------------------------------------------------------ #
#  2. Remove exact duplicate rows                                      #
# ------------------------------------------------------------------ #
section("2. REMOVING EXACT DUPLICATE ROWS")
before = len(df)
df = df.drop_duplicates()
exact_dups_removed = before - len(df)
print("   Exact duplicates removed : {:,}".format(exact_dups_removed))
print("   Rows remaining           : {:,}".format(len(df)))

# ------------------------------------------------------------------ #
#  3. Handle duplicate movie IDs                                       #
# ------------------------------------------------------------------ #
section("3. HANDLING DUPLICATE MOVIE IDs")

# Strategy:
#   Score each row by how much useful text data it has.
#   For a given id, keep the row with the highest score.
#   Score = has_overview (2 pts) + has_genres (2 pts)
#           + has_title (1 pt) + has_tagline (1 pt)
#   Ties are broken by the row that appears first.

def _score(row):
    s = 0
    v = str(row.get("overview", "") or "").strip()
    if v and v.lower() not in ("nan", "none", ""):
        s += 2
    v = str(row.get("genres", "") or "").strip()
    if v and v not in ("[]", "nan", "none", ""):
        s += 2
    v = str(row.get("title", "") or "").strip()
    if v and v.lower() not in ("nan", "none", ""):
        s += 1
    v = str(row.get("tagline", "") or "").strip()
    if v and v.lower() not in ("nan", "none", ""):
        s += 1
    return s

before = len(df)
df["_id_score"] = df.apply(_score, axis=1)
# Sort so highest score comes first, then keep first occurrence per id
df = df.sort_values("_id_score", ascending=False)
df = df.drop_duplicates(subset=["id"], keep="first")
df = df.drop(columns=["_id_score"])
# Restore original index order
df = df.sort_index()
dup_ids_removed = before - len(df)
print("   Duplicate IDs removed (kept best record) : {:,}".format(dup_ids_removed))
print("   Rows remaining                            : {:,}".format(len(df)))
print()
print("   Strategy: For each duplicate id, the row with the most text")
print("   content (overview + genres + title + tagline) was retained.")
print("   Rows with equal scores kept the earlier original row.")

# ------------------------------------------------------------------ #
#  4. Remove rows where BOTH genres and overview are missing           #
# ------------------------------------------------------------------ #
section("4. REMOVING ROWS WITH NO CONTENT")

def _is_empty(val):
    """Return True if value is NaN / None / empty string / empty list '[]'."""
    if pd.isna(val):
        return True
    s = str(val).strip()
    return s in ("", "nan", "none", "None", "[]")

mask_no_genres   = df["genres"].apply(_is_empty)
mask_no_overview = df["overview"].apply(_is_empty)
mask_remove      = mask_no_genres & mask_no_overview

no_content_removed = mask_remove.sum()
df = df[~mask_remove].copy()

# Also require a non-empty title
before = len(df)
df = df[~df["title"].apply(_is_empty)].copy()
no_title_removed = before - len(df)

print("   Rows with both genres AND overview missing : {:,}".format(no_content_removed))
print("   Rows with no valid title                   : {:,}".format(no_title_removed))
print("   Rows remaining                             : {:,}".format(len(df)))

# ------------------------------------------------------------------ #
#  5. Parse genres JSON -> plain names                                 #
# ------------------------------------------------------------------ #
section("5. PARSING GENRES COLUMN")

def parse_genres(raw):
    """
    Accepts strings like:
        [{"id": 18, "name": "Drama"}, {"id": 35, "name": "Comedy"}]
    or already-clean strings or NaN.
    Returns a space-separated string of genre names, e.g. "Drama Comedy".
    """
    if _is_empty(raw):
        return ""
    text = str(raw).strip()
    # Try ast.literal_eval first (handles Python-like dicts safely)
    try:
        parsed = ast.literal_eval(text)
        if isinstance(parsed, list):
            names = [item["name"] for item in parsed if isinstance(item, dict) and "name" in item]
            return " ".join(names)
    except (ValueError, SyntaxError):
        pass
    # Fallback: regex extraction of 'name' values
    names = re.findall(r"'name'\s*:\s*'([^']+)'|\"name\"\s*:\s*\"([^\"]+)\"", text)
    flat  = [a or b for a, b in names]
    return " ".join(flat)

df["genres"] = df["genres"].apply(parse_genres)
sample_genres = df[df["genres"] != ""]["genres"].head(3).tolist()
print("   Genre parsing complete.")
print("   Sample parsed values:")
for g in sample_genres:
    print("     - " + g)

# ------------------------------------------------------------------ #
#  6. Clean overview                                                   #
# ------------------------------------------------------------------ #
section("6. CLEANING OVERVIEW")
df["overview"] = (
    df["overview"]
    .fillna("")
    .astype(str)
    .str.strip()
    .replace(r"\s+", " ", regex=True)
)
# Replace literal 'nan' strings left over from fillna+astype
df["overview"] = df["overview"].replace({"nan": "", "none": "", "None": ""})
print("   Overview cleaned (whitespace normalised, NaN -> empty string).")

# ------------------------------------------------------------------ #
#  7. Clean title                                                      #
# ------------------------------------------------------------------ #
section("7. CLEANING TITLE")
df["title"] = df["title"].astype(str).str.strip()
print("   Title cleaned (leading/trailing whitespace removed).")

# ------------------------------------------------------------------ #
#  8. Create 'tags' column                                             #
# ------------------------------------------------------------------ #
section("8. BUILDING 'TAGS' COLUMN")
# tags = genres (space-separated names) + space + overview
df["tags"] = (df["genres"] + " " + df["overview"]).str.strip()
# Collapse any double spaces
df["tags"] = df["tags"].replace(r"\s+", " ", regex=True)

non_empty_tags = (df["tags"] != "").sum()
print("   'tags' column created.")
print("   Rows with non-empty tags : {:,} / {:,}".format(non_empty_tags, len(df)))

# ------------------------------------------------------------------ #
#  9. Numeric & date conversions                                       #
# ------------------------------------------------------------------ #
section("9. NUMERIC & DATE CONVERSIONS")

df["popularity"]    = pd.to_numeric(df["popularity"],    errors="coerce")
df["vote_average"]  = pd.to_numeric(df["vote_average"],  errors="coerce")
df["vote_count"]    = pd.to_numeric(df["vote_count"],    errors="coerce")
df["release_date"]  = pd.to_datetime(df["release_date"], errors="coerce")

print("   popularity   -> float64  (invalid values coerced to NaN)")
print("   vote_average -> float64")
print("   vote_count   -> float64")
print("   release_date -> datetime64 (invalid values coerced to NaT)")
print("   NaN popularity  : {:,}".format(df["popularity"].isna().sum()))
print("   NaN vote_average: {:,}".format(df["vote_average"].isna().sum()))
print("   NaN vote_count  : {:,}".format(df["vote_count"].isna().sum()))
print("   NaT release_date: {:,}".format(df["release_date"].isna().sum()))

# ------------------------------------------------------------------ #
#  10. Select and order final columns                                  #
# ------------------------------------------------------------------ #
section("10. SELECTING FINAL COLUMNS")

METADATA_COLS = [
    "id", "imdb_id", "title", "original_title", "original_language",
    "release_date", "popularity", "vote_average", "vote_count",
    "poster_path", "runtime", "tagline",
]
FEATURE_COLS  = ["genres", "overview", "tags"]     # recommendation features

# Keep only columns that actually exist in the dataframe
all_keep = [c for c in METADATA_COLS + FEATURE_COLS if c in df.columns]
df_clean = df[all_keep].copy()

print("   Metadata columns : " + ", ".join([c for c in METADATA_COLS if c in df.columns]))
print("   Feature columns  : " + ", ".join(FEATURE_COLS))
print("   Total columns    : {}".format(len(df_clean.columns)))

# Reset index
df_clean = df_clean.reset_index(drop=True)

final_row_count     = len(df_clean)
total_rows_removed  = original_row_count - final_row_count

# ------------------------------------------------------------------ #
#  11. Save cleaned dataset                                            #
# ------------------------------------------------------------------ #
section("11. SAVING CLEANED DATASET")
df_clean.to_csv(CLEANED_PATH, index=False, encoding="utf-8")
print("   Saved : " + CLEANED_PATH)
print("   Rows  : {:,}".format(final_row_count))
print("   Cols  : {}".format(len(df_clean.columns)))

# ------------------------------------------------------------------ #
#  12. Verification                                                    #
# ------------------------------------------------------------------ #
section("12. VERIFICATION")

checks = {}

# a) File exists
checks["movies_cleaned.csv created"] = os.path.exists(CLEANED_PATH)

# b) 'tags' column exists
checks["'tags' column present"] = "tags" in df_clean.columns

# c) 'title' column exists
checks["'title' column present"] = "title" in df_clean.columns

# d) genres is plain text (no '{' or '"id"' patterns)
genre_sample = df_clean["genres"].dropna().head(20)
checks["genres parsed (no JSON ids)"] = not genre_sample.str.contains('"id"').any()

# e) No row has both empty genres AND empty overview
both_empty = (
    (df_clean["genres"].fillna("").str.strip() == "") &
    (df_clean["overview"].fillna("").str.strip() == "")
).sum()
checks["No row has both empty genres & overview"] = (both_empty == 0)

# f) Original CSV unchanged
orig_rows = len(pd.read_csv(CSV_PATH, low_memory=False, nrows=5))
checks["Original movies.csv untouched"] = (orig_rows == 5)

print()
all_passed = True
for check, passed in checks.items():
    status = "[PASS]" if passed else "[FAIL]"
    if not passed:
        all_passed = False
    print("   {} {}".format(status, check))

print()
if all_passed:
    print("   All verification checks PASSED.")
else:
    print("   WARNING: Some checks FAILED - review above.")

# ------------------------------------------------------------------ #
#  13. Show 5 sample cleaned records                                   #
# ------------------------------------------------------------------ #
section("13. SAMPLE CLEANED RECORDS (5 rows)")
pd.set_option("display.max_colwidth", 80)
pd.set_option("display.width", 160)

sample5 = df_clean[["title", "genres", "overview", "tags"]].dropna(
    subset=["genres"]
).sample(5, random_state=7)

for idx, row in sample5.iterrows():
    print()
    print("  --- Record #{} ---".format(idx))
    print("  Title   : " + str(row["title"]))
    print("  Genres  : " + str(row["genres"]))
    overview_trunc = str(row["overview"])[:120] + "..." if len(str(row["overview"])) > 120 else str(row["overview"])
    tags_trunc     = str(row["tags"])[:120] + "..."     if len(str(row["tags"])) > 120     else str(row["tags"])
    print("  Overview: " + overview_trunc)
    print("  Tags    : " + tags_trunc)

# ------------------------------------------------------------------ #
#  14. Missing values in cleaned dataset                               #
# ------------------------------------------------------------------ #
section("14. MISSING VALUES IN CLEANED DATASET")
miss = df_clean.isnull().sum()
miss_pct = (miss / len(df_clean) * 100).round(2)
print("   {:<25} {:>12} {:>10}".format("Column", "Missing", "Missing %"))
print("   {:<25} {:>12} {:>10}".format("-"*25, "-"*12, "-"*10))
for col in df_clean.columns:
    print("   {:<25} {:>12,} {:>9.2f}%".format(col, int(miss[col]), miss_pct[col]))

# ------------------------------------------------------------------ #
#  15. Write preprocessing report                                      #
# ------------------------------------------------------------------ #
section("15. WRITING PREPROCESSING REPORT")

report_missing_lines = []
for col in df_clean.columns:
    report_missing_lines.append(
        "  {:<25} {:>8,}  ({:.2f}%)".format(col, int(miss[col]), miss_pct[col])
    )

report_text = textwrap.dedent("""\
======================================================================
  MOVIE RECOMMENDATION SYSTEM - PREPROCESSING REPORT
  Generated  : 2026-08-31
  Source file: {source}
  Output file: {output}
======================================================================

----------------------------------------------------------------------
 ROW COUNTS
----------------------------------------------------------------------
  Original row count                          : {orig:,}
  Rows removed - exact duplicates             : {exact_dups:,}
  Rows removed - duplicate movie IDs          : {dup_ids:,}
  Rows removed - both genres & overview empty : {no_content:,}
  Rows removed - no valid title               : {no_title:,}
  Final row count                             : {final:,}
  Total rows removed                          : {total_removed:,}

----------------------------------------------------------------------
 DUPLICATE ID HANDLING (DETAIL)
----------------------------------------------------------------------
  When two or more rows shared the same TMDB movie id, each row was
  scored based on how much textual content it contained:
    * overview present    -> +2 points
    * genres present      -> +2 points
    * title present       -> +1 point
    * tagline present     -> +1 point

  The row with the highest score was retained; in case of a tie the
  earlier-appearing row was kept.  This ensures the richest data is
  preserved without blindly dropping movies that share a title.

  Duplicate IDs handled this way: {dup_ids:,}

----------------------------------------------------------------------
 COLUMNS USED FOR RECOMMENDATION (feature text)
----------------------------------------------------------------------
  genres    - Parsed from JSON-like list; plain genre names only
  overview  - Plot synopsis; cleaned and whitespace-normalised
  tags      - Combined feature: genres + overview  (model input)

----------------------------------------------------------------------
 COLUMNS RETAINED AS METADATA (not used in model text)
----------------------------------------------------------------------
  id, imdb_id, title, original_title, original_language,
  release_date, popularity, vote_average, vote_count,
  poster_path, runtime, tagline

----------------------------------------------------------------------
 COLUMNS EXPLICITLY EXCLUDED
----------------------------------------------------------------------
  belongs_to_collection, homepage, budget, revenue,
  adult, video, status, production_companies,
  production_countries, spoken_languages

----------------------------------------------------------------------
 MISSING VALUES IN CLEANED DATASET
----------------------------------------------------------------------
{missing_table}

----------------------------------------------------------------------
 DATA TYPE CONVERSIONS APPLIED
----------------------------------------------------------------------
  popularity    -> float64   (invalid -> NaN via coercion)
  vote_average  -> float64   (invalid -> NaN via coercion)
  vote_count    -> float64   (invalid -> NaN via coercion)
  release_date  -> datetime64 (invalid -> NaT via coercion)

----------------------------------------------------------------------
 OUTPUT
----------------------------------------------------------------------
  movies_cleaned.csv rows    : {final:,}
  movies_cleaned.csv columns : {ncols}

======================================================================
 END OF REPORT
======================================================================
""").format(
    source=CSV_PATH,
    output=CLEANED_PATH,
    orig=original_row_count,
    exact_dups=exact_dups_removed,
    dup_ids=dup_ids_removed,
    no_content=no_content_removed,
    no_title=no_title_removed,
    final=final_row_count,
    total_removed=total_rows_removed,
    missing_table="\n".join(report_missing_lines),
    ncols=len(df_clean.columns),
)

with open(REPORT_PATH, "w", encoding="utf-8") as f:
    f.write(report_text)

print("   Report saved : " + REPORT_PATH)

# ------------------------------------------------------------------ #
#  16. Summary Banner                                                  #
# ------------------------------------------------------------------ #
section("PREPROCESSING COMPLETE - SUMMARY")
print("   Original rows : {:,}".format(original_row_count))
print("   Final rows    : {:,}".format(final_row_count))
print("   Rows removed  : {:,}".format(total_rows_removed))
print()
print("   Breakdown of removed rows:")
print("     Exact duplicates             : {:,}".format(exact_dups_removed))
print("     Duplicate IDs (kept best)    : {:,}".format(dup_ids_removed))
print("     Both genre+overview missing  : {:,}".format(no_content_removed))
print("     No valid title               : {:,}".format(no_title_removed))
print()
print("   Output : " + CLEANED_PATH)
print("   Report : " + REPORT_PATH)
print()
print("   Ready for Step 3 -> TF-IDF vectorisation.\n")
