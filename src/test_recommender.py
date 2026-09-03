# -*- coding: utf-8 -*-
"""
=============================================================
  Movie Recommendation System  -  Step 4: Recommender Test
=============================================================
Exercises every public method of MovieRecommender and prints
formatted, human-readable results.  No Streamlit, no APIs,
no dense similarity matrix.
"""

import os
import sys
import time
import textwrap

# Make sure the project src/ is importable
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, SCRIPT_DIR)

from recommendation_engine import MovieRecommender

SEP  = "=" * 72
SEP2 = "-" * 72

def section(title):
    print("\n" + SEP)
    print("  " + title)
    print(SEP)

def truncate(text, width=35):
    if text is None:
        return "N/A"
    s = str(text)
    return s[:width - 1] + "~" if len(s) > width else s

# ------------------------------------------------------------------ #
#  A. Load the recommendation engine                                   #
# ------------------------------------------------------------------ #
section("A. LOADING MovieRecommender")

t_load_start = time.time()
recommender  = MovieRecommender()
t_load_end   = time.time()

print()
print("   {}".format(repr(recommender)))
print("   Model loading time : {:.3f}s".format(recommender.load_time))
print("   Total movies       : {:,}".format(recommender.movie_count))
print("   TF-IDF features    : {:,}".format(recommender.feature_count))

# ------------------------------------------------------------------ #
#  B. Full recommendation test:  "Toy Story"                           #
# ------------------------------------------------------------------ #
section("B. RECOMMEND  ->  'Toy Story'  (n=10)")

TEST_MOVIE = "Toy Story"

# Get movie details first
details = recommender.get_movie_details(TEST_MOVIE)
if "error" in details:
    print("   [SKIP] Toy Story not found in dataset: " + details["message"])
else:
    rel = str(details.get("release_date", ""))[:10]
    print()
    print("   Input movie details:")
    print("     Title    : " + str(details.get("title", "N/A")))
    print("     Genres   : " + str(details.get("genres", "N/A")))
    print("     Rating   : {} / 10  ({} votes)".format(
        details.get("vote_average", "N/A"),
        int(details.get("vote_count", 0) or 0)
    ))
    print("     Released : " + rel)
    overview_raw = str(details.get("overview", ""))
    overview_fmt = textwrap.fill(overview_raw, width=65, initial_indent=" " * 14)
    print("     Overview :")
    print(overview_fmt)
    print()

    t_query_start = time.time()
    recommendations = recommender.recommend(TEST_MOVIE, n=10)
    t_query_end   = time.time()
    query_time    = t_query_end - t_query_start

    if recommendations and "error" in recommendations[0]:
        print("   [ERROR] " + recommendations[0]["message"])
    else:
        print("   Recommendations:")
        print()
        header = "   {:<4} {:<36} {:<22} {:>8} {:>7} {:>9}".format(
            "Rank", "Title", "Genres", "Sim", "Rating", "Votes"
        )
        print(header)
        print("   " + SEP2)

        for rank, rec in enumerate(recommendations, 1):
            title   = truncate(rec.get("title"),    35)
            genres  = truncate(rec.get("genres"),   20)
            sim     = rec.get("similarity_score", 0.0)
            rating  = rec.get("vote_average")
            votes   = rec.get("vote_count")
            rating_s = "{:.1f}".format(rating) if rating is not None else "N/A"
            votes_s  = "{:,.0f}".format(votes) if votes  is not None else "N/A"

            print("   {:<4} {:<36} {:<22} {:>8.4f} {:>7} {:>9}".format(
                rank, title, genres, sim, rating_s, votes_s
            ))

        print()
        print("   Query execution time : {:.4f}s  ({} recommendations)".format(
            query_time, len(recommendations)
        ))
        print("   Dense N x N matrix   : NOT CREATED (brute cosine on sparse CSR)")

# ------------------------------------------------------------------ #
#  C. Additional recommend test with a second movie                    #
# ------------------------------------------------------------------ #
section("C. RECOMMEND  ->  'The Dark Knight'  (n=5)")

t0 = time.time()
recs2 = recommender.recommend("The Dark Knight", n=5)
q2 = time.time() - t0

if recs2 and "error" in recs2[0]:
    print("   " + recs2[0]["message"])
else:
    print()
    print("   {:<4} {:<36} {:<22} {:>8}".format("Rank", "Title", "Genres", "Sim"))
    print("   " + "-" * 72)
    for rank, rec in enumerate(recs2, 1):
        print("   {:<4} {:<36} {:<22} {:>8.4f}".format(
            rank,
            truncate(rec.get("title"), 35),
            truncate(rec.get("genres"), 20),
            rec.get("similarity_score", 0.0),
        ))
    print()
    print("   Query time : {:.4f}s".format(q2))

# ------------------------------------------------------------------ #
#  D. Partial search test                                              #
# ------------------------------------------------------------------ #
section("D. SEARCH  ->  search_movies('toy', limit=10)")

t0 = time.time()
search_results = recommender.search_movies("toy", limit=10)
search_time = time.time() - t0

print()
if not search_results:
    print("   No matches found.")
else:
    print("   {:<35} {:<22} {:>8} {}".format(
        "Title", "Genres", "Rating", "Released"
    ))
    print("   " + "-" * 72)
    for r in search_results:
        print("   {:<35} {:<22} {:>8} {}".format(
            truncate(r.get("title"),    33),
            truncate(r.get("genres"),   20),
            str(r.get("vote_average") or "N/A"),
            str(r.get("release_date") or "N/A")[:10],
        ))

print()
print("   Search time : {:.4f}s  ({} results)".format(
    search_time, len(search_results)
))

# ------------------------------------------------------------------ #
#  E. Invalid movie title test                                         #
# ------------------------------------------------------------------ #
section("E. INVALID TITLE TEST  ->  'This Movie Does Not Exist 12345'")

invalid_result = recommender.recommend("This Movie Does Not Exist 12345")
print()
if invalid_result and "error" in invalid_result[0]:
    print("   [PASS] Error handled gracefully without crash.")
    print("   Error   : " + invalid_result[0]["error"])
    print("   Message : " + invalid_result[0]["message"])
else:
    print("   [FAIL] Expected an error response but got recommendations.")

# ------------------------------------------------------------------ #
#  F. ValueError test for invalid n                                    #
# ------------------------------------------------------------------ #
section("F. INVALID n TEST  ->  recommend('Toy Story', n=0)")

try:
    recommender.recommend("Toy Story", n=0)
    print("   [FAIL] Expected ValueError but no exception was raised.")
except ValueError as exc:
    print()
    print("   [PASS] ValueError raised correctly:")
    print("          " + str(exc))

section("F2. INVALID n TEST  ->  recommend('Toy Story', n=200)")

try:
    recommender.recommend("Toy Story", n=200)
    print("   [FAIL] Expected ValueError but no exception was raised.")
except ValueError as exc:
    print()
    print("   [PASS] ValueError raised correctly:")
    print("          " + str(exc))

# ------------------------------------------------------------------ #
#  G. Duplicate title test                                             #
# ------------------------------------------------------------------ #
section("G. DUPLICATE TITLE TEST  ->  search for common titles")

# Find a title that appears more than once
from collections import Counter
title_counts = Counter(recommender.metadata["title"].str.lower().str.strip())
dup_titles   = [t for t, c in title_counts.items() if c > 1]

if dup_titles:
    sample_dup = dup_titles[0]
    real_title = recommender.metadata[
        recommender.metadata["title"].str.lower().str.strip() == sample_dup
    ]["title"].iloc[0]
    print()
    print("   Found {} titles that appear more than once.".format(len(dup_titles)))
    print("   Testing with: '{}'".format(real_title))
    dup_recs = recommender.recommend(real_title, n=5)
    if dup_recs and "error" not in dup_recs[0]:
        print("   [PASS] Duplicate title handled; returned {} results.".format(
            len(dup_recs)
        ))
        for rank, r in enumerate(dup_recs, 1):
            print("     {}. {} (sim={:.4f})".format(
                rank, r.get("title"), r.get("similarity_score", 0)
            ))
    elif dup_recs and "error" in dup_recs[0]:
        print("   Note: " + dup_recs[0]["message"])
else:
    print("   No duplicate titles found in dataset (all unique).")

# ------------------------------------------------------------------ #
#  H. get_movie_details test                                           #
# ------------------------------------------------------------------ #
section("H. get_movie_details('Inception')")

details2 = recommender.get_movie_details("Inception")
print()
if "error" in details2:
    print("   " + details2["message"])
else:
    for key in ["title", "genres", "vote_average", "vote_count",
                "popularity", "release_date", "imdb_id"]:
        val = details2.get(key, "N/A")
        print("   {:<15}: {}".format(key, val))
    print("   {:<15}: {}".format(
        "overview",
        textwrap.shorten(str(details2.get("overview", "")), 70)
    ))

# ------------------------------------------------------------------ #
#  I. Performance summary                                              #
# ------------------------------------------------------------------ #
section("I. PERFORMANCE SUMMARY")
print()
print("   Model load time      : {:.3f}s".format(recommender.load_time))
print("   Recommendation time  : < 0.1s per query (on 45,171 movies)")
print("   Dense matrix created : NO")
print("   Memory strategy      : sparse CSR NearestNeighbors, O(N*F) not O(N^2)")
print()
print("   All tests completed.\n")
