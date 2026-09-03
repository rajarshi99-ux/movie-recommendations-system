import pandas as pd
import numpy as np

def run_audit():
    print("Loading movies.csv...")
    try:
        movies_df = pd.read_csv('movies.csv', low_memory=False)
        # tmdb ids in movies.csv
        # some rows have malformed ids (e.g. dates instead of digits)
        movies_df['id'] = pd.to_numeric(movies_df['id'], errors='coerce')
        valid_movies_mask = movies_df['id'].notna()
        movie_tmdb_set = set(movies_df.loc[valid_movies_mask, 'id'].astype(int))
    except Exception as e:
        print("Error loading movies.csv:", e)
        return

    print("Loading movielens/ratings.csv...")
    try:
        ratings_df = pd.read_csv('movielens/ratings.csv', low_memory=False)
        num_ratings = len(ratings_df)
        num_users = ratings_df['userId'].nunique()
        num_ml_movies = ratings_df['movieId'].nunique()
        # To save memory, we can drop ratings_df after extracting exact stats
        ml_movie_set = set(ratings_df['movieId'].unique())
        del ratings_df 
    except Exception as e:
        print("Error loading ratings.csv:", e)
        return

    print("Loading movielens/links.csv...")
    try:
        links_df = pd.read_csv('movielens/links.csv', low_memory=False)
        
        # Filter links to only those movies that actually have ratings
        links_df = links_df[links_df['movieId'].isin(ml_movie_set)]
    except Exception as e:
        print("Error loading links.csv:", e)
        return

    print("\n--- AUDIT RESULTS ---")
    print(f"Number of MovieLens ratings: {num_ratings:,}")
    print(f"Number of unique MovieLens users: {num_users:,}")
    print(f"Number of unique MovieLens movies with ratings: {num_ml_movies:,}")
    
    links_df['tmdbId'] = pd.to_numeric(links_df['tmdbId'], errors='coerce')
    valid_links = links_df.dropna(subset=['tmdbId']).copy()
    valid_links['tmdbId'] = valid_links['tmdbId'].astype(int)
    num_valid_tmdb = len(valid_links)
    print(f"Number of valid TMDB IDs in links: {num_valid_tmdb:,}")
    
    # duplicates
    duplicates = valid_links.duplicated(subset=['tmdbId']).sum()
    print(f"Number of duplicate TMDB mappings: {duplicates:,}")
    
    # unique mapped TMDB IDs from MovieLens
    unique_ml_tmdb_ids = set(valid_links['tmdbId'])
    
    matched_ids = unique_ml_tmdb_ids.intersection(movie_tmdb_set)
    num_matched = len(matched_ids)
    
    unmatched_ids = unique_ml_tmdb_ids - movie_tmdb_set
    num_unmatched = len(unmatched_ids)
    
    print(f"Number of successfully matched TMDB IDs with movies.csv: {num_matched:,}")
    print(f"Number of unmatched TMDB IDs: {num_unmatched:,}")
    
    if len(unique_ml_tmdb_ids) > 0:
        mapping_pct = (num_matched / len(unique_ml_tmdb_ids)) * 100
    else:
        mapping_pct = 0.0
    print(f"Final mapping percentage: {mapping_pct:.2f}%")

if __name__ == '__main__':
    run_audit()
