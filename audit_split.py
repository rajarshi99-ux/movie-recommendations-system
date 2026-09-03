import pandas as pd
import numpy as np
from datetime import datetime

def run_split_audit():
    print("Loading movies.csv...")
    movies_df = pd.read_csv('movies.csv', low_memory=False)
    movies_df['id'] = pd.to_numeric(movies_df['id'], errors='coerce')
    valid_movies_mask = movies_df['id'].notna()
    tmdb_id_set = set(movies_df.loc[valid_movies_mask, 'id'].astype(int))

    print("Loading links.csv...")
    links_df = pd.read_csv('movielens/links.csv')
    links_df['tmdbId'] = pd.to_numeric(links_df['tmdbId'], errors='coerce')
    valid_links = links_df.dropna(subset=['tmdbId']).copy()
    valid_links['tmdbId'] = valid_links['tmdbId'].astype(int)
    # keep only links in our movies dataset
    valid_links = valid_links[valid_links['tmdbId'].isin(tmdb_id_set)]
    ml_to_tmdb = dict(zip(valid_links['movieId'], valid_links['tmdbId']))

    print("Loading ratings.csv...")
    ratings_df = pd.read_csv('movielens/ratings.csv')
    total_interactions = len(ratings_df)
    
    print("Mapping...")
    ratings_df['tmdbId'] = ratings_df['movieId'].map(ml_to_tmdb)
    mapped_mask = ratings_df['tmdbId'].notna()
    
    mapped_interactions = int(mapped_mask.sum())
    removed_interactions = total_interactions - mapped_interactions
    
    mapped_df = ratings_df[mapped_mask].copy()
    mapped_df['tmdbId'] = mapped_df['tmdbId'].astype(int)
    
    # 1/2: Min/Max TS
    min_ts = mapped_df['timestamp'].min()
    max_ts = mapped_df['timestamp'].max()
    min_date = datetime.fromtimestamp(min_ts).strftime('%Y-%m-%d %H:%M:%S')
    max_date = datetime.fromtimestamp(max_ts).strftime('%Y-%m-%d %H:%M:%S')
    
    # 3/4/5/6: Counts
    unique_users_mapped = mapped_df['userId'].nunique()
    
    user_counts = mapped_df['userId'].value_counts()
    users_ge_5 = int((user_counts >= 5).sum())
    users_ge_10 = int((user_counts >= 10).sum())
    
    print("\n==================================================")
    print("PHASE 3 / PART A: DATASET TIMESTAMP SCOPE")
    print("==================================================")
    print(f"1. Minimum timestamp/date: {min_ts} ({min_date})")
    print(f"2. Maximum timestamp/date: {max_ts} ({max_date})")
    print(f"3. Number of unique users after TMDB mapping: {unique_users_mapped:,}")
    print(f"4. Number of mapped movie interactions: {mapped_interactions:,}")
    print(f"5. Number of interactions removed because TMDB mapping is unavailable: {removed_interactions:,}")
    print(f"6. Number of users remaining after mapping: {unique_users_mapped:,}")
    print(f"7. Number of users with at least 5 mapped ratings: {users_ge_5:,}")
    print(f"8. Number of users with at least 10 mapped ratings: {users_ge_10:,}")
    
    # Implement strict temporal split
    print("\n==================================================")
    print("PHASE 3 / PART B: TEMPORAL LEAKAGE-FREE SPLITTING")
    print("==================================================")
    print("Proposing Split Strategy: Temporal Fractional Segmentation")
    print("- Eligible: Users with >= 20 mapped ratings (to ensure adequate training context + val/test resolution).")
    print("- Rule: Sort sequentially by timestamp per user.")
    print("- Sub-rule: Oldest 80% assigned to TRAIN, next 10% to VALIDATION, newest 10% strictly to FINAL TEST.")
    
    eligible_users = user_counts[user_counts >= 20].index
    print(f"Identified {len(eligible_users):,} strictly eligible users for meaningful splits.")
    
    # Subsample 15,000 eligible users for evaluation tractability
    print("Subsetting 15,000 deterministic users for real-time memory scaling (Seed: 42).")
    np.random.seed(42)
    sampled_users = np.random.choice(eligible_users, size=min(15000, len(eligible_users)), replace=False)
    
    split_df = mapped_df[mapped_df['userId'].isin(sampled_users)].copy()
    
    # Sort
    split_df.sort_values(by=['userId', 'timestamp'], inplace=True)
    
    # Rank & Segment
    split_df['rank'] = split_df.groupby('userId').cumcount() + 1
    split_df['user_total'] = split_df.groupby('userId')['userId'].transform('size')
    split_df['pct'] = split_df['rank'] / split_df['user_total']
    
    split_df['split'] = 'train'
    split_df.loc[split_df['pct'] > 0.80, 'split'] = 'val'
    split_df.loc[split_df['pct'] > 0.90, 'split'] = 'test'
    
    train = split_df[split_df['split'] == 'train']
    val = split_df[split_df['split'] == 'val']
    test = split_df[split_df['split'] == 'test']
    
    train_idx = set(train.index)
    val_idx = set(val.index)
    test_idx = set(test.index)
    
    print("\nVERIFYING ZERO-LEAKAGE INTERSECTION PROTOCOL:")
    print(f"TRAIN \u2229 VALIDATION = empty (Found elements: {len(train_idx & val_idx)}) -> {len(train_idx & val_idx) == 0}")
    print(f"TRAIN \u2229 TEST = empty (Found elements: {len(train_idx & test_idx)}) -> {len(train_idx & test_idx) == 0}")
    print(f"VALIDATION \u2229 TEST = empty (Found elements: {len(val_idx & test_idx)}) -> {len(val_idx & test_idx) == 0}")
    
    # Verify temporal isolation logic
    print("\nVERIFYING CHRONOLOGICAL INTEGRITY:")
    train_max = train.groupby('userId')['timestamp'].max()
    val_min = val.groupby('userId')['timestamp'].min()
    val_max = val.groupby('userId')['timestamp'].max()
    test_min = test.groupby('userId')['timestamp'].min()
    
    check_val = pd.merge(train_max, val_min, on='userId', suffixes=('_train_max', '_val_min'))
    val_valid = (check_val['timestamp_train_max'] <= check_val['timestamp_val_min']).all()
    
    check_test = pd.merge(val_max, test_min, on='userId', suffixes=('_val_max', '_test_min'))
    test_valid = (check_test['timestamp_val_max'] <= check_test['timestamp_test_min']).all()
    
    print(f"Check passed: max(train_timestamp) <= min(validation_timestamp) across all valid users -> {val_valid}")
    print(f"Check passed: max(validation_timestamp) <= min(test_timestamp) across all valid users -> {test_valid}")
    
    print("\nSAMPLE TARGET SPLITS (First 5 users):")
    sample_users = sampled_users[:5]
    for su in sample_users:
        uc = split_df[split_df['userId'] == su]
        tc = len(uc[uc['split'] == 'train'])
        vc = len(uc[uc['split'] == 'val'])
        tec = len(uc[uc['split'] == 'test'])
        print(f"User [{su}]: Total: {len(uc)} -> Train={tc}, Val={vc}, Test={tec}")

if __name__ == '__main__':
    run_split_audit()
