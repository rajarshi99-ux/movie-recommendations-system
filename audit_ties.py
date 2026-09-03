import pandas as pd
import numpy as np

def run_tie_audit():
    movies_df = pd.read_csv('movies.csv', low_memory=False)
    movies_df['id'] = pd.to_numeric(movies_df['id'], errors='coerce')
    tmdb_id_set = set(movies_df.loc[movies_df['id'].notna(), 'id'].astype(int))

    links_df = pd.read_csv('movielens/links.csv')
    links_df['tmdbId'] = pd.to_numeric(links_df['tmdbId'], errors='coerce')
    valid_links = links_df.dropna(subset=['tmdbId']).copy()
    valid_links['tmdbId'] = valid_links['tmdbId'].astype(int)
    valid_links = valid_links[valid_links['tmdbId'].isin(tmdb_id_set)]
    ml_to_tmdb = dict(zip(valid_links['movieId'], valid_links['tmdbId']))

    ratings_df = pd.read_csv('movielens/ratings.csv')
    ratings_df['tmdbId'] = ratings_df['movieId'].map(ml_to_tmdb)
    
    mapped_df = ratings_df[ratings_df['tmdbId'].notna()].copy()
    mapped_df['tmdbId'] = mapped_df['tmdbId'].astype(int)
    
    user_counts = mapped_df['userId'].value_counts()
    eligible_users = user_counts[user_counts >= 20].index
    
    np.random.seed(42)
    sampled_users = np.random.choice(eligible_users, size=min(15000, len(eligible_users)), replace=False)
    
    split_df = mapped_df[mapped_df['userId'].isin(sampled_users)].copy()
    
    split_df.sort_values(by=['userId', 'timestamp', 'tmdbId'], inplace=True)
    
    split_df['rank'] = split_df.groupby('userId').cumcount() + 1
    split_df['user_total'] = split_df.groupby('userId')['userId'].transform('size')
    split_df['pct'] = split_df['rank'] / split_df['user_total']
    
    split_df['split'] = 'train'
    split_df.loc[split_df['pct'] > 0.80, 'split'] = 'val'
    split_df.loc[split_df['pct'] > 0.90, 'split'] = 'test'

    train = split_df[split_df['split'] == 'train']
    val = split_df[split_df['split'] == 'val']
    test = split_df[split_df['split'] == 'test']
    
    t_max = train.groupby('userId')['timestamp'].max().reset_index(name='train_max')
    v_min = val.groupby('userId')['timestamp'].min().reset_index(name='val_min')
    v_max = val.groupby('userId')['timestamp'].max().reset_index(name='val_max')
    te_min = test.groupby('userId')['timestamp'].min().reset_index(name='test_min')
    
    tv_df = pd.merge(t_max, v_min, on='userId')
    tv_tied_users = tv_df[tv_df['train_max'] == tv_df['val_min']]['userId'].unique()
    
    vte_df = pd.merge(v_max, te_min, on='userId')
    vte_tied_users = vte_df[vte_df['val_max'] == vte_df['test_min']]['userId'].unique()
    
    all_tied_users = set(tv_tied_users) | set(vte_tied_users)
    affected_interactions = split_df[split_df['userId'].isin(all_tied_users)].shape[0]
    
    print("==================================================")
    print("1. BEFORE REDESIGN: TIMESTAMP TIE AUDIT")
    print("==================================================")
    print(f"Total eligible users: 15,000")
    print(f"Number of users with TRAIN/VAL timestamp-boundary ties: {len(tv_tied_users)}")
    print(f"Number of users with VAL/TEST timestamp-boundary ties: {len(vte_tied_users)}")
    print(f"Total unique affected users: {len(all_tied_users)}")
    print(f"Total affected interactions (for those users): {affected_interactions}")
    print(f"Percentage of users affected: {len(all_tied_users) / 15000 * 100:.2f}%")
    
    print("\nSAMPLE TIES (10 concrete examples):")
    tied_samples = list(all_tied_users)[:10]
    for su in tied_samples:
        uc = split_df[split_df['userId'] == su]
        t_ts = set(uc[uc['split'] == 'train']['timestamp'])
        v_ts = set(uc[uc['split'] == 'val']['timestamp'])
        te_ts = set(uc[uc['split'] == 'test']['timestamp'])
        shared_tv = t_ts & v_ts
        shared_vte = v_ts & te_ts
        shared_ts = shared_tv | shared_vte
        
        for ts in shared_ts:
            ts_group = uc[uc['timestamp'] == ts]
            grouped_splits = ts_group['split'].value_counts().to_dict()
            print(f"UserId: {su} | Timestamp: {ts} | Interactions in group: {len(ts_group)} | Original Assignment: {grouped_splits}")

    print("\n==================================================")
    print("2. AFTER REDESIGN: PRESERVING TIMESTAMP GROUPS")
    print("==================================================")
    
    ts_sizes = split_df.groupby(['userId', 'timestamp']).size().reset_index(name='group_size')
    ts_sizes['cum_size'] = ts_sizes.groupby('userId')['group_size'].cumsum()
    ts_sizes['user_total'] = ts_sizes.groupby('userId')['group_size'].transform('sum')
    
    ts_sizes['pct_after_group'] = ts_sizes['cum_size'] / ts_sizes['user_total']
    ts_sizes['split'] = 'train'
    ts_sizes['pct_before_group'] = ts_sizes.groupby('userId')['pct_after_group'].shift(1).fillna(0)
    ts_sizes['pct_mid'] = (ts_sizes['pct_before_group'] + ts_sizes['pct_after_group']) / 2
    ts_sizes.loc[ts_sizes['pct_mid'] >= 0.80, 'split'] = 'val'
    ts_sizes.loc[ts_sizes['pct_mid'] >= 0.90, 'split'] = 'test'
    
    split_mapping = ts_sizes.set_index(['userId', 'timestamp'])['split'].to_dict()
    split_df['new_split'] = split_df.set_index(['userId', 'timestamp']).index.map(split_mapping)
    
    nt = split_df[split_df['new_split'] == 'train']
    nv = split_df[split_df['new_split'] == 'val']
    nte = split_df[split_df['new_split'] == 'test']
    
    nt_idx = set(nt.index)
    nv_idx = set(nv.index)
    nte_idx = set(nte.index)
    print("VERIFYING PARTITION ISOLATION:")
    print(f"TRAIN AND VALIDATION = empty: {len(nt_idx & nv_idx) == 0}")
    print(f"TRAIN AND TEST = empty: {len(nt_idx & nte_idx) == 0}")
    print(f"VALIDATION AND TEST = empty: {len(nv_idx & nte_idx) == 0}")
    
    nt_max = nt.groupby('userId')['timestamp'].max().reset_index(name='t_max')
    nv_min = nv.groupby('userId')['timestamp'].min().reset_index(name='v_min')
    nv_max = nv.groupby('userId')['timestamp'].max().reset_index(name='v_max')
    nte_min = nte.groupby('userId')['timestamp'].min().reset_index(name='te_min')
    
    check_v = pd.merge(nt_max, nv_min, on='userId')
    v_strict_passed = (check_v['t_max'] < check_v['v_min']).all()
    print(f"max(train_timestamp) < min(validation_timestamp): {v_strict_passed}")
    
    check_te = pd.merge(nv_max, nte_min, on='userId')
    te_strict_passed = (check_te['v_max'] < check_te['te_min']).all()
    print(f"max(validation_timestamp) < min(test_timestamp): {te_strict_passed}")
    
    df_counts = split_df.groupby(['userId', 'new_split']).size().unstack(fill_value=0)
    for col in ['train', 'val', 'test']:
        if col not in df_counts: df_counts[col] = 0
            
    all_3_mask = (df_counts['train'] > 0) & (df_counts['val'] > 0) & (df_counts['test'] > 0)
    users_with_3 = all_3_mask.sum()
    
    print("\n==================================================")
    print("3. FINAL REDESIGNED COUNTS")
    print("==================================================")
    print(f"Final train interactions: {df_counts['train'].sum():,}")
    print(f"Final validation interactions: {df_counts['val'].sum():,}")
    print(f"Final test interactions: {df_counts['test'].sum():,}")
    print(f"Number of users with all three partitions: {users_with_3:,}")
    
    if users_with_3 > 0:
        valid_df = df_counts[all_3_mask]
        print(f"Minimum train interactions per user: {valid_df['train'].min()}")
        print(f"Minimum validation interactions per user: {valid_df['val'].min()}")
        print(f"Minimum test interactions per user: {valid_df['test'].min()}")
        
        print(f"Maximum train/val/test train interactions per user: {valid_df['train'].max()}")
        print(f"Maximum train/val/test validation interactions per user: {valid_df['val'].max()}")
        print(f"Maximum train/val/test test interactions per user: {valid_df['test'].max()}")
        
        print(f"Median counts for Train: {valid_df['train'].median()}")
        print(f"Median counts for Val: {valid_df['val'].median()}")
        print(f"Median counts for Test: {valid_df['test'].median()}")

if __name__ == '__main__':
    run_tie_audit()
