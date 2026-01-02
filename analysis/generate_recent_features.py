import pandas as pd
import glob
from pathlib import Path
from tqdm import tqdm

def read_csv_safe(path):
    encodings = ["cp932", "utf-8", None]
    for enc in encodings:
        try:
            if enc is None:
                return pd.read_csv(path, low_memory=False)
            return pd.read_csv(path, encoding=enc, low_memory=False)
        except UnicodeDecodeError:
            continue
    return None

def load_all_data():
    # Load Entries
    print("Loading entries...")
    entry_files = sorted(glob.glob("data/keirin_race_detail_entries_*.csv"))
    dfs = []
    for f in tqdm(entry_files):
        df = read_csv_safe(f)
        if df is not None:
            dfs.append(df)
    entries = pd.concat(dfs, ignore_index=True)
    
    # Load Results
    print("Loading results...")
    result_files = sorted(glob.glob("data/keirin_results_*.csv"))
    dfs = []
    for f in tqdm(result_files):
        df = read_csv_safe(f)
        if df is not None:
            dfs.append(df)
    results = pd.concat(dfs, ignore_index=True)
    
    return entries, results

def generate_features():
    entries, results = load_all_data()
    
    # Preprocess
    entries['race_date'] = pd.to_datetime(entries['race_date'].astype(str))
    results['race_date'] = pd.to_datetime(results['race_date'].astype(str))
    
    # Prepare results lookup
    # We need to know for each race_date, keirin_cd, race_no, who was 1st, 2nd, 3rd.
    # Results file has pos1_name, pos2_name, pos3_name.
    # We'll melt this to have (race_id, rank, name)
    
    # Create unique race ID
    # Entries has 'track' but not 'keirin_cd'. Results has both.
    # We use 'track' name for matching.
    
    # Clean race_no in results (remove 'R' suffix if present)
    results = results.dropna(subset=['race_no'])
    results['race_no'] = results['race_no'].astype(str).str.replace('R', '', regex=False)
    results = results[results['race_no'].str.isnumeric()]
    results['race_no'] = results['race_no'].astype(int)
    
    results['race_id'] = results['race_date'].astype(str) + "_" + results['track'].astype(str) + "_" + results['race_no'].astype(str)
    entries['race_id'] = entries['race_date'].astype(str) + "_" + entries['track'].astype(str) + "_" + entries['race_no'].astype(str)
    
    # Extract winners
    winners = results[['race_id', 'pos1_name', 'pos2_name', 'pos3_name']].copy()
    
    # Map rider results
    # We want to attach "rank" to entries.
    # Since names might not match perfectly (spaces etc), we'll try to match by name.
    # Assuming names are consistent enough.
    
    print("Mapping results to entries...")
    # This is slow if we loop. Let's try merge.
    # Melt winners
    winners_melt = winners.melt(id_vars=['race_id'], value_vars=['pos1_name', 'pos2_name', 'pos3_name'], var_name='rank_col', value_name='sensyuName')
    winners_melt['rank'] = winners_melt['rank_col'].map({'pos1_name': 1, 'pos2_name': 2, 'pos3_name': 3})
    winners_melt = winners_melt.drop(columns=['rank_col'])
    
    # Merge rank into entries
    # entries has 'sensyuName'
    entries = entries.merge(winners_melt, on=['race_id', 'sensyuName'], how='left')
    entries['rank'] = entries['rank'].fillna(9) # 9 for out of top 3
    
    # Sort by rider and date
    entries = entries.sort_values(['sensyuName', 'race_date'])
    
    # Calculate rolling features
    print("Calculating rolling features...")
    
    # We need to calculate stats *before* the current race.
    # So we shift the results.
    
    # Group by rider
    grouped = entries.groupby('sensyuName')
    
    # Helper for rolling
    def calc_rolling(x, window=5):
        # 1 if rank=1, else 0
        is_win = (x['rank'] == 1).astype(float)
        is_2ren = (x['rank'] <= 2).astype(float)
        is_3ren = (x['rank'] <= 3).astype(float)
        
        # Rolling mean (shifted by 1 to not include current race)
        win_rate = is_win.shift(1).rolling(window, min_periods=1).mean()
        ren2_rate = is_2ren.shift(1).rolling(window, min_periods=1).mean()
        ren3_rate = is_3ren.shift(1).rolling(window, min_periods=1).mean()
        
        return pd.DataFrame({
            'recent_win_rate': win_rate,
            'recent_2ren_rate': ren2_rate,
            'recent_3ren_rate': ren3_rate
        }, index=x.index)

    # Apply rolling calculation
    # group_keys=False ensures the original index is preserved (if pandas >= 1.5)
    # If older pandas, we might need to handle MultiIndex.
    try:
        features_df = grouped.apply(calc_rolling, include_groups=False)
    except TypeError:
        # Fallback for older pandas or different signature
        features_df = grouped.apply(calc_rolling)

    # If features_df has MultiIndex (sensyuName, index), drop level 0
    if isinstance(features_df.index, pd.MultiIndex):
        features_df = features_df.droplevel(0)

    # Merge back
    # Ensure indices match
    entries = entries.join(features_df)
    
    # Fill NA (first races) with 0
    entries['recent_win_rate'] = entries['recent_win_rate'].fillna(0)
    entries['recent_2ren_rate'] = entries['recent_2ren_rate'].fillna(0)
    entries['recent_3ren_rate'] = entries['recent_3ren_rate'].fillna(0)
    
    # Select columns to save
    # entries has 'track' but not 'keirin_cd'
    output_df = entries[['race_date', 'track', 'race_no', 'sensyuName', 'recent_win_rate', 'recent_2ren_rate', 'recent_3ren_rate']]
    
    # Format race_date back to int YYYYMMDD
    output_df['race_date'] = output_df['race_date'].dt.strftime('%Y%m%d').astype(int)
    
    output_path = "analysis/model_outputs/rider_recent_features.csv"
    print(f"Saving to {output_path}...")
    output_df.to_csv(output_path, index=False)

if __name__ == "__main__":
    generate_features()
