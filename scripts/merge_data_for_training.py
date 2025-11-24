import pandas as pd
import glob
import os

def merge_csvs(pattern, output_file):
    print(f"Merging {pattern} into {output_file}...")
    files = sorted(glob.glob(pattern))
    if not files:
        print(f"No files found for {pattern}")
        return

    dfs = []
    for f in files:
        try:
            # Read with low_memory=False to avoid dtypes warning, or specify dtypes if known
            df = pd.read_csv(f, low_memory=False)
            dfs.append(df)
        except Exception as e:
            print(f"Error reading {f}: {e}")
    
    if dfs:
        merged_df = pd.concat(dfs, ignore_index=True)
        merged_df.to_csv(output_file, index=False)
        print(f"Saved {output_file} with {len(merged_df)} rows.")
    else:
        print("Nothing to merge.")

def main():
    # Merge Results
    # Pattern: keirin_results_YYYYMMDD.csv or ranges
    # We want to avoid duplicating data if both daily and quarterly files exist.
    # The file list showed quarterly files like keirin_results_20240101_20240331_full.csv
    # and also daily files.
    # Let's target the quarterly/large files if possible, or just all and drop duplicates.
    
    # Actually, looking at the file list, there are many daily files.
    # Best approach: Merge everything that looks like a result file, then drop duplicates based on race_date, keirin_cd, race_no.
    
    # Results
    merge_csvs("data/keirin_results_202*.csv", "data/keirin_results_20240101_20251004.csv")
    
    # Prerace
    merge_csvs("data/keirin_prerace_202*.csv", "data/keirin_prerace_20240101_20251004.csv")
    
    # Entries
    merge_csvs("data/keirin_race_detail_entries_202*.csv", "data/keirin_race_detail_entries_20240101_20251004.csv")

if __name__ == "__main__":
    main()
