
import pandas as pd
from analysis import prerace_model
from pathlib import Path
import traceback

def debug_loading():
    print("Debugging data loading...")
    
    results_path = Path("data/keirin_results_20240101_20251004.csv")
    prerace_path = Path("data/keirin_prerace_20240101_20251004.csv")
    entries_path = Path("data/keirin_race_detail_entries_20240101_20251004.csv")
    
    print(f"Loading results from {results_path}...")
    try:
        results = prerace_model.load_results_table(results_path, 10000)
        print(f"Results loaded: {len(results)} rows")
    except Exception:
        traceback.print_exc()
        return

    print(f"Loading calendar from {prerace_path}...")
    try:
        calendar = prerace_model.load_prerace_calendar(prerace_path)
        print(f"Calendar loaded: {len(calendar)} rows")
    except Exception:
        traceback.print_exc()
        return

    print(f"Loading entries from {entries_path}...")
    try:
        rider_features = prerace_model.aggregate_rider_features(entries_path)
        print(f"Rider features loaded: {len(rider_features)} rows")
    except Exception:
        traceback.print_exc()
        return

    print("Merging datasets...")
    try:
        print("Results dtypes:\n", results[["race_date", "keirin_cd", "race_no", "track"]].dtypes)
        print("Results sample:\n", results[["race_date", "keirin_cd", "race_no", "track"]].head())
        
        print("Calendar dtypes:\n", calendar[["race_date", "keirin_cd", "race_no"]].dtypes)
        print("Calendar sample:\n", calendar[["race_date", "keirin_cd", "race_no"]].head())
        
        dataset = results.merge(calendar, on=["race_date", "keirin_cd", "race_no"], how="inner")
        print(f"Merged results+calendar: {len(dataset)} rows")
        
        print("Rider Features dtypes:\n", rider_features[["race_date", "race_no", "track"]].dtypes)
        print("Rider Features sample:\n", rider_features[["race_date", "race_no", "track"]].head())
        
        # Check for track name mismatches
        res_tracks = set(dataset["track"].unique())
        feat_tracks = set(rider_features["track"].unique())
        print(f"Common tracks: {len(res_tracks.intersection(feat_tracks))}")
        print(f"Results tracks (first 5): {list(res_tracks)[:5]}")
        print(f"Feature tracks (first 5): {list(feat_tracks)[:5]}")

        dataset = dataset.merge(
            rider_features.drop(columns=["keirin_cd"], errors="ignore"),
            on=["race_date", "race_no", "track"],
            how="inner",
        )
        print(f"Final merged dataset: {len(dataset)} rows")
    except Exception:
        traceback.print_exc()
        return

if __name__ == "__main__":
    debug_loading()
