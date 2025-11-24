import pandas as pd
import numpy as np
import json
from pathlib import Path
import sys
import random

# Add current directory to path
sys.path.append(str(Path.cwd()))

from analysis import prerace_model
from analysis import betting_suggestions

def verify_accuracy():
    print("=" * 70)
    print("Model Accuracy Verification (Rule-Based)")
    print("=" * 70)

    # 1. Load Data
    print("Loading data...")
    results_path = Path("data/keirin_results_20240101_20251004.csv")
    entries_path = Path("data/keirin_race_detail_entries_20240101_20251004.csv")
    
    results = pd.read_csv(results_path)
    results = results.loc[:, ~results.columns.duplicated()]
    
    if "race_no_num" in results.columns:
        if "race_no" in results.columns:
            results = results.drop(columns=["race_no"])
        results = results.rename(columns={"race_no_num": "race_no"})
        
    results = results.loc[:, ~results.columns.duplicated()]
    
    entries = pd.read_csv(entries_path)
    entries = entries.loc[:, ~entries.columns.duplicated()]
    
    # Preprocess entries
    entries["race_date"] = pd.to_numeric(entries["race_date"], errors="coerce")
    entries["race_no"] = pd.to_numeric(entries["race_no"], errors="coerce")
    entries = entries.dropna(subset=["race_date", "race_no"])
    entries["race_date"] = entries["race_date"].astype(int)
    entries["race_no"] = entries["race_no"].astype(int)
    
    if "keirin_cd" in entries.columns:
        entries["keirin_cd"] = entries["keirin_cd"].astype(str).str.zfill(2)
    
    # 2. Select Random Races
    # Filter results that have actual payout info and car numbers
    results["race_date"] = pd.to_numeric(results["race_date"], errors="coerce")
    results["race_no"] = pd.to_numeric(results["race_no"], errors="coerce")
    
    valid_results = results.dropna(subset=["trifecta_payout", "pos1_car_no", "pos2_car_no", "pos3_car_no", "race_date", "race_no"])
    valid_results["race_date"] = valid_results["race_date"].astype(int)
    valid_results["race_no"] = valid_results["race_no"].astype(int)
    
    if len(valid_results) == 0:
        print("No valid results found! Check column names or data types.")
        return

    # Sample 20 races (or less if not enough data)
    n_samples = min(20, len(valid_results))
    sample_races = valid_results.sample(n=n_samples, random_state=42)
    
    hits = 0
    high_payout_hits = 0
    total_high_payouts = 0
    
    print(f"\nVerifying {len(sample_races)} races...")
    
    for idx, row in sample_races.iterrows():
        race_date = int(row["race_date"])
        race_no = int(row["race_no"])
        keirin_cd = str(row["keirin_cd"]).replace(".0", "").zfill(2)
        track_name = row.get("track", "")
        
        race_entries = entries[
            (entries["race_date"] == race_date) & 
            (entries["race_no"] == race_no)
        ]
        
        if len(race_entries["track"].unique()) > 1:
             race_entries = race_entries[race_entries["track"] == track_name]
             
        if len(race_entries) < 5:
            print(f"Skipping {race_date} {track_name} R{race_no}: Not enough rider data")
            continue

        # Prepare rider data using prerace_model's function (renames columns)
        rider_frame = prerace_model._prepare_rider_frame_from_entries(race_entries)
        
        # Build riders list for betting suggestions using original entries (for grade strings)
        riders_for_suggestions = []
        for _, raw_row in race_entries.iterrows():
             riders_for_suggestions.append({
                "name": raw_row.get("name", ""),
                "prefecture": raw_row.get("huKen", ""),
                "grade": raw_row.get("kyuhan", ""),
                "style": raw_row.get("kyakusitu", ""),
                "avg_score": float(raw_row.get("heikinTokuten", 0) or 0),
                "back_count": float(raw_row.get("backCnt", 0) or 0)
            })

        race_info = {
            "riders": riders_for_suggestions,
            "race_date": str(race_date),
            "track": track_name,
            "race_no": race_no
        }
        
        # Calculate Probability (Rule-Based)
        summary_features = prerace_model._summarise_riders(rider_frame)
        # Add race level features to the features dict
        summary_features.features["entry_count"] = len(race_entries)
        summary_features.features["is_final_day"] = 0 # Mock
        
        # Metadata needed for _fallback_probability
        metadata = {"feature_columns": []} 
        
        prob = prerace_model._fallback_probability(summary_features.features, metadata)
        
        # Generate Suggestions
        suggestions_data = betting_suggestions.generate_betting_suggestions(
            race_info=race_info,
            probability=prob,
            confidence="Test"
        )
        
        # Check Result
        actual_1 = int(row["pos1_car_no"])
        actual_2 = int(row["pos2_car_no"])
        actual_3 = int(row["pos3_car_no"])
        actual_combo = f"{actual_1}-{actual_2}-{actual_3}"
        
        try:
            payout_str = str(row["trifecta_payout"]).replace(",", "").replace("円", "").strip()
            payout = float(payout_str)
        except ValueError:
            payout = 0.0
        
        is_high_payout = payout >= 10000
        if is_high_payout:
            total_high_payouts += 1
            
        # Check if actual combo is in suggestions
        suggested_combos = [s["combination"] for s in suggestions_data["suggestions"]]
        hit = actual_combo in suggested_combos
        
        if hit:
            hits += 1
            if is_high_payout:
                high_payout_hits += 1
        
        print(f"Race: {race_date} {track_name} R{race_no} | Prob: {prob:.2f} | Payout: {payout} | Hit: {'YES' if hit else 'NO'}")
        if hit:
            print(f"  -> Winning Combo: {actual_combo} found in top {len(suggested_combos)} suggestions")

    print("\n" + "=" * 70)
    print("Verification Results")
    print("=" * 70)
    print(f"Total Races Verified: {len(sample_races)}")
    print(f"Total Hits (Any Payout): {hits} ({hits/len(sample_races)*100:.1f}%)")
    print(f"High Payout Races (>10k): {total_high_payouts}")
    print(f"High Payout Hits: {high_payout_hits} ({high_payout_hits/total_high_payouts*100:.1f}% of high payout races)" if total_high_payouts > 0 else "High Payout Hits: 0 (0%)")
    print("=" * 70)

if __name__ == "__main__":
    verify_accuracy()
