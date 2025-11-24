import pandas as pd
import numpy as np
import json
from pathlib import Path
import sys

# Add current directory to path
sys.path.append(str(Path.cwd()))

from analysis import prerace_model
from analysis import betting_suggestions

def verify_lightgbm_accuracy():
    print("=" * 70)
    print("LightGBM Model Accuracy Verification")
    print("=" * 70)

    # Load LightGBM model
    print("Loading LightGBM model...")
    try:
        model = prerace_model.load_model()
        metadata = prerace_model.load_metadata()
        print(f"[OK] LightGBM model loaded successfully")
        print(f"     Features: {len(metadata['feature_columns'])}")
    except Exception as e:
        print(f"[ERROR] Failed to load LightGBM model: {e}")
        return

    # Load Data
    print("\nLoading data...")
    results_path = Path("data/keirin_results_20240101_20251004.csv")
    entries_path = Path("data/keirin_race_detail_entries_20240101_20251004.csv")
    prerace_path = Path("data/keirin_prerace_20240101_20251004.csv")
    
    results = pd.read_csv(results_path)
    results = results.loc[:, ~results.columns.duplicated()]
    
    if "race_no_num" in results.columns:
        if "race_no" in results.columns:
            results = results.drop(columns=["race_no"])
        results = results.rename(columns={"race_no_num": "race_no"})
        
    results = results.loc[:, ~results.columns.duplicated()]
    
    entries = pd.read_csv(entries_path)
    entries = entries.loc[:, ~entries.columns.duplicated()]
    
    # Preprocess
    entries["race_date"] = pd.to_numeric(entries["race_date"], errors="coerce")
    entries["race_no"] = pd.to_numeric(entries["race_no"], errors="coerce")
    entries = entries.dropna(subset=["race_date", "race_no"])
    entries["race_date"] = entries["race_date"].astype(int)
    entries["race_no"] = entries["race_no"].astype(int)
    
    if "keirin_cd" in entries.columns:
        entries["keirin_cd"] = entries["keirin_cd"].astype(str).str.zfill(2)
    
    results["race_date"] = pd.to_numeric(results["race_date"], errors="coerce")
    results["race_no"] = pd.to_numeric(results["race_no"], errors="coerce")
    
    valid_results = results.dropna(subset=["trifecta_payout", "pos1_car_no", "pos2_car_no", "pos3_car_no", "race_date", "race_no"])
    valid_results["race_date"] = valid_results["race_date"].astype(int)
    valid_results["race_no"] = valid_results["race_no"].astype(int)
    
    if len(valid_results) == 0:
        print("No valid results found!")
        return

    # Sample 20 races
    n_samples = min(20, len(valid_results))
    sample_races = valid_results.sample(n=n_samples, random_state=42)
    
    hits = 0
    high_payout_hits = 0
    total_high_payouts = 0
    
    print(f"\nVerifying {len(sample_races)} races with LightGBM model...\n")
    
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

        # Prepare rider data
        rider_frame = prerace_model._prepare_rider_frame_from_entries(race_entries)
        
        # Build race_info with calendar features
        race_info = {
            
            # Predict with LightGBM model
            prob = prerace_model.predict_probability(feature_frame, model, metadata, {"track": track_name, "category": ""})
        except Exception as e:
            print(f"Error predicting {race_date} {track_name} R{race_no}: {e}")
            continue
        
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
        
        print(f"Race: {race_date} {track_name} R{race_no}")
        print(f"  AI Probability: {prob:.2%} | Payout: Y{payout:,.0f} | {'[HIT]' if hit else '[MISS]'}")
        if hit:
            print(f"  -> Winning combo {actual_combo} found in top {len(suggested_combos)} suggestions")

    print("\n" + "=" * 70)
    print("LightGBM Model Results")
    print("=" * 70)
    print(f"Total Races Verified: {len(sample_races)}")
    print(f"Overall Hit Rate: {hits}/{len(sample_races)} ({hits/len(sample_races)*100:.1f}%)")
    print(f"High Payout Races (10k+): {total_high_payouts}")
    if total_high_payouts > 0:
        print(f"High Payout Hit Rate: {high_payout_hits}/{total_high_payouts} ({high_payout_hits/total_high_payouts*100:.1f}%)")
    else:
        print(f"High Payout Hit Rate: 0/0 (N/A)")
    print("=" * 70)
    
    # Comparison
    print("\n" + "=" * 70)
    print("Comparison: Rule-Based vs LightGBM")
    print("=" * 70)
    print("Rule-Based Model (Previous Test):")
    print("  Overall Hit Rate: 15.0% (3/20)")
    print("  High Payout Hit Rate: 0.0% (0/4)")
    print(f"\nLightGBM Model (Current Test):")
    print(f"  Overall Hit Rate: {hits/len(sample_races)*100:.1f}% ({hits}/{len(sample_races)})")
    if total_high_payouts > 0:
        print(f"  High Payout Hit Rate: {high_payout_hits/total_high_payouts*100:.1f}% ({high_payout_hits}/{total_high_payouts})")
    else:
        print(f"  High Payout Hit Rate: N/A")
    print("=" * 70)

if __name__ == "__main__":
    verify_lightgbm_accuracy()
