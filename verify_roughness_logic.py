import pandas as pd
import numpy as np
import sys
from pathlib import Path
from typing import Dict, Any, List

# Add current directory to path
sys.path.append(str(Path.cwd()))

from analysis import prerace_model

def _safe_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0

def _normalise_style(value: Any) -> str:
    if value is None:
        return "unknown"
    key = str(value).strip()
    if not key:
        return "unknown"
    direct_map = {
        "先行": "nige",
        "逃げ": "nige",
        "追込": "tsui",
        "追い込み": "tsui",
        "自在": "ryo",
        "両": "ryo",
    }
    if key in direct_map:
        return direct_map[key]
    # Simple check for aliases
    if "逃" in key or "先" in key or "捲" in key:
        return "nige"
    if "追" in key or "差" in key or "マーク" in key:
        return "tsui"
    if "両" in key or "自在" in key:
        return "ryo"
    return "unknown"

def aggregate_features(entries_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate rider entries into race-level features."""
    
    # Pre-process columns
    frame = entries_df.copy()
    frame["score"] = frame["heikinTokuten"].apply(_safe_float)
    frame["nige_count"] = frame["nigeCnt"].apply(_safe_float)
    frame["makuri_count"] = frame["makuriCnt"].apply(_safe_float)
    frame["style_norm"] = frame["kyakusitu"].apply(_normalise_style)
    
    rows = []
    
    # Group by race
    for (race_date, track, race_no), group in frame.groupby(["race_date", "track", "race_no"]):
        # Use prerace_model's summarizer
        # We need to adapt the group dataframe to match what _summarise_riders expects
        # It expects columns: score, style_norm, etc.
        
        # Create a clean rider frame for the summarizer
        rider_frame = group[["score", "style_norm", "nige_count", "makuri_count"]].copy()
        # Add dummy columns if needed by _summarise_riders (it checks for them)
        rider_frame["back_count"] = 0 # Not used in roughness score directly but might be in summarizer
        rider_frame["sasi_count"] = 0
        rider_frame["mark_count"] = 0
        rider_frame["grade_norm"] = ""
        rider_frame["prefecture_norm"] = ""
        
        bundle = prerace_model._summarise_riders(rider_frame)
        
        row = {
            "race_date": race_date,
            "track": track,
            "race_no": race_no,
        }
        row.update(bundle.features)
        rows.append(row)
        
    return pd.DataFrame(rows)

def main():
    print("Loading data...")
    entries_path = Path("data/keirin_race_detail_entries_20240101_20251004.csv")
    results_path = Path("data/keirin_results_20240101_20251004.csv")
    
    # Load data
    entries = pd.read_csv(entries_path)
    results = pd.read_csv(results_path)
    
    # Convert dates
    entries["race_date"] = pd.to_numeric(entries["race_date"], errors="coerce")
    results["race_date"] = pd.to_numeric(results["race_date"], errors="coerce")
    
    # Filter for last 3 months (approx) to get a good sample size
    # 2025-07-01 to 2025-10-07
    start_date = 20250701
    end_date = 20251007
    
    print(f"Filtering data from {start_date} to {end_date}...")
    entries = entries[(entries["race_date"] >= start_date) & (entries["race_date"] <= end_date)]
    results = results[(results["race_date"] >= start_date) & (results["race_date"] <= end_date)]
    
    print(f"Processing {len(entries)} entries...")
    
    # Aggregate features
    features_df = aggregate_features(entries)
    
    # Prepare results for merge
    results["race_no"] = results["race_no"].astype(str).str.replace("R", "", regex=False)
    results["race_no"] = pd.to_numeric(results["race_no"], errors="coerce")
    features_df["race_no"] = pd.to_numeric(features_df["race_no"], errors="coerce")
    
    # Merge
    merged = pd.merge(features_df, results, on=["race_date", "track", "race_no"], how="inner")
    print(f"Analyzed {len(merged)} races.")
    
    # Calculate Roughness Score
    merged["roughness_score"] = merged.apply(lambda row: prerace_model.calculate_roughness_score(row, {}), axis=1)
    
    # Clean payout data
    def clean_payout(x):
        try:
            if pd.isna(x): return 0
            return float(str(x).replace(",", "").replace("円", ""))
        except:
            return 0
            
    merged["payout"] = merged["trifecta_payout"].apply(clean_payout)
    merged = merged[merged["payout"] > 0] # Remove invalid payouts
    
    # --- STATISTICAL ANALYSIS ---
    print("\n" + "="*80)
    print("ROUGHNESS SCORE VALIDATION REPORT")
    print("="*80)
    
    # Correlation
    corr = merged["roughness_score"].corr(merged["payout"])
    print(f"Correlation (Score vs Payout): {corr:.4f}")
    
    # Binned Analysis
    bins = [0, 20, 40, 60, 80, 100]
    labels = ["0-20 (Stable)", "20-40 (Low)", "40-60 (Mid)", "60-80 (High)", "80-100 (Chaos)"]
    merged["score_bin"] = pd.cut(merged["roughness_score"], bins=bins, labels=labels, include_lowest=True)
    
    stats = merged.groupby("score_bin", observed=True)["payout"].agg(["count", "mean", "median", "min", "max"])
    pd.options.display.float_format = '{:,.0f}'.format
    print("\n--- Payout Stats by Score Range ---")
    print(stats)
    
    # High Payout Rate (>10,000 yen) by Bin
    merged["is_high_payout"] = merged["payout"] >= 10000
    high_payout_rate = merged.groupby("score_bin", observed=True)["is_high_payout"].mean() * 100
    print("\n--- High Payout Rate (>10,000 yen) by Score Range ---")
    print(high_payout_rate.map('{:.1f}%'.format))

    # Show top examples for each bin
    print("\n--- Representative Examples ---")
    for label in labels:
        subset = merged[merged["score_bin"] == label]
        if not subset.empty:
            # Pick a random sample or median payout sample
            sample = subset.sort_values("payout").iloc[len(subset)//2]
            print(f"\n[{label}] Score: {sample['roughness_score']:.1f} -> Payout: {sample['payout']:,.0f} Yen")
            print(f"  Race: {int(sample['race_date'])} {sample['track']} {int(sample['race_no'])}R")
            print(f"  CV: {sample.get('score_cv', 0):.4f}, Gap: {sample.get('estimated_favorite_gap', 0):.1f}")

if __name__ == "__main__":
    main()

def print_race_details(race):
    print(f"\nRace: {int(race['race_date'])} {race['track']} {int(race['race_no'])}R")
    print(f"  > ROUGHNESS SCORE: {race['roughness_score']:.1f} / 100")
    print(f"  > Actual Payout (3T): {race.get('trifecta_payout', 'N/A')} Yen")
    print(f"  > Key Factors:")
    print(f"    - Score CV (Variation): {race.get('score_cv', 0):.4f} (Low=Stable, High=Chaos)")
    print(f"    - Favorite Gap: {race.get('estimated_favorite_gap', 0):.1f} (High=Stable)")
    print(f"    - Dominant Line Ratio: {race.get('dominant_line_ratio', 0):.2f}")
    print(f"    - Nige Count: {race.get('style_nige_count', 0)}")
