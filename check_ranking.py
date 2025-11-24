import pandas as pd
import numpy as np
from pathlib import Path
import sys

sys.path.append(str(Path.cwd()))

from analysis import prerace_model
from analysis import betting_suggestions

def check_ranking_accuracy():
    """Check if rider ranking (not 3-連単 combination) is accurate"""
    print("="*70)
    print("Checking Rider Ranking Accuracy")
    print("="*70)
    
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
    
    valid_results = results.dropna(subset=["pos1_car_no", "pos2_car_no", "pos3_car_no", "race_date", "race_no"])
    valid_results["race_date"] = valid_results["race_date"].astype(int)
    valid_results["race_no"] = valid_results["race_no"].astype(int)
    
    # Sample 20 races
    sample_races = valid_results.sample(n=20, random_state=42)
    
    top1_correct = 0
    top2_correct = 0
    top3_correct = 0
    
    print(f"\\nAnalyzing {len(sample_races)} races...\\n")
    
    for idx, row in sample_races.iterrows():
        race_date = int(row["race_date"])
        race_no = int(row["race_no"])
        track_name = row.get("track", "")
        
        race_entries = entries[
            (entries["race_date"] == race_date) & 
            (entries["race_no"] == race_no)
        ]
        
        if len(race_entries["track"].unique()) > 1:
             race_entries = race_entries[race_entries["track"] == track_name]
             
        if len(race_entries) < 5:
            continue

        # Build riders list
        riders_for_suggestions = []
        for _, raw_row in race_entries.iterrows():
             riders_for_suggestions.append({
                "name": raw_row.get("name", ""),
                "prefecture": raw_row.get("huKen", ""),
                "grade": raw_row.get("kyuhan", ""),
                "style": raw_row.get("kyakusitu", ""),
                "avg_score": float(raw_row.get("heikinTokuten", 0) or 0),
                "back_count": float(raw_row.get("backCnt", 0) or 0),
                "nige_count": float(raw_row.get("nigeCnt", 0) or 0),
                "makuri_count": float(raw_row.get("makuriCnt", 0) or 0),
                "sasi_count": float(raw_row.get("sasiCnt", 0) or 0),
                "mark_count": float(raw_row.get("markCnt", 0) or 0)
            })

        # Rank riders
        ranked = betting_suggestions.rank_riders(riders_for_suggestions)
        predicted_top3 = [r[0] for r in ranked[:3]]
        
        # Actual result
        actual_1 = int(row["pos1_car_no"])
        actual_2 = int(row["pos2_car_no"])
        actual_3 = int(row["pos3_car_no"])
        actual_top3 = {actual_1, actual_2, actual_3}
        
        # Check accuracy
        if actual_1 == predicted_top3[0]:
            top1_correct += 1
        
        if actual_1 in predicted_top3[:2] or actual_2 in predicted_top3[:2]:
            top2_correct += 1
            
        overlap = len(set(predicted_top3) & actual_top3)
        if overlap >= 2:
            top3_correct += 1
        
        print(f"Race: {race_date} R{race_no}")
        print(f"  Predicted Top3: {predicted_top3}")
        print(f"  Actual Top3: {actual_1}-{actual_2}-{actual_3}")
        print(f"  Overlap: {overlap}/3")
        print()

    print("="*70)
    print("Ranking Accuracy Results")
    print("="*70)
    print(f"1st Place Prediction: {top1_correct}/{len(sample_races)} ({top1_correct/len(sample_races)*100:.1f}%)")
    print(f"Top-2 Contains Winner/2nd: {top2_correct}/{len(sample_races)} ({top2_correct/len(sample_races)*100:.1f}%)")
    print(f"2+ Riders in Top-3: {top3_correct}/{len(sample_races)} ({top3_correct/len(sample_races)*100:.1f}%)")
    print("="*70)

if __name__ == "__main__":
    check_ranking_accuracy()
