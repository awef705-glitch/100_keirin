#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Generate a master list of riders from historical race data.
Aggregates stats including:
- Average Score (heikinTokuten)
- Style (kyakusitu)
- Grade (kyuhan)
- Prefecture (huKen)
- B Count (backCnt) - NEW
"""

import json
import glob
from pathlib import Path
import pandas as pd
import numpy as np

DATA_DIR = Path("data")
OUTPUT_DIR = Path("analysis/model_outputs")
OUTPUT_PATH = OUTPUT_DIR / "rider_master.json"

def safe_float(x):
    try:
        return float(x)
    except (ValueError, TypeError):
        return 0.0

def safe_int(x):
    try:
        return int(float(x))
    except (ValueError, TypeError):
        return 0

def main():
    print("Scanning for race detail entries...")
    files = glob.glob(str(DATA_DIR / "keirin_race_detail_entries_*.csv"))
    
    if not files:
        print("No data files found.")
        return

    print(f"Found {len(files)} files. Loading data...")
    
    dfs = []
    for f in files:
        try:
            # Read only necessary columns to save memory
            df = pd.read_csv(f, usecols=[
                "sensyuName", "huKen", "kyuhan", "kyakusitu", 
                "heikinTokuten", "backCnt", "nigeCnt", "makuriCnt", "sasiCnt", "markCnt"
            ], encoding="utf-8")
            dfs.append(df)
        except Exception as e:
            # Try cp932 if utf-8 fails
            try:
                df = pd.read_csv(f, usecols=[
                    "sensyuName", "huKen", "kyuhan", "kyakusitu", 
                    "heikinTokuten", "backCnt", "nigeCnt", "makuriCnt", "sasiCnt", "markCnt"
                ], encoding="cp932")
                dfs.append(df)
            except Exception as e2:
                print(f"Skipping {f}: {e2}")

    if not dfs:
        print("No data loaded.")
        return

    full_df = pd.concat(dfs, ignore_index=True)
    print(f"Total rows: {len(full_df)}")

    # Clean data
    full_df["sensyuName"] = full_df["sensyuName"].astype(str).str.replace(r"\s+", "", regex=True)
    full_df = full_df[full_df["sensyuName"] != "nan"]
    
    # Convert numeric columns
    numeric_cols = ["heikinTokuten", "backCnt", "nigeCnt", "makuriCnt", "sasiCnt", "markCnt"]
    for col in numeric_cols:
        full_df[col] = full_df[col].apply(safe_float)

    print("Aggregating rider stats...")
    
    # Group by rider name
    # Note: Using name as ID. In a perfect world we'd use ID, but name is what we have in the UI input.
    # We take the most recent (or mode) for categorical, and mean/max for numeric.
    
    # For categorical, we'll just take the most frequent value
    categorical_agg = full_df.groupby("sensyuName")[["huKen", "kyuhan", "kyakusitu"]].agg(
        lambda x: x.mode().iloc[0] if not x.mode().empty else ""
    )
    
    # For numeric, we take the mean (representing their average performance/stats over the period)
    # For B count, taking the max might be misleading if it fluctuates, but mean gives a "typical" value.
    # Actually, B count in the race card is usually "last 4 months". 
    # So the average of "last 4 months B count" over a year is a decent proxy for their general tendency.
    numeric_agg = full_df.groupby("sensyuName")[numeric_cols].mean()
    
    # Merge
    master_df = pd.concat([categorical_agg, numeric_agg], axis=1)
    
    # Format for JSON
    riders_list = []
    for name, row in master_df.iterrows():
        riders_list.append({
            "name": name,
            "prefecture": row["huKen"],
            "grade": row["kyuhan"],
            "style": row["kyakusitu"],
            "avg_score": round(row["heikinTokuten"], 2),
            "back_count": round(row["backCnt"], 1),
            "nige_count": round(row["nigeCnt"], 1),
            "makuri_count": round(row["makuriCnt"], 1),
            "sasi_count": round(row["sasiCnt"], 1),
            "mark_count": round(row["markCnt"], 1)
        })
    
    # Sort by score descending (popular riders first)
    riders_list.sort(key=lambda x: x["avg_score"], reverse=True)
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(riders_list, f, ensure_ascii=False, indent=2)
        
    print(f"Saved {len(riders_list)} riders to {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
