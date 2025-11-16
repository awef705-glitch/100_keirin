#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Build enhanced training dataset with track/category historical tendencies"""

import pandas as pd
from pathlib import Path
import sys

def clean_payout(x):
    """Convert payout string to number"""
    if pd.isna(x):
        return None
    s = str(x).replace('円', '').replace(',', '').strip()
    try:
        return int(s)
    except:
        return None

def main():
    print("Loading results data...")
    results_path = Path("/tmp/keirin_data/keirin_results.csv")
    if not results_path.exists():
        print(f"Error: {results_path} not found")
        sys.exit(1)

    df = pd.read_csv(results_path)
    print(f"Loaded {len(df)} races")

    # Clean payout
    df['trifecta_payout_num'] = df['trifecta_payout'].apply(clean_payout)
    df = df[df['trifecta_payout_num'].notna()].copy()
    df['target_high_payout'] = (df['trifecta_payout_num'] >= 10000).astype(int)

    print(f"Valid races: {len(df)}")
    print(f"High payout rate: {df['target_high_payout'].mean():.1%}")

    # Calculate track historical high payout rate
    print("\nCalculating track historical rates...")
    track_stats = df.groupby('track').agg({
        'target_high_payout': ['mean', 'count'],
        'trifecta_payout_num': ['mean', 'median']
    })
    track_stats.columns = ['track_high_payout_rate', 'track_race_count',
                           'track_avg_payout', 'track_median_payout']

    df = df.merge(track_stats, left_on='track', right_index=True, how='left')

    # Calculate category historical high payout rate
    print("Calculating category historical rates...")
    cat_stats = df.groupby('category').agg({
        'target_high_payout': ['mean', 'count'],
        'trifecta_payout_num': ['mean', 'median']
    })
    cat_stats.columns = ['category_high_payout_rate', 'category_race_count',
                         'category_avg_payout', 'category_median_payout']

    df = df.merge(cat_stats, left_on='category', right_index=True, how='left')

    # Calculate grade historical high payout rate
    print("Calculating grade historical rates...")
    grade_stats = df.groupby('grade').agg({
        'target_high_payout': ['mean', 'count'],
        'trifecta_payout_num': ['mean', 'median']
    })
    grade_stats.columns = ['grade_high_payout_rate', 'grade_race_count',
                          'grade_avg_payout', 'grade_median_payout']

    df = df.merge(grade_stats, left_on='grade', right_index=True, how='left')

    # Race number effect (later races tend to be more unpredictable)
    df['race_no_int'] = df['race_no'].str.replace('R', '').astype(int)
    df['is_late_race'] = (df['race_no_int'] >= 9).astype(int)
    df['is_early_race'] = (df['race_no_int'] <= 3).astype(int)
    df['is_main_race'] = (df['race_no_int'].between(10, 12)).astype(int)

    # Save enhanced dataset
    output_dir = Path("analysis/model_outputs")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "enhanced_results_for_training.csv"

    # Select columns
    key_cols = ['race_date', 'keirin_cd', 'track', 'race_no', 'race_no_int',
                'category', 'grade', 'trifecta_payout_num', 'target_high_payout']
    feature_cols = ['track_high_payout_rate', 'track_race_count', 'track_avg_payout', 'track_median_payout',
                    'category_high_payout_rate', 'category_race_count', 'category_avg_payout', 'category_median_payout',
                    'grade_high_payout_rate', 'grade_race_count', 'grade_avg_payout', 'grade_median_payout',
                    'is_late_race', 'is_early_race', 'is_main_race']

    df_out = df[key_cols + feature_cols]
    df_out.to_csv(output_path, index=False)

    print(f"\nSaved enhanced dataset to: {output_path}")
    print(f"Shape: {df_out.shape}")
    print(f"\nNew feature statistics:")
    print(df_out[feature_cols].describe())

    # Show top/bottom tracks
    print(f"\n\nTop 10 tracks by high payout rate:")
    track_summary = df.groupby('track')[['track_high_payout_rate']].first().sort_values('track_high_payout_rate', ascending=False)
    print(track_summary.head(10))

    print(f"\n\nTop categories by high payout rate:")
    cat_summary = df.groupby('category')[['category_high_payout_rate']].first().sort_values('category_high_payout_rate', ascending=False)
    print(cat_summary.head(15))

if __name__ == "__main__":
    main()
