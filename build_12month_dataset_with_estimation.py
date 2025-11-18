#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
12ヶ月分のトレーニングデータセットを選手推定データで構築

Strategy:
1. Load 48,700 races results
2. Filter to 2024/01/01 - 2024/12/31 (12 months)
3. Estimate 9 riders per race from grade/category
4. Calculate features
5. Save dataset
"""

import pandas as pd
import numpy as np
import random
from pathlib import Path

print("="*80)
print("【12ヶ月分トレーニングデータセット構築】")
print("="*80)

# Load results
results = pd.read_csv('data/processed_results_48700.csv')
print(f"\nTotal races: {len(results):,}")
print(f"Date range: {results['race_date'].min()} to {results['race_date'].max()}")

# Filter to 12 months (2024/01/01 - 2024/12/31)
results_12m = results[(results['race_date'] >= 20240101) & (results['race_date'] <= 20241231)].copy()
print(f"\n12-month races (2024): {len(results_12m):,}")
print(f"High payout: {results_12m['high_payout_flag'].sum()} ({results_12m['high_payout_flag'].mean()*100:.1f}%)")

# Rider estimation logic from commit f128ebb
def estimate_riders_from_category_grade(category, grade, race_no, keirin_cd):
    """Estimate rider data from category and grade"""
    riders = []
    random.seed(hash(f"{category}_{grade}_{race_no}_{keirin_cd}"))

    # Grade-based settings
    if 'GP' in str(grade):
        base_score = 117.0
        score_range = 3.0
        grades = ['SS'] * 9
        styles = ['追'] * 6 + ['逃'] * 3
    elif 'G1' in str(grade):
        base_score = 115.0
        score_range = 4.0
        grades = ['SS'] * 3 + ['S1'] * 6
        styles = ['追'] * 6 + ['逃'] * 3
    elif 'G2' in str(grade):
        base_score = 114.0
        score_range = 4.5
        grades = ['S1'] * 7 + ['S2'] * 2
        styles = ['追'] * 6 + ['逃'] * 3
    elif 'G3' in str(grade):
        if 'Ｓ級' in str(category):
            base_score = 112.0
            score_range = 3.2
            grades = ['S1'] * 3 + ['S2'] * 6
            styles = ['追'] * 6 + ['逃'] * 3
        else:
            base_score = 92.0
            score_range = 4.5
            grades = ['A1'] * 4 + ['A2'] * 5
            styles = ['追'] * 6 + ['逃'] * 3
    elif 'F1' in str(grade):
        if 'Ｓ級' in str(category):
            base_score = 109.0
            score_range = 4.0
            grades = ['S2'] * 9
            styles = ['追'] * 6 + ['逃'] * 3
        else:
            base_score = 90.0
            score_range = 5.0
            grades = ['A1'] * 5 + ['A2'] * 4
            styles = ['追'] * 6 + ['逃'] * 3
    elif 'F2' in str(grade):
        if 'チャレンジ' in str(category):
            base_score = 82.0
            score_range = 6.0
            grades = ['A3'] * 9
            styles = ['追'] * 6 + ['逃'] * 3
        else:
            base_score = 88.0
            score_range = 5.5
            grades = ['A2'] * 5 + ['A3'] * 4
            styles = ['追'] * 6 + ['逃'] * 3
    elif 'L' in str(grade):
        base_score = 50.0
        score_range = 5.0
        grades = ['L1'] * 9
        styles = ['両'] * 9
    else:
        # Default
        base_score = 90.0
        score_range = 5.0
        grades = ['A2'] * 9
        styles = ['追'] * 6 + ['逃'] * 3

    # Generate 9 riders
    for car_no in range(1, 10):
        score = base_score + random.uniform(-score_range, score_range)
        riders.append({
            'car_no': car_no,
            'kyuhan': grades[car_no-1],
            'kyakusitu': styles[car_no-1],
            'heikinTokuten': round(score, 2),
            'nigeCnt': random.randint(0, 5) if styles[car_no-1] == '逃' else 0,
            'makuriCnt': random.randint(0, 5) if styles[car_no-1] == '捲' else 0,
            'sasiCnt': random.randint(0, 5) if styles[car_no-1] == '差' else 0,
            'markCnt': random.randint(0, 5) if styles[car_no-1] == '追' else 0,
            'backCnt': random.randint(0, 3),
        })

    return riders

# Build dataset
print("\nBuilding dataset...")
all_rows = []

for idx, race in results_12m.iterrows():
    if idx % 1000 == 0:
        print(f"  Processing race {idx}/{len(results_12m)}...")

    # Estimate riders
    riders = estimate_riders_from_category_grade(
        race['category'],
        race['grade'],
        race['race_no'],
        race['keirin_cd']
    )

    # Calculate race-level features
    scores = [r['heikinTokuten'] for r in riders]
    entry_count = len(riders)

    for rider in riders:
        row = {
            # Race info
            'race_date': race['race_date'],
            'track': race['track'],
            'race_no': race['race_no'],
            'keirin_cd': race['keirin_cd'],
            'grade': race['grade'],
            'category': race['category'],
            'meeting_icon': race.get('meeting_icon', 0),

            # Target
            'trifecta_payout_value': race['trifecta_payout_value'],
            'high_payout_flag': race['high_payout_flag'],

            # Rider info (estimated)
            'car_no': rider['car_no'],
            'kyuhan': rider['kyuhan'],
            'kyakusitu': rider['kyakusitu'],
            'heikinTokuten': rider['heikinTokuten'],
            'nigeCnt': rider['nigeCnt'],
            'makuriCnt': rider['makuriCnt'],
            'sasiCnt': rider['sasiCnt'],
            'markCnt': rider['markCnt'],
            'backCnt': rider['backCnt'],

            # Race-level features
            'entry_count': entry_count,
            'score_mean': np.mean(scores),
            'score_std': np.std(scores),
            'score_min': np.min(scores),
            'score_max': np.max(scores),
            'score_cv': np.std(scores) / np.mean(scores) if np.mean(scores) > 0 else 0,

            # Rider relative features
            'score_diff': rider['heikinTokuten'] - np.mean(scores),
            'score_z': (rider['heikinTokuten'] - np.mean(scores)) / np.std(scores) if np.std(scores) > 0 else 0,
        }

        all_rows.append(row)

df = pd.DataFrame(all_rows)
print(f"\nDataset created: {len(df):,} rows ({len(results_12m):,} races × ~9 riders)")
print(f"Date range: {df['race_date'].min()} to {df['race_date'].max()}")
print(f"Unique races: {df[['race_date', 'track', 'race_no']].drop_duplicates().shape[0]:,}")

# Save
output_path = Path('data/keirin_training_dataset_12months_estimated.csv')
df.to_csv(output_path, index=False)
print(f"\nSaved to: {output_path}")
print(f"File size: {output_path.stat().st_size / 1024 / 1024:.1f} MB")

# Summary
print("\n" + "="*80)
print("【サマリー】")
print("="*80)
print(f"期間: 2024/01/01 - 2024/12/31 (12ヶ月)")
print(f"総レース数: {len(results_12m):,}")
print(f"総エントリ数: {len(df):,}")
print(f"高配当レース: {results_12m['high_payout_flag'].sum():,} ({results_12m['high_payout_flag'].mean()*100:.1f}%)")
print(f"特徴量数: {len(df.columns)}")
print("\nNext: Run train_clean_model.py with this dataset to apply V4.0 features and train")
