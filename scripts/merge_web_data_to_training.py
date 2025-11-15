#!/usr/bin/env python3
"""
Web検索データを訓練データセットにマージしてモデルを再訓練
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

print("="*70)
print("MERGING WEB SEARCH DATA WITH TRAINING DATASET")
print("="*70)
print()

# Load existing training data
training_file = Path('analysis/model_outputs/enhanced_results_for_training.csv')
web_data_file = Path('data/web_search_races.csv')

if not training_file.exists():
    print(f"ERROR: Training file not found: {training_file}")
    sys.exit(1)

print(f"Loading existing training data: {training_file}")
df_training = pd.read_csv(training_file)
print(f"  Existing races: {len(df_training):,}")
print(f"  Columns: {len(df_training.columns)}")
print()

# Load web search data
if not web_data_file.exists():
    print(f"ERROR: Web data file not found: {web_data_file}")
    sys.exit(1)

print(f"Loading web search data: {web_data_file}")
df_web = pd.read_csv(web_data_file)
print(f"  Web search races: {len(df_web)}")
print()

# Show web data
print("Web Search Data:")
print(df_web[['race_date', 'track', 'grade', 'trifecta_payout_num', 'target_high_payout']])
print()

# Check columns
training_cols = set(df_training.columns)
web_cols = set(df_web.columns)

common_cols = training_cols & web_cols
only_training = training_cols - web_cols
only_web = web_cols - training_cols

print(f"Common columns: {len(common_cols)}")
print(f"Only in training: {len(only_training)}")
if only_training:
    print(f"  {list(only_training)[:10]}")
print(f"Only in web data: {len(only_web)}")
if only_web:
    print(f"  {list(only_web)}")
print()

# Add missing columns to web data with default values
for col in only_training:
    if col not in df_web.columns:
        # Set default values based on column type
        if col in ['keirin_cd', 'race_no_int']:
            df_web[col] = 0
        elif col.endswith('_ratio'):
            df_web[col] = 0.0
        elif col.endswith('_count'):
            df_web[col] = 0
        elif 'payout' in col.lower() and col != 'trifecta_payout_num':
            df_web[col] = 0
        else:
            df_web[col] = 0.0

# Ensure column order matches
df_web = df_web[df_training.columns]

# Combine datasets
print("Combining datasets...")
df_combined = pd.concat([df_training, df_web], ignore_index=True)

# Remove duplicates based on race_date, track, race_no
print("Removing duplicates...")
before = len(df_combined)
df_combined = df_combined.drop_duplicates(subset=['race_date', 'track'], keep='last')
after = len(df_combined)
removed = before - after
print(f"  Removed {removed} duplicates")
print()

# Sort by date
df_combined = df_combined.sort_values('race_date').reset_index(drop=True)

# Save merged dataset
output_file = Path('analysis/model_outputs/enhanced_results_with_web_data.csv')
df_combined.to_csv(output_file, index=False, encoding='utf-8-sig')

print("="*70)
print("MERGE COMPLETE")
print("="*70)
print(f"Total races: {len(df_combined):,}")
print(f"High payout races: {df_combined['target_high_payout'].sum():,} ({df_combined['target_high_payout'].mean():.1%})")
print(f"Saved to: {output_file}")
print()

# Show last few rows (should be web search data)
print("Latest races added:")
print(df_combined.tail(10)[['race_date', 'track', 'grade', 'trifecta_payout_num', 'score_cv']])
