#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
12ヶ月データセットにV4.0特徴量を追加

Based on build_clean_dataset.py logic but adapted for estimated rider data
"""

import pandas as pd
import numpy as np
from pathlib import Path

print("="*80)
print("【12ヶ月データセットにV4.0特徴量を追加】")
print("="*80)

# Load 12-month dataset
df = pd.read_csv('data/keirin_training_dataset_12months_estimated.csv')
print(f"\nLoaded: {len(df):,} rows")
print(f"Unique races: {df[['race_date', 'track', 'race_no']].drop_duplicates().shape[0]:,}")
print(f"Current features: {len(df.columns)}")

# Group by race to calculate race-level statistics
print("\nCalculating race-level features...")
race_groups = df.groupby(['race_date', 'track', 'race_no'])

# Score statistics (already have basic ones, add more)
df['score_range'] = race_groups['heikinTokuten'].transform(lambda x: x.max() - x.min())
df['score_median'] = race_groups['heikinTokuten'].transform('median')
df['score_q25'] = race_groups['heikinTokuten'].transform(lambda x: x.quantile(0.25))
df['score_q75'] = race_groups['heikinTokuten'].transform(lambda x: x.quantile(0.75))
df['score_iqr'] = df['score_q75'] - df['score_q25']

# Top/bottom performers
df['score_top3_mean'] = race_groups['heikinTokuten'].transform(lambda x: x.nlargest(3).mean())
df['score_bottom3_mean'] = race_groups['heikinTokuten'].transform(lambda x: x.nsmallest(3).mean())
df['score_top_bottom_gap'] = df['score_top3_mean'] - df['score_bottom3_mean']

# Estimated favorite features (based on score)
df['estimated_favorite_gap'] = race_groups['heikinTokuten'].transform('max') - race_groups['heikinTokuten'].transform('median')
df['estimated_top3_vs_others'] = df['score_top3_mean'] - race_groups['heikinTokuten'].transform(lambda x: x[~x.isin(x.nlargest(3))].mean())

# B-count statistics (riding style)
for b_type in ['nigeCnt', 'makuriCnt', 'sasiCnt', 'markCnt', 'backCnt']:
    df[f'{b_type}_mean'] = race_groups[b_type].transform('mean')
    df[f'{b_type}_std'] = race_groups[b_type].transform('std').fillna(0)
    df[f'{b_type}_max'] = race_groups[b_type].transform('max')
    df[f'{b_type}_sum'] = race_groups[b_type].transform('sum')
    df[f'{b_type}_cv'] = df[f'{b_type}_std'] / (df[f'{b_type}_mean'] + 0.001)  # Add small value to avoid division by zero

# Style diversity (based on kyakusitu)
style_counts = df.groupby(['race_date', 'track', 'race_no', 'kyakusitu']).size().unstack(fill_value=0)
for style in ['逃', '追', '差', '捲', '両']:
    col_name = f'style_{style}_count'
    if style in style_counts.columns:
        df[col_name] = df.set_index(['race_date', 'track', 'race_no']).index.map(style_counts[style])
    else:
        df[col_name] = 0

df['style_diversity'] = race_groups['kyakusitu'].transform('nunique')

# Grade distribution
grade_counts = df.groupby(['race_date', 'track', 'race_no', 'kyuhan']).size().unstack(fill_value=0)
for grade in ['SS', 'S1', 'S2', 'A1', 'A2', 'A3', 'L1']:
    col_name = f'grade_{grade}_count'
    if grade in grade_counts.columns:
        df[col_name] = df.set_index(['race_date', 'track', 'race_no']).index.map(grade_counts[grade])
    else:
        df[col_name] = 0
    df[f'grade_{grade}_ratio'] = df[col_name] / df['entry_count']

df['grade_has_mixed'] = (race_groups['kyuhan'].transform('nunique') > 1).astype(int)

# === V4.0 NEW FEATURES ===
print("\nAdding V4.0 interaction and polynomial features...")

# 1. Feature Interactions (simplified version without recent_finish since we don't have it)
df['score_std_x_score_cv'] = df['score_std'] * df['score_cv']
df['entry_count_x_score_cv'] = df['entry_count'] * df['score_cv']
df['score_range_x_sasiCnt_cv'] = df['score_range'] * df['sasiCnt_cv']
df['nigeCnt_cv_x_makuriCnt_cv'] = df['nigeCnt_cv'] * df['makuriCnt_cv']
df['score_top_bottom_gap_x_score_cv'] = df['score_top_bottom_gap'] * df['score_cv']

# 2. Polynomial features
df['score_cv_squared'] = df['score_cv'] ** 2
df['score_std_squared'] = df['score_std'] ** 2
df['entry_count_log'] = np.log1p(df['entry_count'])
df['score_range_log'] = np.log1p(df['score_range'])
df['sasiCnt_sum_log'] = np.log1p(df['sasiCnt_sum'])
df['backCnt_cv_squared'] = df['backCnt_cv'] ** 2

# 3. Time-based features
df['race_no_int'] = df['race_no'].str.extract('(\d+)', expand=False).astype(float).fillna(df['race_no'])
df['year'] = df['race_date'] // 10000
df['month'] = (df['race_date'] % 10000) // 100
df['day'] = df['race_date'] % 100
df['day_of_week'] = pd.to_datetime(df['race_date'], format='%Y%m%d').dt.dayofweek
df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)

print(f"\nFinal features: {len(df.columns)}")

# Remove any columns with all zeros or all NaN
null_cols = df.columns[df.isnull().all()].tolist()
zero_cols = df.columns[(df == 0).all()].tolist()
drop_cols = list(set(null_cols + zero_cols))

if drop_cols:
    print(f"\nRemoving {len(drop_cols)} all-null/all-zero columns: {drop_cols}")
    df = df.drop(columns=drop_cols)

# Fill remaining NaN
df = df.fillna(0)

print(f"Active features: {len(df.columns)}")

# Save
output_path = Path('data/keirin_training_dataset_12months_v4.csv')
df.to_csv(output_path, index=False)
print(f"\nSaved to: {output_path}")
print(f"File size: {output_path.stat().st_size / 1024 / 1024:.1f} MB")

# Summary
print("\n" + "="*80)
print("【サマリー】")
print("="*80)
print(f"総レース数: {df[['race_date', 'track', 'race_no']].drop_duplicates().shape[0]:,}")
print(f"総エントリ数: {len(df):,}")
print(f"高配当レース: {df['high_payout_flag'].sum():,} ({df['high_payout_flag'].mean()*100:.1f}%)")
print(f"特徴量数: {len(df.columns)}")
print(f"期間: {df['race_date'].min()} - {df['race_date'].max()}")

print("\n【Top 10 feature names (sample)】")
print(list(df.columns[:10]))

print("\nReady for training with train_clean_model.py!")
