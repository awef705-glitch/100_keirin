#!/usr/bin/env python3
"""Analyze real race data to determine optimal thresholds"""

import pandas as pd
import numpy as np
from pathlib import Path

# Load training data
data_path = Path('analysis/model_outputs/enhanced_results_for_training.csv')
df = pd.read_csv(data_path)

print(f"Total races: {len(df):,}")
print(f"High payout races: {df['target_high_payout'].sum():,} ({df['target_high_payout'].mean():.1%})")

# Analyze by category
print("\n" + "="*60)
print("High payout rate by category:")
print("="*60)
category_stats = df.groupby('category').agg({
    'target_high_payout': ['count', 'sum', 'mean']
}).round(3)
category_stats.columns = ['Count', 'High Payout', 'Rate']
category_stats = category_stats.sort_values('Rate', ascending=False)
print(category_stats.head(20))

# Analyze by grade
print("\n" + "="*60)
print("High payout rate by grade:")
print("="*60)
grade_stats = df.groupby('grade').agg({
    'target_high_payout': ['count', 'sum', 'mean']
}).round(3)
grade_stats.columns = ['Count', 'High Payout', 'Rate']
grade_stats = grade_stats.sort_values('Rate', ascending=False)
print(grade_stats)

# Analyze by track
print("\n" + "="*60)
print("Top 15 most chaotic tracks:")
print("="*60)
track_stats = df.groupby('track').agg({
    'target_high_payout': ['count', 'sum', 'mean']
}).round(3)
track_stats.columns = ['Count', 'High Payout', 'Rate']
track_stats = track_stats[track_stats['Count'] >= 100]  # Min 100 races
track_stats = track_stats.sort_values('Rate', ascending=False)
print(track_stats.head(15))

# Key insight: What percentage of payouts fall into each bucket?
print("\n" + "="*60)
print("Payout distribution:")
print("="*60)
bins = [0, 1000, 3000, 5000, 10000, 30000, 50000, 100000, 500000, df['trifecta_payout_num'].max()]
labels = ['<1K', '1-3K', '3-5K', '5-10K', '10-30K', '30-50K', '50-100K', '100-500K', '500K+']
df['payout_bucket'] = pd.cut(df['trifecta_payout_num'], bins=bins, labels=labels)
bucket_dist = df['payout_bucket'].value_counts().sort_index()
print(bucket_dist)
print(f"\nMedian payout: ¥{df['trifecta_payout_num'].median():,.0f}")
print(f"Mean payout: ¥{df['trifecta_payout_num'].mean():,.0f}")
print(f"75th percentile: ¥{df['trifecta_payout_num'].quantile(0.75):,.0f}")
print(f"90th percentile: ¥{df['trifecta_payout_num'].quantile(0.90):,.0f}")
print(f"95th percentile: ¥{df['trifecta_payout_num'].quantile(0.95):,.0f}")

# Save summary
summary = {
    'overall_high_payout_rate': float(df['target_high_payout'].mean()),
    'median_payout': float(df['trifecta_payout_num'].median()),
    'mean_payout': float(df['trifecta_payout_num'].mean()),
    'top_chaotic_categories': category_stats.head(10)['Rate'].to_dict(),
    'top_chaotic_tracks': track_stats.head(10)['Rate'].to_dict(),
}

print("\n" + "="*60)
print("SUMMARY")
print("="*60)
print(f"Overall high payout rate (≥¥10,000): {summary['overall_high_payout_rate']:.1%}")
print(f"Median payout: ¥{summary['median_payout']:,.0f}")
print(f"Mean payout: ¥{summary['mean_payout']:,.0f}")
