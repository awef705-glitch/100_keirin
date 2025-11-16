#!/usr/bin/env python3
"""Test prediction system with ACTUAL past race results"""

import sys
sys.path.insert(0, '/home/user/100_keirin')

import pandas as pd
import numpy as np
from pathlib import Path

# Load REAL results
results_path = Path('analysis/model_outputs/enhanced_results_for_training.csv')
df = pd.read_csv(results_path)

print("="*70)
print("TESTING WITH ACTUAL RACE RESULTS")
print("="*70)
print(f"Total races in dataset: {len(df):,}")
print(f"High payout races (≥¥10,000): {df['target_high_payout'].sum():,} ({df['target_high_payout'].mean():.1%})")
print()

# Sample races from different categories
print("Sampling real races by category...")
print()

# Get some actual race details
sample_races = []

# High payout races from different categories
high_payout = df[df['target_high_payout'] == 1].sample(n=min(5, len(df[df['target_high_payout'] == 1])), random_state=42)
low_payout = df[df['target_high_payout'] == 0].sample(n=min(5, len(df[df['target_high_payout'] == 0])), random_state=42)

print("SAMPLE HIGH PAYOUT RACES (actual ≥¥10,000):")
for idx, row in high_payout.iterrows():
    print(f"  {row['race_date']} {row['track']:8s} {row['race_no']:2s} {row['category']:20s} ¥{row['trifecta_payout_num']:>8,.0f}")

print()
print("SAMPLE LOW PAYOUT RACES (actual <¥10,000):")
for idx, row in low_payout.iterrows():
    print(f"  {row['race_date']} {row['track']:8s} {row['race_no']:2s} {row['category']:20s} ¥{row['trifecta_payout_num']:>8,.0f}")

print()
print("="*70)

# Problem: We don't have rider-level data in this file
# We need to show the user that we can't fully validate without rider data

print("LIMITATION FOUND:")
print("  The results CSV doesn't contain rider-level data (scores, grades, etc.)")
print("  We cannot calculate our features (CV, favorite_gap, etc.) without this.")
print()
print("Available data in results CSV:")
print("  - Race date, track, category, grade")
print("  - Final payout amounts")
print("  - Track/category historical statistics")
print()
print("Missing data needed for prediction:")
print("  - Individual rider competition scores (競走得点)")
print("  - Individual rider grades (SS/S1/A1/etc)")
print("  - Individual rider styles (逃げ/追込/両)")
print("  - Individual rider prefectures (for line analysis)")
print()

print("="*70)
print("HONEST ASSESSMENT:")
print("="*70)
print()
print("✓ The 10 test scenarios I created: 100% pass")
print("✗ Real race validation: CANNOT PERFORM without rider data")
print()
print("What this means:")
print("  - Rules are LOGICALLY CORRECT (based on tests)")
print("  - Rules are CALIBRATED to realistic ranges (10-50%)")
print("  - But I CANNOT PROVE they work on actual races")
print()
print("What I can do:")
print("  1. Show you the test results (already done)")
print("  2. Deploy and let YOU test with real data")
print("  3. Fix immediately if it doesn't work")
print()
print("What I CANNOT do:")
print("  - Guarantee it works perfectly")
print("  - Validate against historical race outcomes (no rider data)")
print()
print("="*70)
print("RECOMMENDATION:")
print("="*70)
print()
print("Deploy to Render and test with 3-5 real races from KEIRIN.JP:")
print("  1. Input actual rider data (scores, grades, styles)")
print("  2. See if prediction makes sense")
print("  3. If wrong, I'll fix it immediately")
print()
print("I will NOT say 'it's perfect' until YOU confirm it works.")
print("="*70)
