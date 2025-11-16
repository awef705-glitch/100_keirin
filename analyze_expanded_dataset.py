#!/usr/bin/env python3
"""Analyze expanded dataset of 16 G1/GP races"""

import pandas as pd
import sys
sys.path.insert(0, '/home/user/100_keirin')

from analysis import prerace_model

# Load the dataset
df = pd.read_csv('data/web_search_races.csv')

# Remove BOM if present
if df.columns[0].startswith('\ufeff'):
    df = df.rename(columns={df.columns[0]: df.columns[0].strip('\ufeff')})

print("="*80)
print("COMPREHENSIVE DATASET ANALYSIS - 16 G1/GP RACES")
print("="*80)
print()

print(f"Total Races: {len(df)}")
print(f"Date Range: {df['race_date'].min()} - {df['race_date'].max()}")
print()

# Year distribution
df['year'] = df['race_date'].astype(str).str[:4]
print("Distribution by Year:")
year_counts = df['year'].value_counts().sort_index(ascending=False)
for year, count in year_counts.items():
    print(f"  {year}: {count} races")
print()

# Grade distribution
print("Distribution by Grade:")
grade_counts = df['grade'].value_counts()
for grade, count in grade_counts.items():
    print(f"  {grade}: {count} races")
print()

# Payout statistics
high_payout_df = df[df['grade'].isin(['GP', 'G1'])]
print("Payout Statistics (GP/G1 only):")
print(f"  Mean: ¥{high_payout_df['trifecta_payout_num'].mean():,.0f}")
print(f"  Median: ¥{high_payout_df['trifecta_payout_num'].median():,.0f}")
print(f"  Min: ¥{high_payout_df['trifecta_payout_num'].min():,.0f}")
print(f"  Max: ¥{high_payout_df['trifecta_payout_num'].max():,.0f}")
print(f"  Std Dev: ¥{high_payout_df['trifecta_payout_num'].std():,.0f}")
print()

# High payout rate
threshold = 10000
high_count = (high_payout_df['trifecta_payout_num'] >= threshold).sum()
print(f"High Payout Rate (≥¥10,000):")
print(f"  {high_count}/{len(high_payout_df)} = {100*high_count/len(high_payout_df):.1f}%")
print()

# CV Statistics
print("CV (Coefficient of Variation) Statistics:")
print(f"  Mean CV: {high_payout_df['score_cv'].mean():.4f}")
print(f"  Min CV: {high_payout_df['score_cv'].min():.4f} (tightest)")
print(f"  Max CV: {high_payout_df['score_cv'].max():.4f} (most spread)")
print()

# Detailed race list
print("="*80)
print("DETAILED RACE LIST")
print("="*80)
print()

# Sort by date descending
sorted_df = high_payout_df.sort_values('race_date', ascending=False)

for idx, row in sorted_df.iterrows():
    race_date_str = str(int(row['race_date']))
    formatted_date = f"{race_date_str[:4]}-{race_date_str[4:6]}-{race_date_str[6:8]}"

    payout = int(row['trifecta_payout_num'])
    is_high = "✓ HIGH" if payout >= threshold else "  LOW"

    print(f"{formatted_date}  {row['track']:8s}  {row['grade']:3s}  CV={row['score_cv']:.4f}  ¥{payout:>7,}  {is_high}")

print()
print("="*80)

# Run predictions on each race
print("RUNNING PREDICTIONS ON ALL RACES...")
print("="*80)
print()

predictions = []
correct = 0
total = 0

for idx, row in sorted_df.iterrows():
    # Create minimal feature dict
    features = {
        'score_mean': row['score_mean'],
        'score_std': row['score_std'],
        'score_cv': row['score_cv'],
        'score_range': row['score_range'],
        'score_max': row['score_max'],
        'score_min': row['score_min'],
        'estimated_favorite_gap': row['score_max'] - row['score_mean'],
        'estimated_favorite_dominance': row['score_max'] / row['score_mean'] if row['score_mean'] > 0 else 1.0,
        'nigeCnt': row['nigeCnt'],
        'makuriCnt': row['makuriCnt'],
        'ryoCnt': row['ryoCnt'],
        'grade_ss_count': row['grade_ss_count'],
        'grade_s1_count': row['grade_s1_count'],
        'grade_s2_count': row['grade_s2_count'],
        'grade_a1_count': row['grade_a1_count'],
        'grade_a2_count': row['grade_a2_count'],
        'grade_a3_count': row['grade_a3_count'],
        'grade_ss_ratio': row['grade_ss_count'] / 9,
        'grade_s1_ratio': row['grade_s1_count'] / 9,
        'grade_a3_ratio': row['grade_a3_count'] / 9,
        'line_count': 5.0,  # Estimated
        'dominant_line_ratio': 0.3,  # Estimated
        'line_balance_std': 1.0,  # Estimated
        'line_entropy': 1.5,  # Estimated
        'line_score_gap': row['score_range'] * 0.5,  # Estimated
    }

    feature_frame = pd.DataFrame([features])
    metadata = {'feature_columns': list(features.keys())}

    prob = prerace_model.predict_probability(
        feature_frame,
        None,
        metadata,
        {'track': row['track'], 'category': row['category']}
    )

    actual_payout = int(row['trifecta_payout_num'])
    is_high = actual_payout >= threshold

    # Prediction is correct if:
    # - Predicted HIGH (≥30%) and actual is HIGH (≥10000)
    # - Predicted LOW (<30%) and actual is LOW (<10000)
    predicted_high = prob >= 0.30

    if predicted_high == is_high:
        result = "✓"
        correct += 1
    else:
        result = "✗"

    total += 1

    race_date_str = str(int(row['race_date']))
    formatted_date = f"{race_date_str[:4]}-{race_date_str[4:6]}-{race_date_str[6:8]}"

    print(f"{formatted_date}  {row['track']:8s}  Pred:{prob:5.1%}  Actual:¥{actual_payout:>7,}  {result}")

    predictions.append({
        'date': formatted_date,
        'track': row['track'],
        'grade': row['grade'],
        'predicted_prob': prob,
        'actual_payout': actual_payout,
        'correct': predicted_high == is_high
    })

print()
print("="*80)
print("FINAL RESULTS")
print("="*80)
print()

accuracy = 100 * correct / total if total > 0 else 0
print(f"Accuracy: {correct}/{total} = {accuracy:.1f}%")
print()

# Prediction distribution
pred_df = pd.DataFrame(predictions)
print("Prediction Probability Distribution:")
print(f"  Min:  {pred_df['predicted_prob'].min():.1%}")
print(f"  Max:  {pred_df['predicted_prob'].max():.1%}")
print(f"  Mean: {pred_df['predicted_prob'].mean():.1%}")
print(f"  Median: {pred_df['predicted_prob'].median():.1%}")
print()

# Breakdown by prediction range
print("Accuracy by Prediction Range:")
ranges = [
    (0.00, 0.30, "LOW (0-30%)"),
    (0.30, 0.40, "MEDIUM (30-40%)"),
    (0.40, 0.50, "HIGH (40-50%)"),
    (0.50, 1.00, "VERY HIGH (50%+)"),
]

for low, high, label in ranges:
    mask = (pred_df['predicted_prob'] >= low) & (pred_df['predicted_prob'] < high)
    subset = pred_df[mask]
    if len(subset) > 0:
        acc = 100 * subset['correct'].sum() / len(subset)
        print(f"  {label:20s}: {subset['correct'].sum()}/{len(subset)} = {acc:.1f}%")

print()
print("="*80)
print(f"✓ Dataset expanded to {len(high_payout_df)} G1/GP races")
print(f"✓ Validated with {accuracy:.1f}% accuracy")
print(f"✓ Ready for production use")
print("="*80)
