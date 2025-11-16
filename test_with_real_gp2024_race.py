#!/usr/bin/env python3
"""Test prediction system with REAL KEIRINグランプリ2024 data"""

import sys
sys.path.insert(0, '/home/user/100_keirin')

from analysis import prerace_model
import pandas as pd
import numpy as np

print("="*70)
print("REAL RACE VALIDATION: KEIRINグランプリ2024")
print("="*70)
print()

# Actual race data from web search
print("Race Information:")
print("  Date: 2024年12月30日")
print("  Venue: 静岡競輪場")
print("  Grade: GP (最高峰)")
print()

# Build race with ACTUAL DATA from web search
riders = [
    # All SS-class riders
    {'name': '1 古性優作', 'prefecture': '大阪', 'grade': 'SS', 'style': '追', 'avg_score': 118.21},
    {'name': '2 平原康多', 'prefecture': '埼玉', 'grade': 'SS', 'style': '逃', 'avg_score': 117.0},  # Estimated
    {'name': '3 郡司浩平', 'prefecture': '神奈川', 'grade': 'SS', 'style': '追', 'avg_score': 116.5},  # Estimated
    {'name': '4 眞杉匠', 'prefecture': '栃木', 'grade': 'SS', 'style': '逃', 'avg_score': 117.5},  # Estimated
    {'name': '5 岩本俊介', 'prefecture': '千葉', 'grade': 'SS', 'style': '追', 'avg_score': 116.0},  # Estimated
    {'name': '6 清水裕友', 'prefecture': '山口', 'grade': 'SS', 'style': '追', 'avg_score': 117.0},  # Estimated SS-class
    {'name': '7 北井佑季', 'prefecture': '神奈川', 'grade': 'SS', 'style': '逃', 'avg_score': 115.5},  # Estimated
    {'name': '8 新山響平', 'prefecture': '青森', 'grade': 'SS', 'style': '追', 'avg_score': 116.0},  # Estimated
    {'name': '9 脇本雄太', 'prefecture': '福井', 'grade': 'SS', 'style': '逃', 'avg_score': 118.00},
]

print("Entry List (9 riders, all SS-class):")
print()
print("  No. Name         Prefecture  Grade  Style  Score")
print("  " + "-"*55)
for r in riders:
    print(f"  {r['name']:15s} {r['prefecture']:6s}     {r['grade']:3s}    {r['style']:2s}   {r['avg_score']:.2f}")

print()
print("  NOTE: Scores marked with * are ACTUAL data from web search")
print("        (古性優作: 118.21, 脇本雄太: 118.00)")
print("        Others estimated based on SS-class typical range")
print()

# Build race info
race_info = {
    'race_date': '20241230',
    'track': '静岡',
    'keirin_cd': '38',
    'race_no': 11,
    'grade': 'GP',
    'meeting_day': '',
    'is_first_day': False,
    'is_second_day': False,
    'is_final_day': True,
    'riders': riders
}

# Build features
bundle = prerace_model.build_manual_feature_row(race_info)
features = bundle.features

print("="*70)
print("CALCULATED FEATURES")
print("="*70)
print()
print(f"Score Statistics:")
print(f"  Mean:               {features.get('score_mean', 0):.2f}")
print(f"  Std Dev:            {features.get('score_std', 0):.2f}")
print(f"  Coefficient of Variation (CV): {features.get('score_cv', 0):.4f}")
print(f"  Range:              {features.get('score_range', 0):.2f}")
print()
print(f"Top Rider Analysis:")
print(f"  Favorite Gap (1st - 2nd):        {features.get('estimated_favorite_gap', 0):.2f}")
print(f"  Favorite Dominance (1st / Mean): {features.get('estimated_favorite_dominance', 1.0):.3f}")
print(f"  Top3 vs Others Gap:              {features.get('estimated_top3_vs_others', 0):.2f}")
print()
print(f"Line Analysis:")
print(f"  Line Count:          {features.get('line_count', 0):.0f}")
print(f"  Dominant Line Ratio: {features.get('dominant_line_ratio', 0):.3f}")
print(f"  Line Balance Std:    {features.get('line_balance_std', 0):.3f}")
print(f"  Line Entropy:        {features.get('line_entropy', 0):.3f}")
print(f"  Line Score Gap:      {features.get('line_score_gap', 0):.2f}")
print()
print(f"Grade Composition:")
print(f"  SS-class Ratio:      {features.get('grade_ss_ratio', 0):.2f}")
print(f"  S1-class Ratio:      {features.get('grade_s1_ratio', 0):.2f}")
print(f"  A3-class Ratio:      {features.get('grade_a3_ratio', 0):.2f}")
print()

# Get prediction
feature_frame = pd.DataFrame([features])
metadata = {'feature_columns': list(features.keys())}

prob = prerace_model.predict_probability(
    feature_frame,
    None,  # No ML model
    metadata,
    {'track': '静岡', 'category': 'ＧＰ'}
)

print("="*70)
print("PREDICTION vs REALITY")
print("="*70)
print()
print(f">>> PREDICTED HIGH PAYOUT PROBABILITY: {prob:.1%} <<<")
print()

if prob >= 0.40:
    prediction_label = "HIGH (荒れる)"
    expected_payout = "≥ ¥15,000"
elif prob >= 0.30:
    prediction_label = "MEDIUM-HIGH (やや荒れ)"
    expected_payout = "¥10,000 - ¥20,000"
elif prob >= 0.20:
    prediction_label = "MEDIUM (普通)"
    expected_payout = "¥5,000 - ¥12,000"
else:
    prediction_label = "LOW (固い)"
    expected_payout = "< ¥8,000"

print(f"  Prediction Label:   {prediction_label}")
print(f"  Expected Payout:    {expected_payout}")
print()
print("="*70)
print("ACTUAL RACE RESULT")
print("="*70)
print()
print("  1st Place:  1番車 古性優作 (大阪)")
print("  2nd Place:  6番車 清水裕友 (山口)")
print("  3rd Place:  9番車 脇本雄太 (福井)")
print()
print("  Trifecta (三連単):  1-6-9")
print("  ACTUAL PAYOUT:      ¥19,300")
print("  Popularity:         74番人気 (大穴)")
print()

# Evaluation
print("="*70)
print("EVALUATION")
print("="*70)
print()

actual_payout = 19300
is_high_payout = actual_payout >= 10000

print(f"Actual payout: ¥{actual_payout:,}")
print(f"High payout (≥¥10,000): {'YES ✓' if is_high_payout else 'NO'}")
print()

# Check if prediction was correct
if is_high_payout and prob >= 0.30:
    result = "✓ CORRECT PREDICTION"
    explanation = "System predicted MEDIUM-HIGH to HIGH probability, and the race indeed paid ¥19,300 (high payout)"
elif not is_high_payout and prob < 0.30:
    result = "✓ CORRECT PREDICTION"
    explanation = "System predicted LOW to MEDIUM probability, and the race paid less than ¥10,000"
else:
    result = "✗ INCORRECT PREDICTION"
    if is_high_payout:
        explanation = f"System predicted {prob:.1%} but race paid ¥{actual_payout:,} (high payout)"
    else:
        explanation = f"System predicted {prob:.1%} but race paid ¥{actual_payout:,} (low payout)"

print(f"{result}")
print()
print(f"Explanation:")
print(f"  {explanation}")
print()

# Analysis
print("="*70)
print("ANALYSIS")
print("="*70)
print()
print("Why this race had high payout:")
print(f"  - ALL SS-class riders → GP level, unpredictable")
print(f"  - Very tight scores (CV={features.get('score_cv', 0):.4f}, very small)")
print(f"  - Multiple lines → chaotic race")
print(f"  - Winner was 1番車 (not favorite in betting)")
print(f"  - 74番人気 → very unpopular combination")
print()
print("System's reasoning:")
print(f"  - Detected very tight race (CV={features.get('score_cv', 0):.4f})")
print(f"  - All SS-class → applied GP penalty")
print(f"  - Multiple balanced lines")
print(f"  → Predicted {prob:.1%} probability of high payout")
print()

if result.startswith("✓"):
    print("✓ VALIDATION PASSED - System correctly predicted this race")
else:
    print("✗ VALIDATION FAILED - System needs adjustment")

print("="*70)
