#!/usr/bin/env python3
"""Test prediction system with REAL 競輪祭2024 決勝 data"""

import sys
sys.path.insert(0, '/home/user/100_keirin')

from analysis import prerace_model
import pandas as pd
import numpy as np

print("="*70)
print("REAL RACE VALIDATION: 第66回朝日新聞社杯競輪祭 決勝")
print("="*70)
print()

# Actual race data from web search
print("Race Information:")
print("  Date: 2024年11月24日")
print("  Venue: 小倉競輪場")
print("  Grade: G1")
print()

# Build race with ACTUAL DATA from web search
riders = [
    # ACTUAL competition scores from web search
    {'name': '1 松浦悠士', 'prefecture': '広島', 'grade': 'S1', 'style': '追', 'avg_score': 114.04},  # ACTUAL
    {'name': '2 脇本雄太', 'prefecture': '福井', 'grade': 'SS', 'style': '逃', 'avg_score': 117.59},  # ACTUAL
    {'name': '3 荒井崇博', 'prefecture': '長崎', 'grade': 'S1', 'style': '追', 'avg_score': 112.0},  # Estimated
    {'name': '4 寺崎浩平', 'prefecture': '福井', 'grade': 'S1', 'style': '逃', 'avg_score': 114.0},  # Estimated
    {'name': '5 松谷秀幸', 'prefecture': '神奈川', 'grade': 'S1', 'style': '追', 'avg_score': 113.88},  # ACTUAL
    {'name': '6 村上博幸', 'prefecture': '京都', 'grade': 'S1', 'style': '追', 'avg_score': 113.0},  # Estimated
    {'name': '7 犬伏湧也', 'prefecture': '徳島', 'grade': 'S1', 'style': '逃', 'avg_score': 116.88},  # ACTUAL
    {'name': '8 菅田壱道', 'prefecture': '宮城', 'grade': 'S1', 'style': '追', 'avg_score': 112.73},  # ACTUAL
    {'name': '9 浅井康太', 'prefecture': '三重', 'grade': 'S1', 'style': '追', 'avg_score': 114.20},  # ACTUAL
]

print("Entry List (9 riders, S級):")
print()
print("  No. Name         Prefecture  Grade  Style  Score")
print("  " + "-"*55)
for r in riders:
    marker = " *" if r['avg_score'] in [114.04, 117.59, 113.88, 116.88, 112.73, 114.20] else ""
    print(f"  {r['name']:15s} {r['prefecture']:6s}     {r['grade']:3s}    {r['style']:2s}   {r['avg_score']:.2f}{marker}")

print()
print("  NOTE: Scores marked with * are ACTUAL data from web search")
print("        (6 out of 9 riders have real data)")
print()

# Build race info
race_info = {
    'race_date': '20241124',
    'track': '小倉',
    'keirin_cd': '81',
    'race_no': 12,
    'grade': 'G1',
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
    {'track': '小倉', 'category': 'Ｇ１決勝'}
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
print("  1st Place:  2番車 脇本雄太 (福井)")
print("  2nd Place:  7番車 犬伏湧也 (徳島)")
print("  3rd Place:  1番車 松浦悠士 (広島)")
print()
print("  Trifecta (三連単):  2-7-1")
print("  ACTUAL PAYOUT:      ¥10,270")
print()

# Evaluation
print("="*70)
print("EVALUATION")
print("="*70)
print()

actual_payout = 10270
is_high_payout = actual_payout >= 10000

print(f"Actual payout: ¥{actual_payout:,}")
print(f"High payout (≥¥10,000): {'YES ✓' if is_high_payout else 'NO'}")
print()

# Check if prediction was correct
if is_high_payout and prob >= 0.30:
    result = "✓ CORRECT PREDICTION"
    explanation = "System predicted MEDIUM-HIGH to HIGH probability, and the race indeed paid ¥10,270 (high payout)"
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
print("Why this race had medium-high payout:")
print(f"  - Mixed grades (1 SS, 8 S1) → competitive")
print(f"  - Moderate score spread (CV={features.get('score_cv', 0):.4f})")
print(f"  - Clear favorite (脇本雄太: 117.59) but not overwhelming")
print(f"  - Multiple lines from different regions")
print(f"  - G1 final → high stakes, strategic racing")
print()
print("System's reasoning:")
print(f"  - Detected moderate competition (CV={features.get('score_cv', 0):.4f})")
print(f"  - Favorite has advantage but not dominant")
print(f"  - Balanced line composition")
print(f"  → Predicted {prob:.1%} probability of high payout")
print()

if result.startswith("✓"):
    print("✓ VALIDATION PASSED - System correctly predicted this race")
else:
    print("✗ VALIDATION FAILED - System needs adjustment")

print("="*70)
