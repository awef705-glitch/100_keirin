#!/usr/bin/env python3
"""Test prediction system with REAL 競輪祭2023 決勝 data"""

import sys
sys.path.insert(0, '/home/user/100_keirin')

from analysis import prerace_model
import pandas as pd

print("="*70)
print("REAL RACE VALIDATION: 競輪祭2023 決勝")
print("="*70)
print()

print("Race Information:")
print("  Date: 2023年11月26日 (Sunday)")
print("  Venue: 小倉競輪場")
print("  Grade: G1")
print("  Full Name: 第65回朝日新聞社杯競輪祭")
print()

# Based on web search results - estimated scores for riders
riders = [
    {'name': '1 深谷知広', 'prefecture': '静岡', 'grade': 'S1', 'style': '捲', 'avg_score': 116.0},
    {'name': '2 松浦悠士', 'prefecture': '広島', 'grade': 'SS', 'style': '追', 'avg_score': 117.0},
    {'name': '3 太田海也', 'prefecture': '愛媛', 'grade': 'S1', 'style': '追', 'avg_score': 114.0},
    {'name': '4 松井宏佑', 'prefecture': '神奈川', 'grade': 'S1', 'style': '逃', 'avg_score': 115.0},
    {'name': '5 簗田一輝', 'prefecture': '東京', 'grade': 'S1', 'style': '追', 'avg_score': 113.0},
    {'name': '6 脇本雄太', 'prefecture': '福井', 'grade': 'SS', 'style': '逃', 'avg_score': 118.0},
    {'name': '7 眞杉匠', 'prefecture': '栃木', 'grade': 'SS', 'style': '追', 'avg_score': 117.5},
    {'name': '8 南修二', 'prefecture': '大阪', 'grade': 'S1', 'style': '追', 'avg_score': 112.0},
    {'name': '9 北津留翼', 'prefecture': '福岡', 'grade': 'S1', 'style': '逃', 'avg_score': 110.0},
]

print("Entry List (9 riders):")
print()
print("  No. Name         Prefecture  Grade  Style  Score")
print("  " + "-"*55)
for r in riders:
    marker = " *" if r['grade'] == 'SS' else ""
    print(f"  {r['name']:15s} {r['prefecture']:6s}     {r['grade']:3s}    {r['style']:2s}   {r['avg_score']:.2f}{marker}")

print()
print("  NOTE: Scores marked with * are SS-class riders")
print()

print("  Line Formation: 深谷-松井-簗田 / 太田-松浦 / 眞杉(単騎) / 北津留 / 脇本-南")
print()

race_info = {
    'race_date': '20231126',
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

bundle = prerace_model.build_manual_feature_row(race_info)
features = bundle.features

print("="*70)
print("CALCULATED FEATURES")
print("="*70)
print()
print(f"Score Statistics:")
print(f"  Mean:               {features.get('score_mean', 0):.2f}")
print(f"  Std Dev:            {features.get('score_std', 0):.2f}")
print(f"  CV:                 {features.get('score_cv', 0):.4f}")
print(f"  Range:              {features.get('score_range', 0):.2f}")
print()
print(f"Top Rider Analysis:")
print(f"  Favorite Gap:        {features.get('estimated_favorite_gap', 0):.2f}")
print(f"  Favorite Dominance:  {features.get('estimated_favorite_dominance', 1.0):.3f}")
print()
print(f"Line Analysis:")
print(f"  Line Count:          {features.get('line_count', 0):.0f}")
print(f"  Dominant Line Ratio: {features.get('dominant_line_ratio', 0):.3f}")
print(f"  Line Score Gap:      {features.get('line_score_gap', 0):.2f}")
print()

feature_frame = pd.DataFrame([features])
metadata = {'feature_columns': list(features.keys())}

prob = prerace_model.predict_probability(
    feature_frame,
    None,
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
elif prob >= 0.30:
    prediction_label = "MEDIUM-HIGH (やや荒れ)"
elif prob >= 0.20:
    prediction_label = "MEDIUM (普通)"
else:
    prediction_label = "LOW (固い)"

print(f"  Prediction Label:   {prediction_label}")
print()
print("="*70)
print("ACTUAL RACE RESULT")
print("="*70)
print()
print("  1st Place:  7番車 眞杉匠 (栃木) ★単騎")
print("  2nd Place:  4番車 松井宏佑 (神奈川) ★113期同期")
print("  3rd Place:  2番車 松浦悠士 (広島)")
print()
print("  Trifecta (三連単):  7-4-2")
print("  ACTUAL PAYOUT:      ¥18,750")
print()
print("  ★ 眞杉匠が単騎で戦い、同期の松井宏佑をゴール直前で差す")
print("  ★ 眞杉匠の2023年G1 2勝目 (1勝目はオールスター)")
print()

print("="*70)
print("EVALUATION")
print("="*70)
print()

actual_payout = 18750
is_high_payout = actual_payout >= 10000

print(f"Actual payout: ¥{actual_payout:,}")
print(f"High payout (≥¥10,000): {'YES ✓' if is_high_payout else 'NO'}")
print()

if is_high_payout and prob >= 0.30:
    result = "✓ CORRECT PREDICTION"
    explanation = f"System predicted {prob:.1%}, race paid ¥{actual_payout:,} (high payout)"
elif not is_high_payout and prob < 0.30:
    result = "✓ CORRECT PREDICTION"
    explanation = f"System predicted {prob:.1%}, race paid ¥{actual_payout:,} (low payout)"
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

print("="*70)
print("ANALYSIS")
print("="*70)
print()
print("Why this race had high payout:")
print(f"  - 眞杉匠 raced alone (単騎) against multiple lines")
print(f"  - Passed fellow 113期 classmate just before finish")
print(f"  - 3 SS-class riders in field (眞杉, 脇本, 松浦)")
print(f"  - Score range 8.0 points → competitive field")
print()
print("System's reasoning:")
print(f"  - CV={features.get('score_cv', 0):.4f}")
print(f"  - {features.get('line_count', 0):.0f} regional lines")
print(f"  - Solo rider (単騎) can disrupt predictions")
print(f"  → Predicted {prob:.1%} probability")
print()

if result.startswith("✓"):
    print("✓ VALIDATION PASSED")
else:
    print("✗ VALIDATION FAILED")

print("="*70)
