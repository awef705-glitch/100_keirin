#!/usr/bin/env python3
"""Test if rule-based prediction actually differentiates races"""

import sys
sys.path.insert(0, '/home/user/100_keirin')

from analysis import prerace_model
import pandas as pd

def test_race(name, riders_scores):
    """Test a race scenario"""
    print(f"\n{'='*60}")
    print(f"TEST: {name}")
    print(f"{'='*60}")
    print(f"Rider scores: {riders_scores}")

    # Build riders
    riders = []
    prefectures = ['茨城', '埼玉', '東京', '大阪', '京都', '福岡', '熊本', '愛知', '静岡']
    grades = ['S1', 'S1', 'S1', 'A1', 'A1', 'A1', 'A2', 'A2', 'A2']
    styles = ['逃', '追', '両', '逃', '追', '逃', '追', '両', '逃']

    for i, score in enumerate(riders_scores):
        riders.append({
            'name': f'選手{i+1}',
            'prefecture': prefectures[i] if i < len(prefectures) else '東京',
            'grade': grades[i] if i < len(grades) else 'A1',
            'style': styles[i] if i < len(styles) else '逃',
            'avg_score': score
        })

    race_info = {
        'race_date': '20241115',
        'track': '京王閣',
        'keirin_cd': '27',
        'race_no': 7,
        'grade': 'G3',
        'meeting_day': '',
        'is_first_day': False,
        'is_second_day': False,
        'is_final_day': True,
        'riders': riders
    }

    # Build features
    bundle = prerace_model.build_manual_feature_row(race_info)

    # Print key features
    features = bundle.features
    print(f"\nKey features:")
    print(f"  score_cv:                  {features.get('score_cv', 0):.4f}")
    print(f"  score_std:                 {features.get('score_std', 0):.2f}")
    print(f"  score_range:               {features.get('score_range', 0):.2f}")
    print(f"  estimated_favorite_gap:    {features.get('estimated_favorite_gap', 0):.2f}")
    print(f"  estimated_favorite_dominance: {features.get('estimated_favorite_dominance', 1.0):.3f}")
    print(f"  estimated_top3_vs_others:  {features.get('estimated_top3_vs_others', 0):.2f}")
    print(f"  dominant_line_ratio:       {features.get('dominant_line_ratio', 0):.3f}")
    print(f"  line_count:                {features.get('line_count', 0):.0f}")
    print(f"  line_score_gap:            {features.get('line_score_gap', 0):.2f}")

    # Get prediction
    feature_frame = pd.DataFrame([features])
    metadata = {'feature_columns': list(features.keys())}

    prob = prerace_model.predict_probability(
        feature_frame,
        None,  # No ML model
        metadata,
        {'track': '京王閣', 'category': 'S級決勝'}
    )

    print(f"\n>>> HIGH PAYOUT PROBABILITY: {prob:.1%} <<<")

    if prob > 0.6:
        print(">>> PREDICTION: 荒れる（高配当）")
    elif prob > 0.4:
        print(">>> PREDICTION: 中間")
    else:
        print(">>> PREDICTION: 固い（低配当）")

    return prob

# Test cases
if __name__ == "__main__":
    # Case 1: Very tight race (should be HIGH probability)
    prob1 = test_race(
        "接戦レース（全員ほぼ同じ実力）",
        [106, 105, 104, 103, 102, 101, 100, 99, 98]
    )

    # Case 2: Clear favorite (should be LOW probability)
    prob2 = test_race(
        "本命レース（トップが圧倒）",
        [120, 105, 103, 100, 98, 95, 92, 90, 88]
    )

    # Case 3: Moderate
    prob3 = test_race(
        "中間レース",
        [112, 108, 106, 104, 102, 100, 98, 96, 94]
    )

    print(f"\n{'='*60}")
    print(f"SUMMARY")
    print(f"{'='*60}")
    print(f"接戦レース:   {prob1:.1%} {'✓ CORRECT' if prob1 > 0.55 else '✗ WRONG (should be HIGH)'}")
    print(f"本命レース:   {prob2:.1%} {'✓ CORRECT' if prob2 < 0.35 else '✗ WRONG (should be LOW)'}")
    print(f"中間レース:   {prob3:.1%} {'✓ CORRECT' if 0.35 <= prob3 <= 0.55 else '✗ WRONG (should be MEDIUM)'}")

    # Check differentiation
    if abs(prob1 - prob2) < 0.2:
        print(f"\n⚠️  WARNING: 差が小さすぎる（{abs(prob1-prob2):.1%}）- ルールが機能していない！")
    else:
        print(f"\n✓ 差が十分にある（{abs(prob1-prob2):.1%}）")
