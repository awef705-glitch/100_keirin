#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
簡易推定と詳細推定の違いを分析
"""

import pandas as pd
import json
import numpy as np

print("="*80)
print("【簡易推定 vs 詳細推定 比較分析】")
print("="*80)

# 1. 簡易推定の結果を読み込み
print("\n1. 簡易推定の結果を読み込み中...")
df_simple = pd.read_csv('analysis/model_outputs/recent_1000_races_predictions.csv')
print(f"   - 総レース数: {len(df_simple):,}")
print(f"   - 最良精度: 71.90% (閾値=0.25)")

# 2. 詳細推定の結果を読み込み
print("\n2. 詳細推定の結果を読み込み中...")
df_detailed = pd.read_csv('analysis/model_outputs/1000_races_predictions_with_riders.csv')
print(f"   - 総レース数: {len(df_detailed):,}")
print(f"   - 最良精度: 64.10% (閾値=0.50)")

# 3. 両方のデータをマージ
print("\n3. データをマージ中...")
df_simple['simple_prob'] = df_simple['predicted_prob']
df_detailed['detailed_prob'] = df_detailed['predicted_prob']

# race_date, track, race_noでマージ
# カラム名を統一（trifecta_payout_num → trifecta_payout）
if 'trifecta_payout_num' in df_simple.columns:
    df_simple['trifecta_payout'] = df_simple['trifecta_payout_num']
elif 'trifecta_payout' not in df_simple.columns:
    df_simple['trifecta_payout'] = 0

merged = pd.merge(
    df_simple[['race_date', 'track', 'race_no', 'trifecta_payout', 'target_high_payout', 'simple_prob']],
    df_detailed[['race_date', 'track', 'race_no', 'detailed_prob']],
    on=['race_date', 'track', 'race_no'],
    how='inner'
)

print(f"   - マージ後: {len(merged):,}レース")

# 4. 予測確率の差を計算
merged['prob_diff'] = merged['simple_prob'] - merged['detailed_prob']

# 5. 統計分析
print("\n" + "="*80)
print("【予測確率の比較】")
print("="*80)

print(f"\n平均予測確率:")
print(f"  - 簡易推定: {merged['simple_prob'].mean():.4f}")
print(f"  - 詳細推定: {merged['detailed_prob'].mean():.4f}")

print(f"\n予測確率の標準偏差:")
print(f"  - 簡易推定: {merged['simple_prob'].std():.4f}")
print(f"  - 詳細推定: {merged['detailed_prob'].std():.4f}")

print(f"\n予測確率の範囲:")
print(f"  - 簡易推定: {merged['simple_prob'].min():.4f} 〜 {merged['simple_prob'].max():.4f}")
print(f"  - 詳細推定: {merged['detailed_prob'].min():.4f} 〜 {merged['detailed_prob'].max():.4f}")

print(f"\n予測確率の差（簡易 - 詳細）:")
print(f"  - 平均: {merged['prob_diff'].mean():.4f}")
print(f"  - 標準偏差: {merged['prob_diff'].std():.4f}")
print(f"  - 最小: {merged['prob_diff'].min():.4f}")
print(f"  - 最大: {merged['prob_diff'].max():.4f}")

# 6. 高配当レースでの比較
print("\n" + "="*80)
print("【高配当レースでの比較】")
print("="*80)

high_payout = merged[merged['target_high_payout'] == 1]
low_payout = merged[merged['target_high_payout'] == 0]

print(f"\n高配当レース (n={len(high_payout)}):")
print(f"  - 簡易推定の平均確率: {high_payout['simple_prob'].mean():.4f}")
print(f"  - 詳細推定の平均確率: {high_payout['detailed_prob'].mean():.4f}")

print(f"\n低配当レース (n={len(low_payout)}):")
print(f"  - 簡易推定の平均確率: {low_payout['simple_prob'].mean():.4f}")
print(f"  - 詳細推定の平均確率: {low_payout['detailed_prob'].mean():.4f}")

# 7. 閾値0.25での分類性能比較
print("\n" + "="*80)
print("【閾値0.25での分類性能】")
print("="*80)

threshold = 0.25

# 簡易推定
merged['simple_pred'] = (merged['simple_prob'] >= threshold).astype(int)
simple_correct = (merged['simple_pred'] == merged['target_high_payout']).sum()
simple_accuracy = simple_correct / len(merged)

simple_tp = ((merged['simple_pred'] == 1) & (merged['target_high_payout'] == 1)).sum()
simple_fp = ((merged['simple_pred'] == 1) & (merged['target_high_payout'] == 0)).sum()
simple_tn = ((merged['simple_pred'] == 0) & (merged['target_high_payout'] == 0)).sum()
simple_fn = ((merged['simple_pred'] == 0) & (merged['target_high_payout'] == 1)).sum()

simple_precision = simple_tp / (simple_tp + simple_fp) if (simple_tp + simple_fp) > 0 else 0
simple_recall = simple_tp / (simple_tp + simple_fn) if (simple_tp + simple_fn) > 0 else 0

print(f"\n簡易推定:")
print(f"  - 精度: {simple_accuracy*100:.2f}% ({simple_correct}/{len(merged)})")
print(f"  - 適合率: {simple_precision*100:.2f}%")
print(f"  - 再現率: {simple_recall*100:.2f}%")
print(f"  - TP: {simple_tp}, FP: {simple_fp}, TN: {simple_tn}, FN: {simple_fn}")

# 詳細推定
merged['detailed_pred'] = (merged['detailed_prob'] >= threshold).astype(int)
detailed_correct = (merged['detailed_pred'] == merged['target_high_payout']).sum()
detailed_accuracy = detailed_correct / len(merged)

detailed_tp = ((merged['detailed_pred'] == 1) & (merged['target_high_payout'] == 1)).sum()
detailed_fp = ((merged['detailed_pred'] == 1) & (merged['target_high_payout'] == 0)).sum()
detailed_tn = ((merged['detailed_pred'] == 0) & (merged['target_high_payout'] == 0)).sum()
detailed_fn = ((merged['detailed_pred'] == 0) & (merged['target_high_payout'] == 1)).sum()

detailed_precision = detailed_tp / (detailed_tp + detailed_fp) if (detailed_tp + detailed_fp) > 0 else 0
detailed_recall = detailed_tp / (detailed_tp + detailed_fn) if (detailed_tp + detailed_fn) > 0 else 0

print(f"\n詳細推定:")
print(f"  - 精度: {detailed_accuracy*100:.2f}% ({detailed_correct}/{len(merged)})")
print(f"  - 適合率: {detailed_precision*100:.2f}%")
print(f"  - 再現率: {detailed_recall*100:.2f}%")
print(f"  - TP: {detailed_tp}, FP: {detailed_fp}, TN: {detailed_tn}, FN: {detailed_fn}")

# 8. 予測が異なるレースを分析
print("\n" + "="*80)
print("【予測が異なるレースの分析】")
print("="*80)

# 簡易推定は正解、詳細推定は不正解
simple_correct_detailed_wrong = merged[
    (merged['simple_pred'] == merged['target_high_payout']) &
    (merged['detailed_pred'] != merged['target_high_payout'])
]

# 詳細推定は正解、簡易推定は不正解
detailed_correct_simple_wrong = merged[
    (merged['detailed_pred'] == merged['target_high_payout']) &
    (merged['simple_pred'] != merged['target_high_payout'])
]

print(f"\n簡易推定のみ正解: {len(simple_correct_detailed_wrong)}レース")
print(f"詳細推定のみ正解: {len(detailed_correct_simple_wrong)}レース")
print(f"両方正解: {((merged['simple_pred'] == merged['target_high_payout']) & (merged['detailed_pred'] == merged['target_high_payout'])).sum()}レース")
print(f"両方不正解: {((merged['simple_pred'] != merged['target_high_payout']) & (merged['detailed_pred'] != merged['target_high_payout'])).sum()}レース")

# 9. 簡易推定のみ正解のレースの特徴
if len(simple_correct_detailed_wrong) > 0:
    print(f"\n【簡易推定のみ正解のレース（上位10件）】")
    sample = simple_correct_detailed_wrong.head(10)
    for idx, row in sample.iterrows():
        print(f"{row['race_date']} {row['track']} {row['race_no']} - "
              f"配当:¥{row['trifecta_payout']:,.0f}, "
              f"簡易:{row['simple_prob']:.3f}, 詳細:{row['detailed_prob']:.3f}, "
              f"差:{row['prob_diff']:.3f}")

# 10. 結果を保存
print("\n" + "="*80)
print("【保存中】")
print("="*80)

comparison_result = {
    'total_races': len(merged),
    'simple_estimation': {
        'accuracy': float(simple_accuracy),
        'precision': float(simple_precision),
        'recall': float(simple_recall),
        'avg_prob': float(merged['simple_prob'].mean()),
        'std_prob': float(merged['simple_prob'].std())
    },
    'detailed_estimation': {
        'accuracy': float(detailed_accuracy),
        'precision': float(detailed_precision),
        'recall': float(detailed_recall),
        'avg_prob': float(merged['detailed_prob'].mean()),
        'std_prob': float(merged['detailed_prob'].std())
    },
    'agreement': {
        'both_correct': int(((merged['simple_pred'] == merged['target_high_payout']) &
                             (merged['detailed_pred'] == merged['target_high_payout'])).sum()),
        'simple_only_correct': int(len(simple_correct_detailed_wrong)),
        'detailed_only_correct': int(len(detailed_correct_simple_wrong)),
        'both_wrong': int(((merged['simple_pred'] != merged['target_high_payout']) &
                          (merged['detailed_pred'] != merged['target_high_payout'])).sum())
    }
}

with open('analysis/model_outputs/estimation_comparison.json', 'w', encoding='utf-8') as f:
    json.dump(comparison_result, f, ensure_ascii=False, indent=2)

merged.to_csv('analysis/model_outputs/estimation_comparison_detail.csv', index=False)

print("✓ 保存完了:")
print("  - analysis/model_outputs/estimation_comparison.json")
print("  - analysis/model_outputs/estimation_comparison_detail.csv")

print("\n" + "="*80)
print("【結論】")
print("="*80)
print(f"\n簡易推定が {simple_accuracy*100:.2f}%、詳細推定が {detailed_accuracy*100:.2f}% の精度")
print(f"簡易推定の方が {(simple_accuracy - detailed_accuracy)*100:.2f}ポイント高い")
print(f"\n簡易推定のみ正解: {len(simple_correct_detailed_wrong)}レース")
print(f"→ これらのレースで詳細推定が失敗している原因を特定する必要がある")
print("="*80)
