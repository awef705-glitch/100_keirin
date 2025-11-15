#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
1,000レース予測 - キャリブレーション版
詳細推定の確率を校正して簡易推定と同じ分布にする
"""

import pandas as pd
import json

print("="*80)
print("【1,000レース予測 - キャリブレーション版】")
print("="*80)

# 詳細推定の結果を読み込み
df = pd.read_csv('analysis/model_outputs/1000_races_predictions_with_riders.csv')

print(f"\n元の詳細推定:")
print(f"  平均確率: {df['predicted_prob'].mean():.4f}")
print(f"  標準偏差: {df['predicted_prob'].std():.4f}")
print(f"  範囲: {df['predicted_prob'].min():.4f} 〜 {df['predicted_prob'].max():.4f}")

# 目標値（簡易推定の分布）
target_mean = 0.3413
target_std = 0.0769

# 現在の値
current_mean = df['predicted_prob'].mean()
current_std = df['predicted_prob'].std()

print(f"\n目標（簡易推定の分布）:")
print(f"  平均確率: {target_mean:.4f}")
print(f"  標準偏差: {target_std:.4f}")

# キャリブレーション方法を複数試す
print("\n" + "="*80)
print("【キャリブレーション方法の比較】")
print("="*80)

results_list = []

# 方法1: 線形シフト
shift = target_mean - current_mean
df['calib_shift'] = df['predicted_prob'] + shift
df['calib_shift'] = df['calib_shift'].clip(0.1, 0.9)

print(f"\n【方法1: 線形シフト】")
print(f"  シフト量: {shift:.4f}")
print(f"  校正後平均: {df['calib_shift'].mean():.4f}")
print(f"  校正後標準偏差: {df['calib_shift'].std():.4f}")

# 精度評価
for threshold in [0.25, 0.30, 0.35, 0.40, 0.45, 0.50]:
    pred = (df['calib_shift'] >= threshold).astype(int)
    correct = (pred == df['target_high_payout']).sum()
    accuracy = correct / len(df)
    results_list.append({
        'method': 'shift',
        'threshold': threshold,
        'accuracy': accuracy,
        'correct': correct
    })

# 方法2: 線形スケーリング + シフト（Z-score正規化）
df['calib_zscore'] = (df['predicted_prob'] - current_mean) / current_std * target_std + target_mean
df['calib_zscore'] = df['calib_zscore'].clip(0.1, 0.9)

print(f"\n【方法2: Z-score正規化】")
print(f"  校正後平均: {df['calib_zscore'].mean():.4f}")
print(f"  校正後標準偏差: {df['calib_zscore'].std():.4f}")

for threshold in [0.25, 0.30, 0.35, 0.40, 0.45, 0.50]:
    pred = (df['calib_zscore'] >= threshold).astype(int)
    correct = (pred == df['target_high_payout']).sum()
    accuracy = correct / len(df)
    results_list.append({
        'method': 'zscore',
        'threshold': threshold,
        'accuracy': accuracy,
        'correct': correct
    })

# 方法3: 単純な比率スケーリング
scale = target_mean / current_mean
df['calib_scale'] = df['predicted_prob'] * scale
df['calib_scale'] = df['calib_scale'].clip(0.1, 0.9)

print(f"\n【方法3: 比率スケーリング】")
print(f"  スケール: {scale:.4f}")
print(f"  校正後平均: {df['calib_scale'].mean():.4f}")
print(f"  校正後標準偏差: {df['calib_scale'].std():.4f}")

for threshold in [0.25, 0.30, 0.35, 0.40, 0.45, 0.50]:
    pred = (df['calib_scale'] >= threshold).astype(int)
    correct = (pred == df['target_high_payout']).sum()
    accuracy = correct / len(df)
    results_list.append({
        'method': 'scale',
        'threshold': threshold,
        'accuracy': accuracy,
        'correct': correct
    })

# 結果をDataFrameにまとめる
results_df = pd.DataFrame(results_list)

# 各方法の最良精度を見つける
print("\n" + "="*80)
print("【各方法の最良精度】")
print("="*80)

for method in ['shift', 'zscore', 'scale']:
    method_results = results_df[results_df['method'] == method]
    best_idx = method_results['accuracy'].idxmax()
    best = method_results.loc[best_idx]

    method_name = {
        'shift': '線形シフト',
        'zscore': 'Z-score正規化',
        'scale': '比率スケーリング'
    }[method]

    print(f"\n{method_name}:")
    print(f"  最良閾値: {best['threshold']:.2f}")
    print(f"  精度: {best['accuracy']*100:.2f}% ({best['correct']:.0f}/1000)")

# 詳細な閾値別結果
print("\n" + "="*80)
print("【閾値別精度比較】")
print("="*80)

for threshold in [0.25, 0.30, 0.35, 0.40, 0.45, 0.50]:
    print(f"\n閾値 {threshold:.2f}:")
    for method in ['shift', 'zscore', 'scale']:
        result = results_df[(results_df['method'] == method) &
                           (results_df['threshold'] == threshold)].iloc[0]
        method_name = {
            'shift': '線形シフト',
            'zscore': 'Z-score  ',
            'scale': '比率スケール'
        }[method]
        print(f"  {method_name}: {result['accuracy']*100:6.2f}% ({result['correct']:.0f}/1000)")

# 最良の方法を選択
best_overall = results_df.loc[results_df['accuracy'].idxmax()]
best_method = best_overall['method']
best_threshold = best_overall['threshold']
best_accuracy = best_overall['accuracy']

print("\n" + "="*80)
print("【最良結果】")
print("="*80)

method_name_map = {
    'shift': '線形シフト',
    'zscore': 'Z-score正規化',
    'scale': '比率スケーリング'
}

print(f"\n最良方法: {method_name_map[best_method]}")
print(f"最良閾値: {best_threshold:.2f}")
print(f"精度: {best_accuracy*100:.2f}% ({best_overall['correct']:.0f}/1000)")

# 最良の方法で予測結果を保存
df['calibrated_prob'] = df[f'calib_{best_method}']
df['calibrated_predicted_high'] = (df['calibrated_prob'] >= best_threshold).astype(int)

# 元の詳細推定との比較
original_best = 641  # 元の詳細推定の最良精度（閾値0.50で64.10%）
improvement = best_overall['correct'] - original_best

print(f"\n改善:")
print(f"  元の詳細推定: 64.10% (641/1000)")
print(f"  キャリブレーション版: {best_accuracy*100:.2f}% ({best_overall['correct']:.0f}/1000)")
print(f"  改善: {improvement:+.0f}レース ({(best_accuracy - 0.641)*100:+.2f}ポイント)")

# 簡易推定との比較
simple_best = 0.719  # 簡易推定の最良精度
gap = best_accuracy - simple_best

print(f"\n簡易推定との比較:")
print(f"  簡易推定: 71.90% (719/1000)")
print(f"  キャリブレーション版: {best_accuracy*100:.2f}% ({best_overall['correct']:.0f}/1000)")
print(f"  差: {gap*100:+.2f}ポイント")

# 保存
output_df = df[['race_date', 'track', 'race_no', 'category', 'grade',
                'trifecta_payout', 'target_high_payout',
                'predicted_prob', 'calibrated_prob', 'calibrated_predicted_high']]
output_df.to_csv('analysis/model_outputs/1000_races_predictions_calibrated.csv', index=False)

summary = {
    'total_races': len(df),
    'calibration_method': best_method,
    'best_threshold': float(best_threshold),
    'best_accuracy': float(best_accuracy),
    'improvement_over_original': float((best_accuracy - 0.641)*100),
    'gap_from_simple': float(gap*100),
    'original_mean_prob': float(current_mean),
    'calibrated_mean_prob': float(df['calibrated_prob'].mean()),
    'target_mean_prob': float(target_mean),
    'threshold_results': results_df.to_dict('records')
}

with open('analysis/model_outputs/1000_races_calibrated_summary.json', 'w', encoding='utf-8') as f:
    json.dump(summary, f, ensure_ascii=False, indent=2)

print("\n✓ 保存完了:")
print("  - analysis/model_outputs/1000_races_predictions_calibrated.csv")
print("  - analysis/model_outputs/1000_races_calibrated_summary.json")

print("\n" + "="*80)
print("【完了】")
print("="*80)
