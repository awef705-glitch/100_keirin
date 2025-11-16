#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
48,683レース全件検証スクリプト

トラック・カテゴリ・グレードの統計情報のみを使用した簡易予測モデル
選手詳細データなしで実行可能
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
from datetime import datetime

# データ読み込み
print("[1/5] 48,683レースデータを読み込み中...")
df = pd.read_csv('analysis/model_outputs/enhanced_results_for_training.csv')
print(f"✓ {len(df):,}レース読み込み完了")

# 予測関数（統計ベース）
def predict_high_payout_simple(row):
    """
    トラック・カテゴリ・グレードの統計のみを使った簡易予測

    戦略：
    1. 各統計の高配当率を重み付け平均
    2. トラック: 40%（会場の特性が最も重要）
    3. カテゴリ: 35%（レースクラスが次に重要）
    4. グレード: 25%（グレードは補助的）
    """
    track_rate = row.get('track_high_payout_rate', 0.266)
    category_rate = row.get('category_high_payout_rate', 0.266)
    grade_rate = row.get('grade_high_payout_rate', 0.266)

    # 重み付け平均
    weighted_prob = (
        track_rate * 0.40 +
        category_rate * 0.35 +
        grade_rate * 0.25
    )

    # レース時間帯による補正
    if row.get('is_main_race', 0) == 1:
        # メインレース（決勝など）は荒れやすい
        weighted_prob *= 1.05
    elif row.get('is_early_race', 0) == 1:
        # 序盤レースは固い
        weighted_prob *= 0.95

    return weighted_prob

print("\n[2/5] 48,683レース全件に対して予測実行中...")
df['predicted_prob'] = df.apply(predict_high_payout_simple, axis=1)
print(f"✓ 予測完了")

# 閾値を複数試す
thresholds = [0.25, 0.266, 0.27, 0.28, 0.29, 0.30]

print("\n[3/5] 複数の閾値で精度を評価中...")
results = []

for threshold in thresholds:
    df['predicted_high'] = (df['predicted_prob'] >= threshold).astype(int)

    # 精度計算
    correct = (df['predicted_high'] == df['target_high_payout']).sum()
    total = len(df)
    accuracy = correct / total

    # 混同行列
    tp = ((df['predicted_high'] == 1) & (df['target_high_payout'] == 1)).sum()
    fp = ((df['predicted_high'] == 1) & (df['target_high_payout'] == 0)).sum()
    tn = ((df['predicted_high'] == 0) & (df['target_high_payout'] == 0)).sum()
    fn = ((df['predicted_high'] == 0) & (df['target_high_payout'] == 1)).sum()

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    results.append({
        'threshold': threshold,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'tp': tp,
        'fp': fp,
        'tn': tn,
        'fn': fn
    })

results_df = pd.DataFrame(results)

print("\n" + "="*80)
print("【48,683レース 全件検証結果】")
print("="*80)
print(f"\nデータ期間: {df['race_date'].min()} 〜 {df['race_date'].max()}")
print(f"総レース数: {len(df):,}レース")
print(f"高配当レース: {df['target_high_payout'].sum():,}レース ({df['target_high_payout'].mean()*100:.1f}%)")
print(f"\n予測手法: トラック・カテゴリ・グレード統計による重み付け平均")
print(f"  - トラック高配当率: 40%")
print(f"  - カテゴリ高配当率: 35%")
print(f"  - グレード高配当率: 25%")

print("\n" + "-"*80)
print("閾値別精度:")
print("-"*80)
print(results_df.to_string(index=False))

# 最良の閾値を選択（F1スコア最大）
best_idx = results_df['f1'].idxmax()
best_threshold = results_df.loc[best_idx, 'threshold']
best_accuracy = results_df.loc[best_idx, 'accuracy']
best_f1 = results_df.loc[best_idx, 'f1']

print("\n" + "="*80)
print(f"【最良閾値】: {best_threshold}")
print(f"  - 精度: {best_accuracy*100:.2f}%")
print(f"  - F1スコア: {best_f1:.4f}")
print(f"  - 適合率: {results_df.loc[best_idx, 'precision']*100:.2f}%")
print(f"  - 再現率: {results_df.loc[best_idx, 'recall']*100:.2f}%")
print("="*80)

# 最良閾値で予測
df['final_predicted'] = (df['predicted_prob'] >= best_threshold).astype(int)

print("\n[4/5] 期間別・カテゴリ別精度を分析中...")

# 年別精度
df['year'] = df['race_date'].astype(str).str[:4].astype(int)
yearly_accuracy = []
for year in sorted(df['year'].unique()):
    year_df = df[df['year'] == year]
    correct = (year_df['final_predicted'] == year_df['target_high_payout']).sum()
    acc = correct / len(year_df)
    yearly_accuracy.append({
        'year': year,
        'races': len(year_df),
        'accuracy': acc
    })

yearly_df = pd.DataFrame(yearly_accuracy)
print("\n年別精度:")
print(yearly_df.to_string(index=False))

# カテゴリ別精度
category_accuracy = []
for category in df['category'].unique()[:10]:  # 上位10カテゴリ
    cat_df = df[df['category'] == category]
    if len(cat_df) >= 100:  # 100レース以上のカテゴリのみ
        correct = (cat_df['final_predicted'] == cat_df['target_high_payout']).sum()
        acc = correct / len(cat_df)
        category_accuracy.append({
            'category': category,
            'races': len(cat_df),
            'accuracy': acc,
            'high_payout_rate': cat_df['target_high_payout'].mean()
        })

if category_accuracy:
    category_df = pd.DataFrame(category_accuracy)
    category_df = category_df.sort_values('races', ascending=False)
    print("\nカテゴリ別精度（レース数上位）:")
    print(category_df.to_string(index=False))

# グレード別精度
grade_accuracy = []
for grade in df['grade'].unique()[:10]:
    grade_df = df[df['grade'] == grade]
    if len(grade_df) >= 100:
        correct = (grade_df['final_predicted'] == grade_df['target_high_payout']).sum()
        acc = correct / len(grade_df)
        grade_accuracy.append({
            'grade': grade,
            'races': len(grade_df),
            'accuracy': acc,
            'high_payout_rate': grade_df['target_high_payout'].mean()
        })

if grade_accuracy:
    grade_df = pd.DataFrame(grade_accuracy)
    grade_df = grade_df.sort_values('races', ascending=False)
    print("\nグレード別精度（レース数上位）:")
    print(grade_df.to_string(index=False))

print("\n[5/5] 検証結果をCSVに保存中...")

# 予測結果を保存
output_df = df[['race_date', 'keirin_cd', 'track', 'race_no', 'category', 'grade',
                'trifecta_payout_num', 'target_high_payout', 'predicted_prob', 'final_predicted']]
output_df.to_csv('analysis/model_outputs/all_races_predictions.csv', index=False)
print(f"✓ 保存完了: analysis/model_outputs/all_races_predictions.csv")

# サマリーレポート作成
summary = {
    'validation_date': datetime.now().isoformat(),
    'total_races': int(len(df)),
    'date_range': {
        'start': int(df['race_date'].min()),
        'end': int(df['race_date'].max())
    },
    'high_payout_races': int(df['target_high_payout'].sum()),
    'high_payout_rate': float(df['target_high_payout'].mean()),
    'best_threshold': float(best_threshold),
    'best_accuracy': float(best_accuracy),
    'best_f1_score': float(best_f1),
    'prediction_method': 'Weighted average of track/category/grade statistics',
    'weights': {
        'track': 0.40,
        'category': 0.35,
        'grade': 0.25
    },
    'threshold_results': results_df.to_dict('records')
}

with open('analysis/model_outputs/all_races_validation_summary.json', 'w', encoding='utf-8') as f:
    json.dump(summary, f, ensure_ascii=False, indent=2)
print(f"✓ 保存完了: analysis/model_outputs/all_races_validation_summary.json")

print("\n" + "="*80)
print("【検証完了】")
print("="*80)
print(f"\n48,683レース全件の検証が完了しました。")
print(f"\n重要な発見:")
print(f"  ✓ 精度: {best_accuracy*100:.2f}% ({int(best_accuracy*len(df)):,}/{len(df):,}レース的中)")
print(f"  ✓ F1スコア: {best_f1:.4f}")
print(f"  ✓ 最良閾値: {best_threshold}")
print(f"\n結果ファイル:")
print(f"  - all_races_predictions.csv (全予測結果)")
print(f"  - all_races_validation_summary.json (サマリー)")
print("="*80)
