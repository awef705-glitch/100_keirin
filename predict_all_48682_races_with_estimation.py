#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
全48,682レース予測（選手推定データ使用・キャリブレーション版）

処理手順：
1. 全48,682レースのデータを読み込み
2. グレード・カテゴリから各レースの選手データを推定（9人×48,682 = 437,138人分）
3. 特徴量を計算
4. キャリブレーション済み予測モデルで予測
5. 精度を評価
"""

import pandas as pd
import numpy as np
import json
import random
import sys
from collections import Counter

sys.path.append('analysis')
from prerace_model import predict_probability, load_metadata

# キャリブレーション係数
CALIBRATION_SCALE = 0.7151

print("="*80)
print("【全48,682レース 推定データによる予測検証】")
print("="*80)

# データセット読み込み
df = pd.read_csv('analysis/model_outputs/enhanced_results_for_training.csv')
print(f"\n総レース数: {len(df):,}")
print(f"期間: {df['race_date'].min()} 〜 {df['race_date'].max()}")

# 全レース使用
df_sample = df.copy()
df_sample = df_sample.sort_values('race_date').reset_index(drop=True)

print(f"\nサンプル: {len(df_sample):,}レース")
print(f"高配当レース: {df_sample['target_high_payout'].sum()}レース ({df_sample['target_high_payout'].mean()*100:.1f}%)")

# 選手データ推定ロジック
def estimate_riders_from_category_grade(category, grade, race_no):
    """カテゴリとグレードから選手データを推定"""
    riders = []
    random.seed(hash(f"{category}_{grade}_{race_no}"))

    # グレードレベル別の設定
    if 'GP' in grade:
        base_score = 117.0
        score_range = 3.0
        grades = ['SS'] * 9
        styles = ['追'] * 6 + ['逃'] * 3
    elif 'G1' in grade:
        base_score = 115.0
        score_range = 4.0
        grades = ['SS'] * 3 + ['S1'] * 6
        styles = ['追'] * 6 + ['逃'] * 3
    elif 'G2' in grade:
        base_score = 114.0
        score_range = 4.5
        grades = ['S1'] * 7 + ['S2'] * 2
        styles = ['追'] * 6 + ['逃'] * 3
    elif 'G3' in grade:
        if 'Ｓ級' in category:
            base_score = 112.0
            score_range = 3.2
            grades = ['S1'] * 3 + ['S2'] * 6
            styles = ['追'] * 6 + ['逃'] * 3
        else:
            base_score = 92.0
            score_range = 4.5
            grades = ['A1'] * 4 + ['A2'] * 5
            styles = ['追'] * 6 + ['逃'] * 3
    elif 'F1' in grade:
        if 'Ｓ級' in category:
            base_score = 109.0
            score_range = 4.0
            grades = ['S2'] * 9
            styles = ['追'] * 6 + ['逃'] * 3
        else:
            base_score = 90.0
            score_range = 5.0
            grades = ['A1'] * 5 + ['A2'] * 4
            styles = ['追'] * 6 + ['逃'] * 3
    elif 'F2' in grade:
        if 'Ａ級チャレンジ' in category:
            base_score = 82.0
            score_range = 6.0
            grades = ['A3'] * 9
            styles = ['追'] * 6 + ['逃'] * 3
        else:
            base_score = 88.0
            score_range = 5.5
            grades = ['A2'] * 5 + ['A3'] * 4
            styles = ['追'] * 6 + ['逃'] * 3
    elif 'L' in grade:
        base_score = 50.0
        score_range = 5.0
        grades = ['L1'] * 9
        styles = ['追'] * 6 + ['逃'] * 3
    else:
        base_score = 90.0
        score_range = 8.0
        grades = ['A2'] * 9
        styles = ['追'] * 6 + ['逃'] * 3

    prefectures = ['東京', '大阪', '神奈川', '埼玉', '千葉', '愛知', '福岡', '北海道', '静岡']

    for i in range(9):
        score = base_score + (random.random() - 0.5) * score_range
        riders.append({
            'car_no': i + 1,
            'grade': grades[i],
            'style': styles[i],
            'avg_score': round(score, 1),
            'prefecture': prefectures[i]
        })

    return riders

# 特徴量計算
def calculate_features_from_riders(riders, race_data):
    """選手データから特徴量を計算"""
    scores = [r['avg_score'] for r in riders]
    score_mean = np.mean(scores)
    score_std = np.std(scores, ddof=0)
    score_cv = score_std / (score_mean + 1e-6)

    scores_sorted = sorted(scores, reverse=True)
    estimated_favorite_dominance = scores_sorted[0] / score_mean
    estimated_favorite_gap = scores_sorted[0] - scores_sorted[1]
    score_top3_mean = np.mean(scores_sorted[:3])
    estimated_top3_vs_others = score_top3_mean - np.mean(scores_sorted[3:])

    # 級班分布
    grade_counts = Counter(r['grade'] for r in riders)
    total = len(riders)
    grade_SS_ratio = grade_counts.get('SS', 0) / total
    grade_S1_ratio = grade_counts.get('S1', 0) / total
    grade_S2_ratio = grade_counts.get('S2', 0) / total
    grade_A1_ratio = grade_counts.get('A1', 0) / total
    grade_A2_ratio = grade_counts.get('A2', 0) / total
    grade_A3_ratio = grade_counts.get('A3', 0) / total
    grade_L1_ratio = grade_counts.get('L1', 0) / total

    # 脚質分布
    style_counts = Counter(r['style'] for r in riders)
    style_nige_count = style_counts.get('逃', 0)
    style_tsui_count = style_counts.get('追', 0)

    # グレードフラグ
    grade = race_data['grade']
    grade_flag_GP = 1 if 'GP' in grade else 0
    grade_flag_G1 = 1 if 'G1' in grade else 0

    # 日付情報
    race_date = race_data['race_date']
    date_str = str(race_date).zfill(8)
    dt = pd.to_datetime(date_str, format='%Y%m%d')

    category = race_data['category']
    is_first_day = 1 if '初日' in category else 0
    is_final_day = 1 if ('決勝' in category or '最終' in category) else 0

    return {
        'score_cv': float(score_cv),
        'estimated_favorite_dominance': float(estimated_favorite_dominance),
        'estimated_favorite_gap': float(estimated_favorite_gap),
        'estimated_top3_vs_others': float(estimated_top3_vs_others),
        'grade_SS_ratio': float(grade_SS_ratio),
        'grade_S1_ratio': float(grade_S1_ratio),
        'grade_S2_ratio': float(grade_S2_ratio),
        'grade_A1_ratio': float(grade_A1_ratio),
        'grade_A2_ratio': float(grade_A2_ratio),
        'grade_A3_ratio': float(grade_A3_ratio),
        'grade_L1_ratio': float(grade_L1_ratio),
        'style_nige_count': float(style_nige_count),
        'style_tsui_count': float(style_tsui_count),
        'dominant_line_ratio': 0.4,
        'line_score_gap': float(score_std * 1.5),
        'entry_count': 9.0,
        'is_final_day': float(is_final_day),
        'grade_flag_GP': float(grade_flag_GP),
        'grade_flag_G1': float(grade_flag_G1)
    }

# 処理開始
print("\n選手データ推定と特徴量計算中...")
features_list = []

for i, (idx, row) in enumerate(df_sample.iterrows()):
    if i % 1000 == 0 and i > 0:
        print(f"  {i:,}/{len(df_sample):,} レース処理済み ({i/len(df_sample)*100:.1f}%)")

    # 選手データ推定
    riders = estimate_riders_from_category_grade(
        row['category'],
        row['grade'],
        row['race_no']
    )

    # 特徴量計算
    race_data = {'grade': row['grade'], 'category': row['category'], 'race_date': row['race_date']}
    features = calculate_features_from_riders(riders, race_data)
    features_list.append(features)

print(f"  {len(df_sample):,}/{len(df_sample):,} レース処理済み (100.0%)")

df_features = pd.DataFrame(features_list)

# 予測実行
print("\n予測実行中...")
metadata = {'best_threshold': 0.30}
predictions = []

for i, (idx, row) in enumerate(df_features.iterrows()):
    if i % 1000 == 0 and i > 0:
        print(f"  {i:,}/{len(df_features):,} レース予測済み ({i/len(df_features)*100:.1f}%)")

    feature_frame = pd.DataFrame([row])
    race_context = {
        'track': df_sample.iloc[i]['track'],
        'category': df_sample.iloc[i]['category']
    }

    # 元の予測確率
    raw_prob = predict_probability(feature_frame, None, metadata, race_context)

    # キャリブレーション
    calibrated_prob = raw_prob * CALIBRATION_SCALE
    calibrated_prob = max(0.1, min(0.9, calibrated_prob))

    predictions.append(calibrated_prob)

print(f"  {len(df_features):,}/{len(df_features):,} レース予測済み (100.0%)")

# 結果集計
df_sample['predicted_prob'] = predictions

print("\n精度評価中...")
thresholds = [0.25, 0.30, 0.35, 0.40, 0.45, 0.50]
results_list = []

for threshold in thresholds:
    df_sample['predicted_high'] = (df_sample['predicted_prob'] >= threshold).astype(int)

    correct = (df_sample['predicted_high'] == df_sample['target_high_payout']).sum()
    accuracy = correct / len(df_sample)

    tp = ((df_sample['predicted_high'] == 1) & (df_sample['target_high_payout'] == 1)).sum()
    fp = ((df_sample['predicted_high'] == 1) & (df_sample['target_high_payout'] == 0)).sum()
    tn = ((df_sample['predicted_high'] == 0) & (df_sample['target_high_payout'] == 0)).sum()
    fn = ((df_sample['predicted_high'] == 0) & (df_sample['target_high_payout'] == 1)).sum()

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    results_list.append({
        'threshold': threshold,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'correct': correct
    })

results_df = pd.DataFrame(results_list)
best_idx = results_df['f1'].idxmax()

print("\n" + "="*80)
print(f"【{len(df_sample):,}レース 予測結果】")
print("="*80)
print(f"\nデータ期間: {df_sample['race_date'].min()} 〜 {df_sample['race_date'].max()}")
print(f"高配当レース: {df_sample['target_high_payout'].sum()}レース ({df_sample['target_high_payout'].mean()*100:.1f}%)")

print("\n閾値別性能:")
print(results_df.to_string(index=False))

print(f"\n【最良結果（F1スコア最大）】")
print(f"  閾値: {results_df.loc[best_idx, 'threshold']:.2f}")
print(f"  F1スコア: {results_df.loc[best_idx, 'f1']:.4f}")
print(f"  精度: {results_df.loc[best_idx, 'accuracy']*100:.2f}%")
print(f"  適合率: {results_df.loc[best_idx, 'precision']*100:.2f}%")
print(f"  再現率: {results_df.loc[best_idx, 'recall']*100:.2f}%")

# 保存
df_sample[['race_date', 'track', 'race_no', 'category', 'grade',
           'trifecta_payout_num', 'target_high_payout',
           'predicted_prob']].to_csv(f'analysis/model_outputs/{len(df_sample)}_races_predictions.csv', index=False)

summary = {
    'total_races': len(df_sample),
    'high_payout_races': int(df_sample['target_high_payout'].sum()),
    'best_threshold': float(results_df.loc[best_idx, 'threshold']),
    'best_f1': float(results_df.loc[best_idx, 'f1']),
    'best_accuracy': float(results_df.loc[best_idx, 'accuracy']),
    'method': 'Rider estimation + Rule-based prediction + Calibration (scale=0.7151)',
    'threshold_results': results_df.to_dict('records')
}

with open(f'analysis/model_outputs/{len(df_sample)}_races_summary.json', 'w', encoding='utf-8') as f:
    json.dump(summary, f, ensure_ascii=False, indent=2)

print(f"\n✓ 保存完了")
print("="*80)
