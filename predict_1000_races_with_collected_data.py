#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
収集した1,000レースの選手詳細データで予測実行
"""

import pandas as pd
import numpy as np
import json
import sys
sys.path.append('analysis')
from prerace_model import predict_probability, load_metadata, _default_feature_columns

print("="*80)
print("【1,000レース 選手詳細データで予測実行】")
print("="*80)

# 収集したデータを読み込み
with open('analysis/model_outputs/collected_1000_races_with_riders.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

races = data['collected_races']
print(f"\n総レース数: {len(races):,}レース")
print(f"総選手データ: {len(races) * 9:,}人分")

# 各レースの特徴量を計算
def calculate_features_from_riders(race_data):
    """選手データから特徴量を計算"""
    riders = race_data['riders']

    # スコア統計
    scores = [r['avg_score'] for r in riders]
    score_mean = np.mean(scores)
    score_std = np.std(scores, ddof=0)
    score_min = np.min(scores)
    score_max = np.max(scores)
    score_range = score_max - score_min
    score_median = np.median(scores)
    score_q25 = np.percentile(scores, 25)
    score_q75 = np.percentile(scores, 75)
    score_iqr = score_q75 - score_q25
    score_cv = score_std / (score_mean + 1e-6)

    scores_sorted = sorted(scores, reverse=True)
    score_top3_mean = np.mean(scores_sorted[:3])
    score_bottom3_mean = np.mean(scores_sorted[-3:])
    score_top_bottom_gap = score_top3_mean - score_bottom3_mean

    # 人気関連（スコアベース推定）
    estimated_top3_score_sum = sum(scores_sorted[:3])
    estimated_favorite_dominance = scores_sorted[0] / score_mean
    estimated_favorite_gap = scores_sorted[0] - scores_sorted[1]
    estimated_top3_vs_others = score_top3_mean - np.mean(scores_sorted[3:]) if len(scores_sorted) > 3 else 0.0

    # 脚質分布
    from collections import Counter
    style_counts = Counter(r['style'] for r in riders)
    style_nige_count = style_counts.get('逃', 0)
    style_tsui_count = style_counts.get('追', 0)
    style_ryo_count = style_counts.get('差', 0)  # 差 = ryo

    total = len(riders)
    style_nige_ratio = style_nige_count / total
    style_tsui_ratio = style_tsui_count / total
    style_ryo_ratio = style_ryo_count / total
    style_unknown_ratio = 0.0

    style_ratios = np.array([style_nige_ratio, style_tsui_ratio, style_ryo_ratio])
    style_diversity = 1.0 - np.sum(style_ratios ** 2)
    style_max_ratio = style_ratios.max()
    style_min_ratio = style_ratios.min()
    style_entropy = -np.sum(style_ratios * np.log(style_ratios + 1e-10))

    # 級班分布
    grade_counts = Counter(r['grade'] for r in riders)
    grade_SS_count = grade_counts.get('SS', 0)
    grade_S1_count = grade_counts.get('S1', 0)
    grade_S2_count = grade_counts.get('S2', 0)
    grade_A1_count = grade_counts.get('A1', 0)
    grade_A2_count = grade_counts.get('A2', 0)
    grade_A3_count = grade_counts.get('A3', 0)
    grade_L1_count = grade_counts.get('L1', 0)

    grade_SS_ratio = grade_SS_count / total
    grade_S1_ratio = grade_S1_count / total
    grade_S2_ratio = grade_S2_count / total
    grade_A1_ratio = grade_A1_count / total
    grade_A2_ratio = grade_A2_count / total
    grade_A3_ratio = grade_A3_count / total
    grade_L1_ratio = grade_L1_count / total

    grade_ratios = np.array([grade_SS_ratio, grade_S1_ratio, grade_S2_ratio,
                             grade_A1_ratio, grade_A2_ratio, grade_A3_ratio, grade_L1_ratio])
    grade_entropy = -np.sum(grade_ratios * np.log(grade_ratios + 1e-10))

    has_s_class = any(grade_counts.get(g, 0) > 0 for g in ['SS', 'S1', 'S2'])
    has_a_class = any(grade_counts.get(g, 0) > 0 for g in ['A1', 'A2', 'A3'])
    grade_has_mixed = float(has_s_class and has_a_class)

    # 府県数
    prefecture_unique_count = len(set(r['prefecture'] for r in riders))

    # ライン（簡易推定）
    line_count = 3.0  # 標準的な値
    dominant_line_ratio = 0.4
    line_balance_std = 1.2
    line_entropy = 1.0
    line_score_gap = score_std * 1.5

    # カレンダー特徴量
    race_date = race_data['race_date']
    date_str = str(race_date).zfill(8)
    dt = pd.to_datetime(date_str, format='%Y%m%d')
    year = dt.year
    month = dt.month
    day = dt.day
    day_of_week = dt.dayofweek
    is_weekend = 1 if day_of_week >= 5 else 0

    # レース情報
    race_no = race_data['race_no_int']
    keirin_cd = race_data.get('keirin_cd', '00')
    keirin_cd_num = int(keirin_cd) if keirin_cd.isdigit() else 0

    # グレードフラグ
    grade = race_data['grade']
    grade_flag_GP = 1 if 'GP' in grade else 0
    grade_flag_G1 = 1 if 'G1' in grade else 0
    grade_flag_G2 = 1 if 'G2' in grade else 0
    grade_flag_G3 = 1 if 'G3' in grade else 0
    grade_flag_F1 = 1 if 'F1' in grade else 0
    grade_flag_F2 = 1 if 'F2' in grade else 0
    grade_flag_F3 = 1 if 'F3' in grade else 0
    grade_flag_L = 1 if 'L' in grade else 0

    # 日程情報（簡易推定）
    category = race_data['category']
    is_first_day = 1 if '初日' in category else 0
    is_second_day = 0
    is_final_day = 1 if ('決勝' in category or '最終' in category) else 0

    return {
        'keirin_cd_num': float(keirin_cd_num),
        'race_no': float(race_no),
        'year': float(year),
        'month': float(month),
        'day': float(day),
        'day_of_week': float(day_of_week),
        'is_weekend': float(is_weekend),
        'is_first_day': float(is_first_day),
        'is_second_day': float(is_second_day),
        'is_final_day': float(is_final_day),
        'grade_flag_GP': float(grade_flag_GP),
        'grade_flag_G1': float(grade_flag_G1),
        'grade_flag_G2': float(grade_flag_G2),
        'grade_flag_G3': float(grade_flag_G3),
        'grade_flag_F1': float(grade_flag_F1),
        'grade_flag_F2': float(grade_flag_F2),
        'grade_flag_F3': float(grade_flag_F3),
        'grade_flag_L': float(grade_flag_L),
        'entry_count': float(total),
        'score_mean': float(score_mean),
        'score_std': float(score_std),
        'score_min': float(score_min),
        'score_max': float(score_max),
        'score_range': float(score_range),
        'score_median': float(score_median),
        'score_q25': float(score_q25),
        'score_q75': float(score_q75),
        'score_iqr': float(score_iqr),
        'score_cv': float(score_cv),
        'score_top3_mean': float(score_top3_mean),
        'score_bottom3_mean': float(score_bottom3_mean),
        'score_top_bottom_gap': float(score_top_bottom_gap),
        'estimated_top3_score_sum': float(estimated_top3_score_sum),
        'estimated_favorite_dominance': float(estimated_favorite_dominance),
        'estimated_favorite_gap': float(estimated_favorite_gap),
        'estimated_top3_vs_others': float(estimated_top3_vs_others),
        'style_nige_ratio': float(style_nige_ratio),
        'style_tsui_ratio': float(style_tsui_ratio),
        'style_ryo_ratio': float(style_ryo_ratio),
        'style_unknown_ratio': float(style_unknown_ratio),
        'style_diversity': float(style_diversity),
        'style_max_ratio': float(style_max_ratio),
        'style_min_ratio': float(style_min_ratio),
        'style_nige_count': float(style_nige_count),
        'style_tsui_count': float(style_tsui_count),
        'style_ryo_count': float(style_ryo_count),
        'style_entropy': float(style_entropy),
        'grade_SS_ratio': float(grade_SS_ratio),
        'grade_S1_ratio': float(grade_S1_ratio),
        'grade_S2_ratio': float(grade_S2_ratio),
        'grade_A1_ratio': float(grade_A1_ratio),
        'grade_A2_ratio': float(grade_A2_ratio),
        'grade_A3_ratio': float(grade_A3_ratio),
        'grade_L1_ratio': float(grade_L1_ratio),
        'grade_SS_count': float(grade_SS_count),
        'grade_S1_count': float(grade_S1_count),
        'grade_S2_count': float(grade_S2_count),
        'grade_A1_count': float(grade_A1_count),
        'grade_A2_count': float(grade_A2_count),
        'grade_A3_count': float(grade_A3_count),
        'grade_L1_count': float(grade_L1_count),
        'grade_entropy': float(grade_entropy),
        'grade_has_mixed': float(grade_has_mixed),
        'prefecture_unique_count': float(prefecture_unique_count),
        'line_count': float(line_count),
        'dominant_line_ratio': float(dominant_line_ratio),
        'line_balance_std': float(line_balance_std),
        'line_entropy': float(line_entropy),
        'line_score_gap': float(line_score_gap),
    }

print("\n特徴量計算中...")
features_list = []
for i, race in enumerate(races):
    if i % 100 == 0:
        print(f"  {i}/{len(races)} レース処理済み ({i/len(races)*100:.1f}%)")
    features = calculate_features_from_riders(race)
    features_list.append(features)

print(f"  {len(races)}/{len(races)} レース処理済み (100.0%)")

df_features = pd.DataFrame(features_list)

# 予測実行
print("\n予測実行中...")
metadata = {'best_threshold': 0.30, 'high_confidence_threshold': 0.40}

predictions = []
for i, row in df_features.iterrows():
    feature_frame = pd.DataFrame([row])
    race_context = {
        'track': races[i]['track'],
        'category': races[i]['category']
    }
    prob = predict_probability(feature_frame, None, metadata, race_context)
    predictions.append(prob)

print("✓ 予測完了")

# 結果をDataFrameにまとめる
results_df = pd.DataFrame([
    {
        'race_date': r['race_date'],
        'track': r['track'],
        'race_no': r['race_no'],
        'category': r['category'],
        'grade': r['grade'],
        'trifecta_payout': r['trifecta_payout'],
        'target_high_payout': r['target_high_payout'],
        'predicted_prob': predictions[i]
    }
    for i, r in enumerate(races)
])

# 精度計算
print("\n精度評価中...")
thresholds = [0.25, 0.30, 0.35, 0.40, 0.45, 0.50]
results_list = []

for threshold in thresholds:
    results_df['predicted_high'] = (results_df['predicted_prob'] >= threshold).astype(int)

    correct = (results_df['predicted_high'] == results_df['target_high_payout']).sum()
    accuracy = correct / len(results_df)

    tp = ((results_df['predicted_high'] == 1) & (results_df['target_high_payout'] == 1)).sum()
    fp = ((results_df['predicted_high'] == 1) & (results_df['target_high_payout'] == 0)).sum()
    tn = ((results_df['predicted_high'] == 0) & (results_df['target_high_payout'] == 0)).sum()
    fn = ((results_df['predicted_high'] == 0) & (results_df['target_high_payout'] == 1)).sum()

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

results_table = pd.DataFrame(results_list)

# 最良閾値
best_idx = results_table['accuracy'].idxmax()
best_threshold = results_table.loc[best_idx, 'threshold']
best_accuracy = results_table.loc[best_idx, 'accuracy']

print("\n" + "="*80)
print("【1,000レース 予測結果（選手詳細データ使用）】")
print("="*80)
print(f"\nデータ期間: {results_df['race_date'].min()} 〜 {results_df['race_date'].max()}")
print(f"総レース数: {len(results_df):,}レース")
print(f"高配当レース: {results_df['target_high_payout'].sum()}レース ({results_df['target_high_payout'].mean()*100:.1f}%)")

print("\n閾値別精度:")
print(results_table.to_string(index=False))

print(f"\n【最良閾値】: {best_threshold}")
print(f"  - 精度: {best_accuracy*100:.2f}% ({results_table.loc[best_idx, 'correct']}/{len(results_df)}レース的中)")
print(f"  - F1スコア: {results_table.loc[best_idx, 'f1']:.4f}")
print(f"  - 適合率: {results_table.loc[best_idx, 'precision']*100:.2f}%")
print(f"  - 再現率: {results_table.loc[best_idx, 'recall']*100:.2f}%")

# 保存
results_df.to_csv('analysis/model_outputs/1000_races_predictions_with_riders.csv', index=False)
print(f"\n✓ 保存完了: analysis/model_outputs/1000_races_predictions_with_riders.csv")

summary = {
    'total_races': len(results_df),
    'high_payout_races': int(results_df['target_high_payout'].sum()),
    'best_threshold': float(best_threshold),
    'best_accuracy': float(best_accuracy),
    'prediction_method': 'Rule-based with collected rider data (9,000 riders)',
    'threshold_results': results_table.to_dict('records')
}

with open('analysis/model_outputs/1000_races_summary_with_riders.json', 'w', encoding='utf-8') as f:
    json.dump(summary, f, ensure_ascii=False, indent=2)

print(f"✓ 保存完了: analysis/model_outputs/1000_races_summary_with_riders.json")

print("\n" + "="*80)
print("【完了】")
print("="*80)
print(f"\n精度: {best_accuracy*100:.2f}%")
print(f"\n比較:")
print(f"  - 48,682レース（統計のみ）: 49.97%")
print(f"  - 1,000レース（推定ベース）: 71.90%")
print(f"  - 1,000レース（選手データあり）: {best_accuracy*100:.2f}%")
print("="*80)
