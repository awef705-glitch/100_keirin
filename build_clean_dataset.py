#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
競輪高配当予測：クリーンなデータセット構築
事後データを完全に除外し、事前データのみで特徴量を構築
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Regional line mapping (競輪のライン: 地域別共同作戦)
REGIONAL_LINES = {
    # 北日本ライン
    "北海道": "北日本", "青森": "北日本", "岩手": "北日本", "宮城": "北日本",
    "秋田": "北日本", "山形": "北日本", "福島": "北日本",
    # 関東ライン
    "茨城": "関東", "栃木": "関東", "群馬": "関東", "埼玉": "関東",
    "千葉": "関東", "東京": "関東", "神奈川": "関東", "山梨": "関東",
    # 北陸ライン
    "新潟": "北陸", "富山": "北陸", "石川": "北陸", "福井": "北陸",
    # 中部ライン
    "長野": "中部", "岐阜": "中部", "静岡": "中部", "愛知": "中部",
    # 近畿ライン
    "三重": "近畿", "滋賀": "近畿", "京都": "近畿", "大阪": "近畿",
    "兵庫": "近畿", "奈良": "近畿", "和歌山": "近畿",
    # 中国ライン
    "鳥取": "中国", "島根": "中国", "岡山": "中国", "広島": "中国", "山口": "中国",
    # 四国ライン
    "徳島": "四国", "香川": "四国", "愛媛": "四国", "高知": "四国",
    # 九州ライン
    "福岡": "九州", "佐賀": "九州", "長崎": "九州", "熊本": "九州",
    "大分": "九州", "宮崎": "九州", "鹿児島": "九州", "沖縄": "九州",
}

def get_regional_line(prefecture):
    """府県からライン（地域）を取得"""
    pref = str(prefecture).strip().replace('　', '')
    return REGIONAL_LINES.get(pref, "その他")


def normalize_style(kyakusitu):
    """脚質を正規化"""
    s = str(kyakusitu).strip()
    if '逃' in s or '先' in s or '捲' in s:
        return 'nige'
    elif '追' in s or '差' in s or 'マーク' in s:
        return 'tsui'
    elif '両' in s or '自在' in s:
        return 'ryo'
    return 'unknown'


def normalize_grade(kyuhan):
    """級班を正規化"""
    g = str(kyuhan).strip().upper()
    if g in ['SS', 'S1', 'S2', 'A1', 'A2', 'A3', 'L1']:
        return g
    return 'unknown'


def calculate_entropy(counts):
    """エントロピー計算（多様性指標）"""
    total = sum(counts.values())
    if total == 0:
        return 0.0
    probs = [c / total for c in counts.values() if c > 0]
    return -sum(p * np.log2(p) for p in probs)


def aggregate_race_features(race_df):
    """
    レース単位で選手情報を集約し、事前データのみの特徴量を構築
    """
    # Convert race_date to integer YYYYMMDD format
    race_date_raw = race_df['race_date'].iloc[0]
    if isinstance(race_date_raw, str):
        # Convert '2024-01-01' to 20240101
        race_date_int = int(race_date_raw.replace('-', ''))
    else:
        race_date_int = int(race_date_raw)

    # Basic race info
    track = str(race_df['track'].iloc[0])
    keirin_cd = str(race_df['keirin_cd'].iloc[0]) if 'keirin_cd' in race_df.columns else track
    category = str(race_df['category'].iloc[0]) if 'category' in race_df.columns else ''

    race_info = {
        'race_date': race_date_int,
        'track': track,
        'keirin_cd': keirin_cd,
        'race_no': int(race_df['race_no'].iloc[0]),
        'grade': str(race_df['grade'].iloc[0]),
        'category': category,
    }

    # Target variable
    if 'high_payout_flag' in race_df.columns:
        race_info['target_high_payout'] = int(race_df['high_payout_flag'].iloc[0])
    elif 'trifecta_payout_value' in race_df.columns:
        payout = float(race_df['trifecta_payout_value'].iloc[0])
        race_info['target_high_payout'] = int(payout >= 10000)
    else:
        race_info['target_high_payout'] = 0

    # Entry count
    entry_count = len(race_df)
    race_info['entry_count'] = entry_count

    # === 競走得点の統計 ===
    scores = pd.to_numeric(race_df['heikinTokuten'], errors='coerce').dropna()
    if len(scores) > 0:
        race_info['score_mean'] = float(scores.mean())
        race_info['score_std'] = float(scores.std()) if len(scores) > 1 else 0.0
        race_info['score_min'] = float(scores.min())
        race_info['score_max'] = float(scores.max())
        race_info['score_range'] = race_info['score_max'] - race_info['score_min']
        race_info['score_median'] = float(scores.median())
        race_info['score_q25'] = float(scores.quantile(0.25))
        race_info['score_q75'] = float(scores.quantile(0.75))
        race_info['score_iqr'] = race_info['score_q75'] - race_info['score_q25']
        race_info['score_cv'] = race_info['score_std'] / (race_info['score_mean'] + 1e-6)

        # Top3 vs Bottom3
        top3 = scores.nlargest(min(3, len(scores)))
        bottom3 = scores.nsmallest(min(3, len(scores)))
        race_info['score_top3_mean'] = float(top3.mean())
        race_info['score_bottom3_mean'] = float(bottom3.mean())
        race_info['score_top_bottom_gap'] = race_info['score_top3_mean'] - race_info['score_bottom3_mean']

        # 推定人気度（得点ベース）
        sorted_scores = scores.sort_values(ascending=False)
        rank1 = float(sorted_scores.iloc[0]) if len(sorted_scores) > 0 else race_info['score_mean']
        rank2 = float(sorted_scores.iloc[1]) if len(sorted_scores) > 1 else rank1
        rank3 = float(sorted_scores.iloc[2]) if len(sorted_scores) > 2 else rank2

        race_info['estimated_top3_score_sum'] = rank1 + rank2 + rank3
        race_info['estimated_favorite_dominance'] = rank1 / (race_info['score_mean'] + 1e-6)
        race_info['estimated_favorite_gap'] = rank1 - rank2

        if len(scores) > 3:
            others_mean = float(scores.iloc[3:].mean())
            race_info['estimated_top3_vs_others'] = race_info['score_top3_mean'] - others_mean
        else:
            race_info['estimated_top3_vs_others'] = 0.0
    else:
        # Default values if no scores
        for key in ['score_mean', 'score_std', 'score_min', 'score_max', 'score_range',
                    'score_median', 'score_q25', 'score_q75', 'score_iqr', 'score_cv',
                    'score_top3_mean', 'score_bottom3_mean', 'score_top_bottom_gap',
                    'estimated_top3_score_sum', 'estimated_favorite_dominance',
                    'estimated_favorite_gap', 'estimated_top3_vs_others']:
            race_info[key] = 0.0

    # === 脚質の分析 ===
    styles = race_df['kyakusitu'].apply(normalize_style)
    style_counts = styles.value_counts().to_dict()

    for style in ['nige', 'tsui', 'ryo', 'unknown']:
        count = style_counts.get(style, 0)
        race_info[f'style_{style}_count'] = count
        race_info[f'style_{style}_ratio'] = count / entry_count if entry_count > 0 else 0.0

    # 脚質の多様性
    style_diversity = len([c for c in style_counts.values() if c > 0])
    race_info['style_diversity'] = style_diversity
    race_info['style_entropy'] = calculate_entropy(style_counts)

    ratios = [r for r in [race_info[f'style_{s}_ratio'] for s in ['nige', 'tsui', 'ryo']] if r > 0]
    race_info['style_max_ratio'] = max(ratios) if ratios else 0.0
    race_info['style_min_ratio'] = min(ratios) if ratios else 0.0

    # === 級班の分析 ===
    grades = race_df['kyuhan'].apply(normalize_grade)
    grade_counts = grades.value_counts().to_dict()

    for grade in ['SS', 'S1', 'S2', 'A1', 'A2', 'A3', 'L1']:
        count = grade_counts.get(grade, 0)
        race_info[f'grade_{grade}_count'] = count
        race_info[f'grade_{grade}_ratio'] = count / entry_count if entry_count > 0 else 0.0

    race_info['grade_entropy'] = calculate_entropy(grade_counts)
    race_info['grade_has_mixed'] = int(len(grade_counts) > 1)

    # === ラインの分析（地域別） ===
    race_df['line'] = race_df['entry_prefecture'].apply(get_regional_line)
    line_counts = race_df['line'].value_counts().to_dict()

    race_info['line_count'] = len(line_counts)
    race_info['line_entropy'] = calculate_entropy(line_counts)

    if line_counts:
        dominant_line_count = max(line_counts.values())
        race_info['dominant_line_ratio'] = dominant_line_count / entry_count

        # ライン別の平均得点差
        line_scores = race_df.groupby('line')['heikinTokuten'].apply(
            lambda x: pd.to_numeric(x, errors='coerce').mean()
        )
        race_info['line_balance_std'] = float(line_scores.std()) if len(line_scores) > 1 else 0.0
        race_info['line_score_gap'] = float(line_scores.max() - line_scores.min()) if len(line_scores) > 0 else 0.0
    else:
        race_info['dominant_line_ratio'] = 0.0
        race_info['line_balance_std'] = 0.0
        race_info['line_score_gap'] = 0.0

    # === 府県の多様性 ===
    race_info['prefecture_unique_count'] = race_df['entry_prefecture'].nunique()

    # === 脚質カウント（B関連）の分析 ===
    # 逃げ、捲り、差し、マーク、バックの勝利回数は選手の戦績（事前データ）
    b_columns = ['nigeCnt', 'makuriCnt', 'sasiCnt', 'markCnt', 'backCnt']

    for b_col in b_columns:
        if b_col in race_df.columns:
            b_values = pd.to_numeric(race_df[b_col], errors='coerce').fillna(0)

            race_info[f'{b_col}_mean'] = float(b_values.mean())
            race_info[f'{b_col}_std'] = float(b_values.std()) if len(b_values) > 1 else 0.0
            race_info[f'{b_col}_max'] = float(b_values.max())
            race_info[f'{b_col}_sum'] = float(b_values.sum())

            # CV (coefficient of variation)
            if race_info[f'{b_col}_mean'] > 0:
                race_info[f'{b_col}_cv'] = race_info[f'{b_col}_std'] / race_info[f'{b_col}_mean']
            else:
                race_info[f'{b_col}_cv'] = 0.0
        else:
            # Default values if column doesn't exist
            for suffix in ['mean', 'std', 'max', 'sum', 'cv']:
                race_info[f'{b_col}_{suffix}'] = 0.0

    # 合計経験値（全脚質の合計）
    total_b_experience = sum([race_info[f'{b}_sum'] for b in b_columns])
    race_info['total_b_experience'] = total_b_experience

    # 経験値の多様性（どの脚質が強いか）
    if total_b_experience > 0:
        b_distribution = {b: race_info[f'{b}_sum'] for b in b_columns}
        race_info['b_experience_entropy'] = calculate_entropy(b_distribution)
    else:
        race_info['b_experience_entropy'] = 0.0

    # === レース番号（時間帯） ===
    race_info['race_no_int'] = int(race_df['race_no'].iloc[0])

    # === 日付関連 ===
    race_date_str = str(int(race_info['race_date'])).zfill(8)
    try:
        dt = pd.to_datetime(race_date_str, format='%Y%m%d')
        race_info['year'] = dt.year
        race_info['month'] = dt.month
        race_info['day'] = dt.day
        race_info['day_of_week'] = dt.dayofweek
        race_info['is_weekend'] = int(dt.dayofweek >= 5)
    except:
        race_info['year'] = 0
        race_info['month'] = 0
        race_info['day'] = 0
        race_info['day_of_week'] = 0
        race_info['is_weekend'] = 0

    return race_info


def main():
    print("=" * 80)
    print("競輪高配当予測：クリーンなデータセット構築")
    print("=" * 80)

    # Load raw data
    input_file = Path('data/keirin_training_dataset_20240101_20240331.csv')
    output_file = Path('data/clean_training_dataset.csv')

    if not input_file.exists():
        print(f"❌ Error: {input_file} not found")
        sys.exit(1)

    print(f"\n📂 Loading data from: {input_file}")
    df = pd.read_csv(input_file)
    print(f"✓ Loaded {len(df):,} rows, {len(df.columns)} columns")

    # Identify unique races
    print(f"\n🔍 Identifying unique races...")
    race_groups = df.groupby(['race_date', 'track', 'race_no'])
    print(f"✓ Found {len(race_groups):,} unique races")

    # Aggregate features for each race
    print(f"\n⚙️  Aggregating race-level features (事前データのみ)...")
    race_features = []

    for idx, (race_id, race_df) in enumerate(race_groups, 1):
        if idx % 500 == 0:
            print(f"  Processing race {idx:,} / {len(race_groups):,}...")

        try:
            features = aggregate_race_features(race_df)
            race_features.append(features)
        except Exception as e:
            print(f"⚠️  Warning: Failed to process race {race_id}: {e}")
            continue

    # Create final dataset
    print(f"\n📊 Creating final dataset...")
    clean_df = pd.DataFrame(race_features)

    # Sort by date
    clean_df = clean_df.sort_values(['race_date', 'keirin_cd', 'race_no_int']).reset_index(drop=True)

    # Save
    output_file.parent.mkdir(parents=True, exist_ok=True)
    clean_df.to_csv(output_file, index=False)

    print(f"\n✅ Clean dataset created!")
    print(f"   Output: {output_file}")
    print(f"   Races: {len(clean_df):,}")
    print(f"   Features: {len(clean_df.columns)}")
    print(f"   High payout rate: {clean_df['target_high_payout'].mean():.1%}")

    # Feature summary
    print(f"\n📈 Feature Categories:")
    feature_groups = {
        '得点統計': [c for c in clean_df.columns if 'score' in c],
        '脚質分析': [c for c in clean_df.columns if 'style' in c],
        '級班分析': [c for c in clean_df.columns if 'grade' in c and 'flag' not in c],
        'ライン分析': [c for c in clean_df.columns if 'line' in c],
        '推定人気': [c for c in clean_df.columns if 'estimated' in c],
        '基本情報': [c for c in clean_df.columns if c in ['entry_count', 'race_no_int', 'year', 'month', 'day_of_week', 'is_weekend', 'prefecture_unique_count']],
    }

    for group_name, features in feature_groups.items():
        print(f"   {group_name}: {len(features)} features")

    print("\n" + "=" * 80)
    print("✓ Complete! 事後データは完全に除外されています。")
    print("=" * 80)


if __name__ == "__main__":
    main()
