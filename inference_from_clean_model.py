#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
クリーンモデルで推論を行うための関数群
Web UIやCLIから利用される
"""

import json
from pathlib import Path
from typing import Dict, List, Any

import lightgbm as lgb
import numpy as np
import pandas as pd


# Regional line mapping
REGIONAL_LINES = {
    "北海道": "北日本", "青森": "北日本", "岩手": "北日本", "宮城": "北日本",
    "秋田": "北日本", "山形": "北日本", "福島": "北日本",
    "茨城": "関東", "栃木": "関東", "群馬": "関東", "埼玉": "関東",
    "千葉": "関東", "東京": "関東", "神奈川": "関東", "山梨": "関東",
    "新潟": "北陸", "富山": "北陸", "石川": "北陸", "福井": "北陸",
    "長野": "中部", "岐阜": "中部", "静岡": "中部", "愛知": "中部",
    "三重": "近畿", "滋賀": "近畿", "京都": "近畿", "大阪": "近畿",
    "兵庫": "近畿", "奈良": "近畿", "和歌山": "近畿",
    "鳥取": "中国", "島根": "中国", "岡山": "中国", "広島": "中国", "山口": "中国",
    "徳島": "四国", "香川": "四国", "愛媛": "四国", "高知": "四国",
    "福岡": "九州", "佐賀": "九州", "長崎": "九州", "熊本": "九州",
    "大分": "九州", "宮崎": "九州", "鹿児島": "九州", "沖縄": "九州",
}


def normalize_style(kyakusitu: str) -> str:
    """脚質を正規化"""
    s = str(kyakusitu).strip()
    if '逃' in s or '先' in s or '捲' in s:
        return 'nige'
    elif '追' in s or '差' in s or 'マーク' in s:
        return 'tsui'
    elif '両' in s or '自在' in s:
        return 'ryo'
    return 'unknown'


def normalize_grade(kyuhan: str) -> str:
    """級班を正規化"""
    g = str(kyuhan).strip().upper()
    if g in ['SS', 'S1', 'S2', 'A1', 'A2', 'A3', 'L1']:
        return g
    return 'unknown'


def get_regional_line(prefecture: str) -> str:
    """府県からライン（地域）を取得"""
    pref = str(prefecture).strip().replace('　', '')
    return REGIONAL_LINES.get(pref, "その他")


def calculate_entropy(counts: Dict[str, int]) -> float:
    """エントロピー計算"""
    total = sum(counts.values())
    if total == 0:
        return 0.0
    probs = [c / total for c in counts.values() if c > 0]
    return -sum(p * np.log2(p) for p in probs)


def build_features_from_manual_input(race_info: Dict[str, Any]) -> pd.DataFrame:
    """
    手動入力からレース特徴量を構築

    race_info = {
        'race_date': 20250101,  # YYYYMMDD
        'track': '京王閣',
        'keirin_cd': '27',
        'race_no': 7,
        'grade': 'G3',
        'category': 'S級選抜',
        'riders': [
            {
                'name': '山田太郎',
                'prefecture': '東京',
                'grade': 'S1',
                'style': '逃げ',
                'avg_score': 115.20,
            },
            ...
        ]
    }
    """
    riders = race_info.get('riders', [])
    entry_count = len(riders)

    features = {}
    features['race_no'] = int(race_info.get('race_no', 7))

    # Date features
    race_date = int(race_info.get('race_date', 0))
    race_date_str = str(race_date).zfill(8)
    try:
        dt = pd.to_datetime(race_date_str, format='%Y%m%d')
        features['year'] = dt.year
        features['month'] = dt.month
        features['day'] = dt.day
        features['day_of_week'] = dt.dayofweek
        features['is_weekend'] = int(dt.dayofweek >= 5)
    except:
        features['year'] = 0
        features['month'] = 0
        features['day'] = 0
        features['day_of_week'] = 0
        features['is_weekend'] = 0

    features['entry_count'] = entry_count

    # Score statistics
    scores = [r.get('avg_score', 0) for r in riders if r.get('avg_score')]
    if scores:
        scores = pd.Series(scores)
        features['score_mean'] = float(scores.mean())
        features['score_std'] = float(scores.std()) if len(scores) > 1 else 0.0
        features['score_min'] = float(scores.min())
        features['score_max'] = float(scores.max())
        features['score_range'] = features['score_max'] - features['score_min']
        features['score_median'] = float(scores.median())
        features['score_q25'] = float(scores.quantile(0.25))
        features['score_q75'] = float(scores.quantile(0.75))
        features['score_iqr'] = features['score_q75'] - features['score_q25']
        features['score_cv'] = features['score_std'] / (features['score_mean'] + 1e-6)

        top3 = scores.nlargest(min(3, len(scores)))
        bottom3 = scores.nsmallest(min(3, len(scores)))
        features['score_top3_mean'] = float(top3.mean())
        features['score_bottom3_mean'] = float(bottom3.mean())
        features['score_top_bottom_gap'] = features['score_top3_mean'] - features['score_bottom3_mean']

        sorted_scores = scores.sort_values(ascending=False)
        rank1 = float(sorted_scores.iloc[0]) if len(sorted_scores) > 0 else features['score_mean']
        rank2 = float(sorted_scores.iloc[1]) if len(sorted_scores) > 1 else rank1
        rank3 = float(sorted_scores.iloc[2]) if len(sorted_scores) > 2 else rank2

        features['estimated_top3_score_sum'] = rank1 + rank2 + rank3
        features['estimated_favorite_dominance'] = rank1 / (features['score_mean'] + 1e-6)
        features['estimated_favorite_gap'] = rank1 - rank2

        if len(scores) > 3:
            others_mean = float(scores.iloc[3:].mean())
            features['estimated_top3_vs_others'] = features['score_top3_mean'] - others_mean
        else:
            features['estimated_top3_vs_others'] = 0.0
    else:
        for key in ['score_mean', 'score_std', 'score_min', 'score_max', 'score_range',
                    'score_median', 'score_q25', 'score_q75', 'score_iqr', 'score_cv',
                    'score_top3_mean', 'score_bottom3_mean', 'score_top_bottom_gap',
                    'estimated_top3_score_sum', 'estimated_favorite_dominance',
                    'estimated_favorite_gap', 'estimated_top3_vs_others']:
            features[key] = 0.0

    # Style analysis
    styles = [normalize_style(r.get('style', '')) for r in riders]
    style_counts = pd.Series(styles).value_counts().to_dict()

    for style in ['nige', 'tsui', 'ryo', 'unknown']:
        count = style_counts.get(style, 0)
        features[f'style_{style}_count'] = count
        features[f'style_{style}_ratio'] = count / entry_count if entry_count > 0 else 0.0

    style_diversity = len([c for c in style_counts.values() if c > 0])
    features['style_diversity'] = style_diversity
    features['style_entropy'] = calculate_entropy(style_counts)

    ratios = [features[f'style_{s}_ratio'] for s in ['nige', 'tsui', 'ryo'] if features[f'style_{s}_ratio'] > 0]
    features['style_max_ratio'] = max(ratios) if ratios else 0.0
    features['style_min_ratio'] = min(ratios) if ratios else 0.0

    # Grade analysis
    grades = [normalize_grade(r.get('grade', '')) for r in riders]
    grade_counts = pd.Series(grades).value_counts().to_dict()

    for grade in ['SS', 'S1', 'S2', 'A1', 'A2', 'A3', 'L1']:
        count = grade_counts.get(grade, 0)
        features[f'grade_{grade}_count'] = count
        features[f'grade_{grade}_ratio'] = count / entry_count if entry_count > 0 else 0.0

    features['grade_entropy'] = calculate_entropy(grade_counts)
    features['grade_has_mixed'] = int(len(grade_counts) > 1)

    # Line analysis
    prefectures = [r.get('prefecture', '') for r in riders]
    lines = [get_regional_line(p) for p in prefectures]
    line_counts = pd.Series(lines).value_counts().to_dict()

    features['line_count'] = len(line_counts)
    features['line_entropy'] = calculate_entropy(line_counts)

    if line_counts:
        dominant_line_count = max(line_counts.values())
        features['dominant_line_ratio'] = dominant_line_count / entry_count

        # Line score balance
        rider_df = pd.DataFrame([
            {'line': get_regional_line(r.get('prefecture', '')),
             'score': r.get('avg_score', 0)}
            for r in riders if r.get('avg_score')
        ])
        if len(rider_df) > 0:
            line_scores = rider_df.groupby('line')['score'].mean()
            features['line_balance_std'] = float(line_scores.std()) if len(line_scores) > 1 else 0.0
            features['line_score_gap'] = float(line_scores.max() - line_scores.min()) if len(line_scores) > 0 else 0.0
        else:
            features['line_balance_std'] = 0.0
            features['line_score_gap'] = 0.0
    else:
        features['dominant_line_ratio'] = 0.0
        features['line_balance_std'] = 0.0
        features['line_score_gap'] = 0.0

    features['prefecture_unique_count'] = len(set(prefectures))

    return pd.DataFrame([features])


def load_model_and_metadata():
    """Load trained model and metadata"""
    model_dir = Path('analysis/model_outputs')
    model_path = model_dir / 'clean_model_lgbm.txt'
    metadata_path = model_dir / 'clean_model_metadata.json'
    stats_path = model_dir / 'track_category_stats.json'

    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata not found: {metadata_path}")

    model = lgb.Booster(model_file=str(model_path))

    with open(metadata_path, 'r', encoding='utf-8') as f:
        metadata = json.load(f)

    stats = None
    if stats_path.exists():
        with open(stats_path, 'r', encoding='utf-8') as f:
            stats = json.load(f)

    return model, metadata, stats


def predict_race(race_info: Dict[str, Any]) -> Dict[str, Any]:
    """
    レース情報から高配当確率を予測

    Returns:
        {
            'probability': 0.45,
            'confidence': 'high' | 'medium' | 'low',
            'track_adjustment': 0.05,
            'category_adjustment': 0.10,
            'features_used': {...},
        }
    """
    model, metadata, stats = load_model_and_metadata()

    # Build features
    X = build_features_from_manual_input(race_info)

    # Align to model's feature columns
    feature_cols = metadata['feature_columns']
    for col in feature_cols:
        if col not in X.columns:
            X[col] = 0.0

    X = X[feature_cols]

    # Predict base probability
    base_prob = float(model.predict(X)[0])

    # Apply track adjustment
    track_adjustment = 0.0
    if stats and 'track' in stats:
        track = race_info.get('track', '')
        if track in stats['track']:
            track_rate = stats['track'][track]['high_payout_rate']
            global_rate = stats['global_high_payout_rate']
            track_adjustment = track_rate - global_rate

    # Apply category adjustment
    category_adjustment = 0.0
    if stats and 'category' in stats:
        category = race_info.get('category', '')
        if category in stats['category']:
            category_rate = stats['category'][category]['high_payout_rate']
            global_rate = stats['global_high_payout_rate']
            category_adjustment = category_rate - global_rate

    # Final probability
    final_prob = base_prob + track_adjustment * 0.3 + category_adjustment * 0.3
    final_prob = max(0.05, min(0.95, final_prob))  # Clip to reasonable range

    # Determine confidence
    if final_prob >= 0.7:
        confidence = 'high'
    elif final_prob >= 0.4:
        confidence = 'medium'
    else:
        confidence = 'low'

    return {
        'probability': final_prob,
        'base_probability': base_prob,
        'confidence': confidence,
        'track_adjustment': track_adjustment,
        'category_adjustment': category_adjustment,
        'features_summary': {
            'score_cv': float(X['score_cv'].iloc[0]),
            'estimated_favorite_gap': float(X['estimated_favorite_gap'].iloc[0]),
            'line_balance_std': float(X['line_balance_std'].iloc[0]),
        },
    }


if __name__ == "__main__":
    # Test
    test_race = {
        'race_date': 20250101,
        'track': '京王閣',
        'keirin_cd': '27',
        'race_no': 7,
        'grade': 'G3',
        'category': 'S級選抜',
        'riders': [
            {'name': '選手A', 'prefecture': '東京', 'grade': 'S1', 'style': '逃げ', 'avg_score': 115.2},
            {'name': '選手B', 'prefecture': '神奈川', 'grade': 'S1', 'style': '追込', 'avg_score': 113.5},
            {'name': '選手C', 'prefecture': '埼玉', 'grade': 'S2', 'style': '逃げ', 'avg_score': 110.8},
            {'name': '選手D', 'prefecture': '千葉', 'grade': 'S2', 'style': '追込', 'avg_score': 109.2},
            {'name': '選手E', 'prefecture': '大阪', 'grade': 'A1', 'style': '両', 'avg_score': 108.5},
            {'name': '選手F', 'prefecture': '福岡', 'grade': 'A1', 'style': '追込', 'avg_score': 107.1},
            {'name': '選手G', 'prefecture': '愛知', 'grade': 'A2', 'style': '逃げ', 'avg_score': 105.8},
        ]
    }

    result = predict_race(test_race)
    print(json.dumps(result, ensure_ascii=False, indent=2))
