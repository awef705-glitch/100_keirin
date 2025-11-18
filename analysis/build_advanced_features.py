#!/usr/bin/env python3
"""
高度な特徴量エンジニアリングスクリプト

選手間の格差、脚質構成、級班構成などの高度な特徴量を追加
"""

import pandas as pd
import numpy as np
from scipy.stats import entropy

def add_advanced_rider_features(entries_df):
    """
    エントリデータから高度な選手関連特徴量を作成

    Args:
        entries_df: エントリデータ（選手個別情報）

    Returns:
        DataFrame: レースレベルの高度特徴量
    """
    features_list = []

    # race_encp（レース識別子）でグループ化
    for race_encp, group in entries_df.groupby('race_encp'):
        race_date = group['race_date'].iloc[0]
        track = group['track'].iloc[0]
        race_no = group['race_no'].iloc[0]

        features = {
            'race_encp': race_encp,
            'race_date': race_date,
            'track': track,
            'race_no': race_no,
        }

        # 選手得点の統計
        scores = group['heikinTokuten'].values
        scores_sorted = np.sort(scores)[::-1]  # 降順

        # 1. 選手間の格差特徴量
        if len(scores_sorted) >= 2:
            features['score_gap_max_min'] = scores_sorted[0] - scores_sorted[-1]  # 最強vs最弱
            features['score_gap_top2'] = scores_sorted[0] - scores_sorted[1]  # 1位vs2位
        else:
            features['score_gap_max_min'] = 0
            features['score_gap_top2'] = 0

        if len(scores_sorted) >= 3:
            features['score_top3_mean'] = scores_sorted[:3].mean()  # 上位3選手平均
            features['score_gap_top3_bottom'] = scores_sorted[:3].mean() - scores_sorted[3:].mean()
        else:
            features['score_top3_mean'] = scores.mean()
            features['score_gap_top3_bottom'] = 0

        # 得点の偏り（最強選手が圧倒的か）
        features['score_dominance'] = scores_sorted[0] / (scores.mean() + 1e-6)

        # 得点の集中度（ジニ係数風）
        if len(scores) > 1:
            features['score_concentration'] = np.std(scores) / (np.mean(scores) + 1e-6)
        else:
            features['score_concentration'] = 0

        # 2. 脚質構成の特徴量
        # kyakusitu（脚質）のマッピング
        kyakusitu_map = {'逃': 'nige', '追': 'oi', '両': 'ryo', '自': 'ji'}
        kyakusitu_counts = {}
        for k in ['nige', 'oi', 'ryo', 'ji']:
            kyakusitu_counts[k] = 0

        for val in group['kyakusitu'].values:
            if pd.notna(val) and val in kyakusitu_map:
                kyakusitu_counts[kyakusitu_map[val]] += 1

        features['kyakusitu_nige_count'] = kyakusitu_counts['nige']
        features['kyakusitu_oi_count'] = kyakusitu_counts['oi']
        features['kyakusitu_ryo_count'] = kyakusitu_counts['ryo']
        features['kyakusitu_ji_count'] = kyakusitu_counts['ji']

        # 脚質の多様性（エントロピー）
        total = sum(kyakusitu_counts.values())
        if total > 0:
            probs = [v/total for v in kyakusitu_counts.values() if v > 0]
            features['kyakusitu_entropy'] = entropy(probs) if len(probs) > 1 else 0
        else:
            features['kyakusitu_entropy'] = 0

        # 逃げ選手の割合
        entry_count = len(group)
        features['kyakusitu_nige_ratio'] = kyakusitu_counts['nige'] / entry_count if entry_count > 0 else 0

        # 3. 級班構成の特徴量
        kyuhan_counts = {'S1': 0, 'S2': 0, 'A1': 0, 'A2': 0, 'A3': 0}
        for val in group['kyuhan'].values:
            if pd.notna(val) and val in kyuhan_counts:
                kyuhan_counts[val] += 1

        features['kyuhan_S1_count'] = kyuhan_counts['S1']
        features['kyuhan_S2_count'] = kyuhan_counts['S2']
        features['kyuhan_A1_count'] = kyuhan_counts['A1']
        features['kyuhan_A2_count'] = kyuhan_counts['A2']
        features['kyuhan_A3_count'] = kyuhan_counts['A3']

        # S級選手の割合
        s_class_count = kyuhan_counts['S1'] + kyuhan_counts['S2']
        features['kyuhan_S_ratio'] = s_class_count / entry_count if entry_count > 0 else 0

        # 級班の多様性
        total_kyuhan = sum(kyuhan_counts.values())
        if total_kyuhan > 0:
            probs = [v/total_kyuhan for v in kyuhan_counts.values() if v > 0]
            features['kyuhan_entropy'] = entropy(probs) if len(probs) > 1 else 0
        else:
            features['kyuhan_entropy'] = 0

        # 4. 戦法回数の特徴量（選手別のばらつき）
        for col in ['nigeCnt', 'makuriCnt', 'sasiCnt', 'markCnt', 'backCnt']:
            vals = group[col].values
            # トップ選手とボトム選手の差
            if len(vals) >= 2:
                features[f'{col}_gap_max_min'] = vals.max() - vals.min()
            else:
                features[f'{col}_gap_max_min'] = 0

        features_list.append(features)

    return pd.DataFrame(features_list)


def main():
    print("=== 高度特徴量の構築開始 ===\n")

    # エントリデータの読み込み
    print("1. エントリデータ読み込み中...")
    entries = pd.read_csv('data/keirin_race_detail_entries_20240101_20241231_combined.csv')
    print(f"   エントリ数: {len(entries):,}行")

    # 高度特徴量の作成
    print("\n2. 高度特徴量の作成中...")
    advanced_features = add_advanced_rider_features(entries)
    print(f"   作成したレース数: {len(advanced_features):,}")
    print(f"   新特徴量数: {len(advanced_features.columns) - 4}")  # race_encp等を除く

    # trackカラムを数値に変換（マッピング）
    track_mapping = {}
    for idx, track_name in enumerate(sorted(advanced_features['track'].unique())):
        track_mapping[track_name] = idx
    advanced_features['track'] = advanced_features['track'].map(track_mapping)

    # 既存の訓練データに結合
    print("\n3. 既存訓練データと結合中...")
    training_data = pd.read_csv('data/clean_training_dataset.csv')
    print(f"   既存訓練データ: {len(training_data):,}行, {len(training_data.columns)}列")

    # race_dateとtrackでマージ（race_encpがない場合の対応）
    # race_no_intを文字列化してマージキーとする
    training_data['race_no_str'] = training_data['race_no_int'].astype(str)
    advanced_features['race_no_str'] = advanced_features['race_no'].astype(str)

    merged = training_data.merge(
        advanced_features,
        on=['race_date', 'track', 'race_no_str'],
        how='left'
    )

    # race_encpとrace_no_strは削除
    merged = merged.drop(columns=['race_encp', 'race_no', 'race_no_str'], errors='ignore')

    print(f"   結合後: {len(merged):,}行, {len(merged.columns)}列")

    # NaN処理
    print("\n4. 欠損値処理中...")
    new_feature_cols = [c for c in advanced_features.columns if c not in ['race_encp', 'race_date', 'track', 'race_no', 'race_no_str']]
    for col in new_feature_cols:
        if col in merged.columns:
            merged[col] = merged[col].fillna(0)

    nan_count = merged.isna().sum().sum()
    print(f"   残存NaN数: {nan_count}")

    # 重要度0の特徴量を削除
    print("\n5. 重要度0の特徴量を削除中...")
    zero_importance_features = ['nigeCnt_min', 'makuriCnt_min', 'backCnt_min', 'ozz_flg', 'entry_intensity', 'year']
    removed = []
    for feat in zero_importance_features:
        if feat in merged.columns:
            merged = merged.drop(columns=[feat])
            removed.append(feat)
    print(f"   削除した特徴量: {removed}")

    # 保存
    output_file = 'data/training_dataset_enhanced_v5.csv'
    print(f"\n6. 保存中: {output_file}")
    merged.to_csv(output_file, index=False)
    print(f"   ✅ 完了: {len(merged):,}行, {len(merged.columns)}列")

    # 新特徴量のサマリー
    print("\n=== 新特徴量サマリー ===")
    for col in new_feature_cols:
        if col in merged.columns:
            print(f"  {col}: mean={merged[col].mean():.3f}, std={merged[col].std():.3f}")


if __name__ == '__main__':
    main()
