#!/usr/bin/env python3
"""
選手個別データの徹底活用

現在は集約統計（mean, std, CV）のみだが、
個別の選手情報（最強選手、2番手、最弱選手）が高配当予測に重要
"""

import pandas as pd
import numpy as np


def add_individual_rider_features(entries_df):
    """
    選手個別の特徴量を作成

    最強選手、2番手、3番手、最弱選手の詳細情報を特徴量化
    """
    features_list = []

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

        # 得点でソート
        sorted_riders = group.sort_values('heikinTokuten', ascending=False)

        # 最強選手（1位）の情報
        if len(sorted_riders) >= 1:
            top1 = sorted_riders.iloc[0]
            features['top1_score'] = top1['heikinTokuten']
            features['top1_syaban'] = pd.to_numeric(top1['syaban'], errors='coerce')
            features['top1_kyakusitu'] = top1['kyakusitu'] if pd.notna(top1['kyakusitu']) else 'unknown'
            features['top1_kyuhan'] = top1['kyuhan'] if pd.notna(top1['kyuhan']) else 'unknown'
            features['top1_nige'] = top1['nigeCnt']
            features['top1_makuri'] = top1['makuriCnt']
            features['top1_sasi'] = top1['sasiCnt']
            features['top1_back'] = top1['backCnt']
        else:
            features['top1_score'] = 0
            features['top1_syaban'] = 0
            features['top1_kyakusitu'] = 'unknown'
            features['top1_kyuhan'] = 'unknown'
            features['top1_nige'] = 0
            features['top1_makuri'] = 0
            features['top1_sasi'] = 0
            features['top1_back'] = 0

        # 2番手選手の情報
        if len(sorted_riders) >= 2:
            top2 = sorted_riders.iloc[1]
            features['top2_score'] = top2['heikinTokuten']
            features['top2_syaban'] = pd.to_numeric(top2['syaban'], errors='coerce')
            features['top2_kyakusitu'] = top2['kyakusitu'] if pd.notna(top2['kyakusitu']) else 'unknown'
            features['top2_kyuhan'] = top2['kyuhan'] if pd.notna(top2['kyuhan']) else 'unknown'
        else:
            features['top2_score'] = 0
            features['top2_syaban'] = 0
            features['top2_kyakusitu'] = 'unknown'
            features['top2_kyuhan'] = 'unknown'

        # 3番手選手の情報
        if len(sorted_riders) >= 3:
            top3 = sorted_riders.iloc[2]
            features['top3_score'] = top3['heikinTokuten']
            features['top3_syaban'] = pd.to_numeric(top3['syaban'], errors='coerce')
        else:
            features['top3_score'] = 0
            features['top3_syaban'] = 0

        # 最弱選手の情報
        if len(sorted_riders) >= 1:
            bottom = sorted_riders.iloc[-1]
            features['bottom_score'] = bottom['heikinTokuten']
            features['bottom_syaban'] = pd.to_numeric(bottom['syaban'], errors='coerce')
        else:
            features['bottom_score'] = 0
            features['bottom_syaban'] = 0

        # 選手間の関係性特徴量
        if len(sorted_riders) >= 2:
            # 1位と2位の差（支配度）
            features['top1_top2_gap'] = features['top1_score'] - features['top2_score']
            features['top1_top2_ratio'] = features['top1_score'] / (features['top2_score'] + 1e-6)

            # 車番の関係（1位と2位が隣接しているか）
            if pd.notna(features['top1_syaban']) and pd.notna(features['top2_syaban']):
                features['top1_top2_syaban_diff'] = abs(features['top1_syaban'] - features['top2_syaban'])
                features['top1_top2_adjacent'] = 1 if features['top1_top2_syaban_diff'] == 1 else 0
            else:
                features['top1_top2_syaban_diff'] = 0
                features['top1_top2_adjacent'] = 0

        # 上位3選手の車番の散らばり
        if len(sorted_riders) >= 3:
            top3_syaban = [features['top1_syaban'], features['top2_syaban'], features['top3_syaban']]
            top3_syaban = [s for s in top3_syaban if pd.notna(s) and s > 0]
            if len(top3_syaban) >= 2:
                features['top3_syaban_std'] = np.std(top3_syaban)
                features['top3_syaban_range'] = max(top3_syaban) - min(top3_syaban)
            else:
                features['top3_syaban_std'] = 0
                features['top3_syaban_range'] = 0

        # 最強選手が内枠（1-3番）か外枠（7-9番）か
        if pd.notna(features['top1_syaban']):
            features['top1_is_inner'] = 1 if features['top1_syaban'] <= 3 else 0
            features['top1_is_outer'] = 1 if features['top1_syaban'] >= 7 else 0
        else:
            features['top1_is_inner'] = 0
            features['top1_is_outer'] = 0

        # 脚質の組み合わせ（最強選手と2番手）
        kyakusitu_combo = f"{features['top1_kyakusitu']}_{features['top2_kyakusitu']}"
        features['top1_top2_kyakusitu_combo'] = kyakusitu_combo

        # 級班の組み合わせ（最強選手と2番手）
        kyuhan_combo = f"{features['top1_kyuhan']}_{features['top2_kyuhan']}"
        features['top1_top2_kyuhan_combo'] = kyuhan_combo

        # 最強選手の戦法の偏り
        top1_total_cnt = features['top1_nige'] + features['top1_makuri'] + features['top1_sasi'] + features['top1_back']
        if top1_total_cnt > 0:
            features['top1_nige_ratio'] = features['top1_nige'] / top1_total_cnt
            features['top1_back_ratio'] = features['top1_back'] / top1_total_cnt
        else:
            features['top1_nige_ratio'] = 0
            features['top1_back_ratio'] = 0

        features_list.append(features)

    return pd.DataFrame(features_list)


def main():
    print("=== 選手個別データ特徴量の構築 ===\n")

    # エントリデータ読み込み
    print("1. エントリデータ読み込み中...")
    entries = pd.read_csv('data/keirin_race_detail_entries_20240101_20241231_combined.csv')
    print(f"   エントリ数: {len(entries):,}行")

    # 個別特徴量作成
    print("\n2. 選手個別特徴量の作成中...")
    individual_features = add_individual_rider_features(entries)
    print(f"   作成したレース数: {len(individual_features):,}")
    print(f"   新特徴量数: {len(individual_features.columns) - 4}")

    # trackを数値化
    track_mapping = {}
    for idx, track_name in enumerate(sorted(individual_features['track'].unique())):
        track_mapping[track_name] = idx
    individual_features['track'] = individual_features['track'].map(track_mapping)

    # V5データに結合
    print("\n3. V5データと結合中...")
    v5_df = pd.read_csv('data/training_dataset_enhanced_v5.csv')
    print(f"   V5データ: {len(v5_df):,}行, {len(v5_df.columns)}列")

    # マージキー作成
    v5_df['race_no_str'] = v5_df['race_no_int'].astype(str)
    individual_features['race_no_str'] = individual_features['race_no'].astype(str)

    # 結合
    merged = v5_df.merge(
        individual_features,
        on=['race_date', 'track', 'race_no_str'],
        how='left'
    )

    # 不要カラム削除
    merged = merged.drop(columns=['race_encp', 'race_no', 'race_no_str'], errors='ignore')

    print(f"   結合後: {len(merged):,}行, {len(merged.columns)}列")

    # カテゴリカル特徴量のエンコーディング
    print("\n4. カテゴリカル特徴量のエンコーディング中...")
    from sklearn.preprocessing import LabelEncoder

    cat_cols = ['top1_kyakusitu', 'top1_kyuhan', 'top2_kyakusitu', 'top2_kyuhan',
                'top1_top2_kyakusitu_combo', 'top1_top2_kyuhan_combo']

    for col in cat_cols:
        if col in merged.columns:
            le = LabelEncoder()
            merged[col] = le.fit_transform(merged[col].astype(str))

    # NaN処理
    print("\n5. 欠損値処理中...")
    numeric_cols = merged.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        merged[col] = merged[col].fillna(0)

    nan_count = merged.isna().sum().sum()
    print(f"   残存NaN数: {nan_count}")

    # 保存
    output_file = 'data/training_dataset_v7_individual.csv'
    print(f"\n6. 保存中: {output_file}")
    merged.to_csv(output_file, index=False)
    print(f"   ✅ 完了: {len(merged):,}行, {len(merged.columns)}列")

    # 新特徴量サマリー
    print("\n=== 新特徴量サマリー ===")
    new_cols = ['top1_score', 'top1_syaban', 'top2_score', 'top1_top2_gap',
                'top1_top2_ratio', 'top1_is_inner', 'top1_nige_ratio',
                'top3_syaban_std', 'top1_top2_adjacent']

    for col in new_cols:
        if col in merged.columns and merged[col].dtype in [np.float64, np.int64]:
            print(f"  {col}: mean={merged[col].mean():.3f}, std={merged[col].std():.3f}, non-zero={(merged[col]!=0).sum()}")


if __name__ == '__main__':
    main()
