#!/usr/bin/env python3
"""
Version 6 超高度特徴量エンジニアリング

1. 車番・枠番の統計特徴量
2. ターゲットエンコーディング（会場別・カテゴリ別の高配当率）
3. 級班変化の特徴量
4. 高度な統計量（歪度、尖度、パーセンタイル）
5. カテゴリカル特徴量の組み合わせ
"""

import pandas as pd
import numpy as np
from scipy.stats import skew, kurtosis
from sklearn.model_selection import TimeSeriesSplit


def add_position_features(entries_df):
    """車番・枠番の統計特徴量"""
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

        # 車番の統計（数値として扱える場合）
        try:
            syaban_vals = pd.to_numeric(group['syaban'], errors='coerce').dropna()
            if len(syaban_vals) > 0:
                features['syaban_mean'] = syaban_vals.mean()
                features['syaban_std'] = syaban_vals.std() if len(syaban_vals) > 1 else 0
        except:
            features['syaban_mean'] = 0
            features['syaban_std'] = 0

        # 内枠（1-3番）の選手の平均得点
        inner_riders = group[pd.to_numeric(group['syaban'], errors='coerce').isin([1, 2, 3])]
        outer_riders = group[pd.to_numeric(group['syaban'], errors='coerce').isin([5, 6, 7, 8, 9])]

        features['inner_avg_score'] = inner_riders['heikinTokuten'].mean() if len(inner_riders) > 0 else 0
        features['outer_avg_score'] = outer_riders['heikinTokuten'].mean() if len(outer_riders) > 0 else 0
        features['inner_outer_gap'] = features['inner_avg_score'] - features['outer_avg_score']

        # 内枠に強い選手がいるか（上位3選手が内枠にいる割合）
        scores_sorted = group.nlargest(3, 'heikinTokuten')
        inner_top3 = scores_sorted[pd.to_numeric(scores_sorted['syaban'], errors='coerce').isin([1, 2, 3])]
        features['top3_inner_ratio'] = len(inner_top3) / 3 if len(scores_sorted) >= 3 else 0

        features_list.append(features)

    return pd.DataFrame(features_list)


def add_kyuhan_change_features(entries_df):
    """級班変化の特徴量"""
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

        # 級班マッピング（数値化）
        kyuhan_map = {'S1': 5, 'S2': 4, 'A1': 3, 'A2': 2, 'A3': 1}

        upgraded = 0  # 昇級した選手数
        downgraded = 0  # 降級した選手数
        no_change = 0

        for _, rider in group.iterrows():
            prev = rider.get('prevKyuhan')
            curr = rider.get('kyuhan')

            if pd.notna(prev) and pd.notna(curr) and prev in kyuhan_map and curr in kyuhan_map:
                prev_val = kyuhan_map[prev]
                curr_val = kyuhan_map[curr]

                if curr_val > prev_val:
                    upgraded += 1
                elif curr_val < prev_val:
                    downgraded += 1
                else:
                    no_change += 1

        total = upgraded + downgraded + no_change
        features['upgraded_count'] = upgraded
        features['downgraded_count'] = downgraded
        features['kyuhan_change_ratio'] = (upgraded + downgraded) / total if total > 0 else 0
        features['upgrade_ratio'] = upgraded / total if total > 0 else 0

        features_list.append(features)

    return pd.DataFrame(features_list)


def add_advanced_statistics(df):
    """高度な統計量（歪度、尖度、パーセンタイル）"""

    # 得点の高度統計
    score_cols = ['heikinTokuten_mean', 'heikinTokuten_std', 'heikinTokuten_min', 'heikinTokuten_max']

    # レースごとに選手得点を再構築（簡易版：既存の統計から推定）
    # 実際のデータがないので、既存特徴量から派生

    # 得点分布の歪度推定（CV値から推定）
    if 'heikinTokuten_cv' in df.columns:
        df['score_skewness_proxy'] = df['heikinTokuten_cv'] * (df['heikinTokuten_max'] - df['heikinTokuten_mean']) / (df['heikinTokuten_std'] + 1e-6)

    # 得点分布の尖度推定
    if 'heikinTokuten_range' in df.columns:
        df['score_kurtosis_proxy'] = (df['heikinTokuten_range'] / (df['heikinTokuten_std'] + 1e-6)) ** 2

    # パーセンタイル推定（25%, 75%）
    # 正規分布を仮定して推定
    if 'heikinTokuten_mean' in df.columns and 'heikinTokuten_std' in df.columns:
        df['score_q25'] = df['heikinTokuten_mean'] - 0.674 * df['heikinTokuten_std']
        df['score_q75'] = df['heikinTokuten_mean'] + 0.674 * df['heikinTokuten_std']
        df['score_iqr'] = df['score_q75'] - df['score_q25']

    # 各戦法の偏り（歪度的な指標）
    for style in ['nigeCnt', 'makuriCnt', 'sasiCnt', 'backCnt']:
        if f'{style}_mean' in df.columns and f'{style}_std' in df.columns and f'{style}_max' in df.columns:
            df[f'{style}_skew_proxy'] = (df[f'{style}_max'] - df[f'{style}_mean']) / (df[f'{style}_std'] + 1e-6)

    return df


def add_categorical_combinations(df):
    """カテゴリカル特徴量の組み合わせ"""

    # 会場×カテゴリ
    if 'track' in df.columns and 'category' in df.columns:
        df['track_x_category'] = df['track'].astype(str) + '_' + df['category'].astype(str)

    # 会場×グレード
    if 'track' in df.columns and 'grade' in df.columns:
        df['track_x_grade'] = df['track'].astype(str) + '_' + df['grade'].astype(str)

    # カテゴリ×月（季節性）
    if 'category' in df.columns and 'month' in df.columns:
        df['category_x_month'] = df['category'].astype(str) + '_' + df['month'].astype(str)

    # 会場×月（会場の季節性）
    if 'track' in df.columns and 'month' in df.columns:
        df['track_x_month'] = df['track'].astype(str) + '_' + df['month'].astype(str)

    return df


def add_target_encoding(df, target_col='target_high_payout', n_folds=5):
    """
    ターゲットエンコーディング（リーケージ防止のためTimeSeriesSplit使用）

    会場別、カテゴリ別、グレード別の高配当率を特徴量化
    """

    # TimeSeriesSplitでfoldを作成
    tscv = TimeSeriesSplit(n_splits=n_folds)

    # エンコーディング対象のカラム
    encode_cols = []
    if 'track' in df.columns:
        encode_cols.append('track')
    if 'category' in df.columns:
        encode_cols.append('category')
    if 'grade' in df.columns:
        encode_cols.append('grade')
    if 'track_x_category' in df.columns:
        encode_cols.append('track_x_category')
    if 'track_x_month' in df.columns:
        encode_cols.append('track_x_month')

    # 各カラムに対してターゲットエンコーディング
    for col in encode_cols:
        encoded_col = f'{col}_target_enc'
        df[encoded_col] = np.nan

        for train_idx, val_idx in tscv.split(df):
            train_df = df.iloc[train_idx]

            # 訓練データで各カテゴリの平均目的変数を計算
            target_means = train_df.groupby(col)[target_col].mean()
            global_mean = train_df[target_col].mean()

            # 検証データにマッピング（未知カテゴリはグローバル平均）
            val_df = df.iloc[val_idx]
            df.loc[val_idx, encoded_col] = val_df[col].map(target_means).fillna(global_mean)

    return df


def main():
    print("=== Version 6 超高度特徴量の構築開始 ===\n")

    # V5データの読み込み
    print("1. V5訓練データ読み込み中...")
    df = pd.read_csv('data/training_dataset_enhanced_v5.csv')
    print(f"   {len(df):,}行, {len(df.columns)}列")

    # エントリデータの読み込み
    print("\n2. エントリデータ読み込み中...")
    entries = pd.read_csv('data/keirin_race_detail_entries_20240101_20241231_combined.csv')
    print(f"   エントリ数: {len(entries):,}行")

    # 車番・枠番特徴量
    print("\n3. 車番・枠番特徴量の作成中...")
    position_features = add_position_features(entries)
    print(f"   新特徴量数: {len(position_features.columns) - 4}")

    # 級班変化特徴量
    print("\n4. 級班変化特徴量の作成中...")
    kyuhan_features = add_kyuhan_change_features(entries)
    print(f"   新特徴量数: {len(kyuhan_features.columns) - 4}")

    # trackを数値に変換（マージ用）
    track_mapping = {}
    for idx, track_name in enumerate(sorted(position_features['track'].unique())):
        track_mapping[track_name] = idx
    position_features['track'] = position_features['track'].map(track_mapping)
    kyuhan_features['track'] = kyuhan_features['track'].map(track_mapping)

    # 結合
    print("\n5. 既存データと結合中...")
    df['race_no_str'] = df['race_no_int'].astype(str)
    position_features['race_no_str'] = position_features['race_no'].astype(str)
    kyuhan_features['race_no_str'] = kyuhan_features['race_no'].astype(str)

    df = df.merge(position_features, on=['race_date', 'track', 'race_no_str'], how='left')
    df = df.merge(kyuhan_features, on=['race_date', 'track', 'race_no_str'], how='left', suffixes=('', '_dup'))

    # 重複カラム削除
    dup_cols = [c for c in df.columns if c.endswith('_dup')]
    df = df.drop(columns=dup_cols + ['race_encp', 'race_no', 'race_no_str'], errors='ignore')

    print(f"   結合後: {len(df):,}行, {len(df.columns)}列")

    # 高度統計量
    print("\n6. 高度統計量の追加中...")
    df = add_advanced_statistics(df)
    print(f"   追加後: {len(df.columns)}列")

    # カテゴリカル組み合わせ
    print("\n7. カテゴリカル組み合わせの追加中...")
    df = add_categorical_combinations(df)
    print(f"   追加後: {len(df.columns)}列")

    # ターゲットエンコーディング
    print("\n8. ターゲットエンコーディング実施中...")
    df = add_target_encoding(df, target_col='target_high_payout', n_folds=5)
    print(f"   追加後: {len(df.columns)}列")

    # NaN処理
    print("\n9. 欠損値処理中...")
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        df[col] = df[col].fillna(0)

    # カテゴリカル組み合わせカラムのNaN処理
    categorical_comb_cols = [c for c in df.columns if '_x_' in c and df[c].dtype == 'object']
    for col in categorical_comb_cols:
        df[col] = df[col].fillna('unknown')

    nan_count = df.isna().sum().sum()
    print(f"   残存NaN数: {nan_count}")

    # 保存
    output_file = 'data/training_dataset_ultra_v6.csv'
    print(f"\n10. 保存中: {output_file}")
    df.to_csv(output_file, index=False)
    print(f"   ✅ 完了: {len(df):,}行, {len(df.columns)}列")

    # 新特徴量のサマリー
    print("\n=== 新特徴量サマリー（数値型のみ）===")
    new_cols = [
        'syaban_mean', 'syaban_std', 'inner_avg_score', 'outer_avg_score',
        'inner_outer_gap', 'top3_inner_ratio', 'upgraded_count', 'downgraded_count',
        'kyuhan_change_ratio', 'upgrade_ratio', 'score_skewness_proxy',
        'score_kurtosis_proxy', 'score_q25', 'score_q75', 'score_iqr'
    ]

    for col in new_cols:
        if col in df.columns and df[col].dtype in [np.float64, np.int64]:
            print(f"  {col}: mean={df[col].mean():.3f}, std={df[col].std():.3f}, non-zero={(df[col]!=0).sum()}")


if __name__ == '__main__':
    main()
