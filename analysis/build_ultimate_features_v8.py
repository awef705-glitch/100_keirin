#!/usr/bin/env python3
"""
Version 8: 究極の特徴量セット - 未使用データを完全活用

新規追加:
1. 府県・地域特徴量 (45都道府県)
2. 欠場補充特徴量 (追加選手パターン)
3. 色情報特徴量
4. 重要特徴量の相互作用項
5. 地域クラスタリング特徴量
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder


def add_prefecture_features(entries_df):
    """
    府県・地域特徴量の作成

    Args:
        entries_df: レースx選手のエントリデータ

    Returns:
        dict: レース単位の府県特徴量
    """
    features = {}

    # 府県の種類数
    features['unique_prefectures'] = entries_df['huKen'].nunique()

    # 最多府県の選手数
    prefecture_counts = entries_df['huKen'].value_counts()
    features['max_prefecture_count'] = prefecture_counts.iloc[0] if len(prefecture_counts) > 0 else 0

    # 府県の多様性 (エントロピー)
    prefecture_probs = prefecture_counts / prefecture_counts.sum()
    features['prefecture_entropy'] = -np.sum(prefecture_probs * np.log(prefecture_probs + 1e-10))

    # 地域ブロック分類
    kanto = ['東　京', '神奈川', '埼　玉', '千　葉', '茨　城', '栃　木', '群　馬']
    kansai = ['大　阪', '兵　庫', '京　都', '奈　良', '滋　賀', '和歌山']
    chubu = ['愛　知', '静　岡', '岐　阜', '三　重', '長　野', '新　潟', '山　梨', '富　山', '石　川', '福　井']
    kyushu = ['福　岡', '熊　本', '大　分', '佐　賀', '長　崎', '宮　崎', '鹿児島', '沖　縄']
    tohoku = ['福　島', '宮　城', '岩　手', '青　森', '秋　田', '山　形']

    features['kanto_count'] = entries_df['huKen'].isin(kanto).sum()
    features['kansai_count'] = entries_df['huKen'].isin(kansai).sum()
    features['chubu_count'] = entries_df['huKen'].isin(chubu).sum()
    features['kyushu_count'] = entries_df['huKen'].isin(kyushu).sum()
    features['tohoku_count'] = entries_df['huKen'].isin(tohoku).sum()

    # 同一府県選手の有無
    features['has_same_prefecture'] = 1 if features['max_prefecture_count'] > 1 else 0

    # 府県エンコーディング (上位5府県)
    top_prefectures = ['福　岡', '神奈川', '岡　山', '静　岡', '埼　玉']
    for pref in top_prefectures:
        features[f'has_{pref.replace("　", "")}'] = 1 if pref in entries_df['huKen'].values else 0

    return features


def add_replacement_features(entries_df, prerace_row):
    """
    欠場補充特徴量の作成

    Args:
        entries_df: レースx選手のエントリデータ
        prerace_row: preraceデータの該当行

    Returns:
        dict: 欠場補充特徴量
    """
    features = {}

    # 追加選手数
    replacement_count = entries_df['ketujyouTuikaHojyu'].notna().sum()
    features['replacement_count'] = replacement_count
    features['has_replacement'] = 1 if replacement_count > 0 else 0
    features['replacement_ratio'] = replacement_count / len(entries_df)

    # assenデータから追加を確認
    assen_replacement_count = 0
    entry_count = int(prerace_row.get('entry_count', 7))
    for i in range(1, min(entry_count + 1, 10)):
        assen_val = prerace_row.get(f'entry{i}_assen', '')
        if pd.notna(assen_val) and '追加' in str(assen_val):
            assen_replacement_count += 1

    features['assen_replacement_count'] = assen_replacement_count

    # 追加選手の枠番パターン
    if replacement_count > 0:
        replacement_syabans = entries_df[entries_df['ketujyouTuikaHojyu'].notna()]['syaban']
        features['replacement_syaban_mean'] = pd.to_numeric(replacement_syabans, errors='coerce').mean()
        features['replacement_syaban_std'] = pd.to_numeric(replacement_syabans, errors='coerce').std()
    else:
        features['replacement_syaban_mean'] = 0
        features['replacement_syaban_std'] = 0

    return features


def add_color_features(entries_df):
    """
    色情報特徴量の作成

    Args:
        entries_df: レースx選手のエントリデータ

    Returns:
        dict: 色情報特徴量
    """
    features = {}

    # 車番BG色の種類数
    features['unique_bg_colors'] = entries_df['syabanBgColorInfo'].nunique()

    # 車番Char色の種類数
    features['unique_char_colors'] = entries_df['syabanCharColorInfo'].nunique()

    # 最多BG色の出現数
    if entries_df['syabanBgColorInfo'].notna().sum() > 0:
        bg_counts = entries_df['syabanBgColorInfo'].value_counts()
        features['max_bg_color_count'] = bg_counts.iloc[0]
    else:
        features['max_bg_color_count'] = 0

    # 最多Char色の出現数
    if entries_df['syabanCharColorInfo'].notna().sum() > 0:
        char_counts = entries_df['syabanCharColorInfo'].value_counts()
        features['max_char_color_count'] = char_counts.iloc[0]
    else:
        features['max_char_color_count'] = 0

    return features


def add_interaction_features(base_features):
    """
    重要特徴量の相互作用項を作成

    Args:
        base_features: ベース特徴量のdict

    Returns:
        dict: 相互作用特徴量
    """
    interactions = {}

    # V5の重要特徴量同士の相互作用
    important_pairs = [
        ('trifecta_popularity', 'heikinTokuten_cv'),
        ('trifecta_popularity', 'nigeCnt_cv'),
        ('heikinTokuten_cv', 'nigeCnt_cv'),
        ('heikinTokuten_mean', 'heikinTokuten_std'),
        ('makuriCnt_mean', 'nigeCnt_mean'),
    ]

    for feat1, feat2 in important_pairs:
        if feat1 in base_features and feat2 in base_features:
            val1 = base_features[feat1]
            val2 = base_features[feat2]
            if pd.notna(val1) and pd.notna(val2):
                interactions[f'{feat1}_x_{feat2}'] = val1 * val2

    return interactions


def build_v8_features(entries, prerace, results):
    """
    V8特徴量セットを構築

    V5ベース + 府県 + 欠場補充 + 色情報 + 相互作用
    """
    print("=== Version 8: 究極特徴量セット構築 ===\n")

    # V5特徴量をロード (ベースとして使用)
    print("1. V5ベース特徴量読み込み中...")
    try:
        v5_data = pd.read_csv('data/clean_training_dataset.csv')
        print(f"   V5データ: {len(v5_data):,}行, {len(v5_data.columns)}列")
    except FileNotFoundError:
        print("   ⚠️  V5データが見つかりません。entriesから再構築します。")
        v5_data = None

    # Entriesをrace単位に集約
    print("\n2. 新規特徴量の作成中...")

    race_features = []

    grouped = entries.groupby(['race_date', 'track', 'race_no'])
    total = len(grouped)

    for idx, ((race_date, track, race_no), group_df) in enumerate(grouped):
        if (idx + 1) % 1000 == 0:
            print(f"   進捗: {idx+1}/{total}")

        features = {
            'race_date': race_date,
            'track': track,
            'race_no': race_no,
        }

        # 府県特徴量
        prefecture_feats = add_prefecture_features(group_df)
        features.update(prefecture_feats)

        # prerace データ取得
        prerace_match = prerace[
            (prerace['race_date'] == race_date) &
            (prerace['track'] == track) &
            (prerace['race_no'] == race_no)
        ]

        if len(prerace_match) > 0:
            prerace_row = prerace_match.iloc[0]

            # 欠場補充特徴量
            replacement_feats = add_replacement_features(group_df, prerace_row)
            features.update(replacement_feats)
        else:
            # デフォルト値
            features.update({
                'replacement_count': 0,
                'has_replacement': 0,
                'replacement_ratio': 0,
                'assen_replacement_count': 0,
                'replacement_syaban_mean': 0,
                'replacement_syaban_std': 0,
            })

        # 色情報特徴量
        color_feats = add_color_features(group_df)
        features.update(color_feats)

        race_features.append(features)

    v8_new_features = pd.DataFrame(race_features)
    print(f"\n   新規特徴量: {len(v8_new_features.columns) - 3}個作成")

    # V5データとマージ
    print("\n3. V5データとマージ中...")
    if v5_data is not None:
        # race_no と track を文字列化してマージ
        v5_data['race_no_str'] = v5_data['race_no_int'].astype(str)
        v8_new_features['race_no_str'] = v8_new_features['race_no'].astype(str)
        v5_data['track_str'] = v5_data['track'].astype(str)
        v8_new_features['track_str'] = v8_new_features['track'].astype(str)

        merged = v5_data.merge(
            v8_new_features,
            left_on=['race_date', 'track_str', 'race_no_str'],
            right_on=['race_date', 'track_str', 'race_no_str'],
            how='inner',
            suffixes=('', '_v8')
        )

        # 重複列を削除
        merged = merged.loc[:, ~merged.columns.duplicated()]

        print(f"   マージ後: {len(merged):,}行, {len(merged.columns)}列")
    else:
        merged = v8_new_features

    # 相互作用特徴量を追加
    print("\n4. 相互作用特徴量の作成中...")
    interaction_features = []
    for idx, row in merged.iterrows():
        interactions = add_interaction_features(row.to_dict())
        interaction_features.append(interactions)

    interaction_df = pd.DataFrame(interaction_features)
    merged = pd.concat([merged.reset_index(drop=True), interaction_df], axis=1)

    print(f"   相互作用特徴量: {len(interaction_df.columns)}個追加")

    # 保存
    print("\n5. 保存中...")
    output_file = 'data/training_dataset_v8_ultimate.csv'
    merged.to_csv(output_file, index=False)
    print(f"   保存完了: {output_file}")
    print(f"   最終: {len(merged):,}行, {len(merged.columns)}列")

    return merged


def main():
    # データ読み込み
    print("データ読み込み中...")
    entries = pd.read_csv('data/keirin_race_detail_entries_20240101_20241231_combined.csv')
    prerace = pd.read_csv('data/keirin_prerace_20240101_20241231_combined.csv')
    results = pd.read_csv('data/keirin_results_20240101_20251004.csv')

    print(f"  Entries: {len(entries):,}行")
    print(f"  Prerace: {len(prerace):,}行")
    print(f"  Results: {len(results):,}行")
    print()

    # V8特徴量構築
    v8_data = build_v8_features(entries, prerace, results)

    print("\n✅ Version 8 究極特徴量セット構築完了！")

    # 特徴量リスト出力
    print("\n=== 新規追加特徴量 ===")
    new_features = [col for col in v8_data.columns if col not in ['race_date', 'track', 'race_no', 'target_high_payout', 'keirin_cd', 'grade', 'category']]
    print(f"総特徴量数: {len(new_features)}")

    v8_only = [col for col in new_features if 'prefecture' in col or 'replacement' in col or 'color' in col or '_x_' in col]
    print(f"\nV8新規特徴量 ({len(v8_only)}個):")
    for feat in sorted(v8_only)[:30]:  # 上位30個表示
        print(f"  - {feat}")
    if len(v8_only) > 30:
        print(f"  ... 他{len(v8_only) - 30}個")


if __name__ == '__main__':
    main()
