#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Web検索で1,000レース分の選手詳細データを収集

手順：
1. 各レースについてWeb検索で出走表を探す
2. 選手名、府県、級班を収集
3. 競走得点は級班から推定
4. JSON形式で保存
"""

import pandas as pd
import json
import time
from pathlib import Path

# 収集対象レースリスト
df_races = pd.read_csv('first_20_races_for_collection.csv')

print("="*80)
print("【Web検索による選手詳細データ収集】")
print("="*80)
print(f"\n対象レース数: {len(df_races)}レース")
print(f"日付範囲: {df_races['race_date'].min()} 〜 {df_races['race_date'].max()}")

collected_races = []

# 既に収集済みのレース
with open('collected_races_web_search.json', 'r', encoding='utf-8') as f:
    existing_data = json.load(f)
    collected_races = existing_data['collected_races']

print(f"\n既に収集済み: {len(collected_races)}レース")
print(f"残り: {len(df_races) - len(collected_races)}レース")

# 2レース目から開始
START_INDEX = len(collected_races)

print(f"\n{START_INDEX}レース目から収集を開始します...")
print("="*80)

# レース情報の表示（Web検索用）
for i in range(START_INDEX, min(START_INDEX + 10, len(df_races))):
    race = df_races.iloc[i]
    print(f"\n【レース{i+1}】")
    print(f"  日付: {race['race_date']}")
    print(f"  会場: {race['track']}")
    print(f"  レース番号: {race['race_no']}")
    print(f"  グレード: {race['grade']}")
    print(f"  カテゴリ: {race['category']}")
    print(f"  三連単配当: ¥{race['trifecta_payout_num']:,.0f}")
    print(f"\n  検索キーワード例:")

    # 日付をフォーマット
    date_str = str(int(race['race_date']))
    year = date_str[:4]
    month = date_str[4:6]
    day = date_str[6:8]

    print(f"    「{year}年{month}月{day}日 {race['track']} {race['race_no']} 出走表」")
    print(f"    「{year}/{month}/{day} {race['track']} 競輪 {race['race_no']}」")
    print(f"    「競輪 {race['track']} {year}{month}{day} {race['race_no']} 選手」")

    print("\n  → Web検索を実行してください")
    print("  → 見つかった選手名、府県、級班を記録")
    print("-" * 80)

print("\n" + "="*80)
print("【収集作業の手順】")
print("="*80)
print("""
1. 上記の検索キーワードを使ってWeb検索を実行
2. 出走表ページを見つける
3. 各選手の以下の情報を収集:
   - 車番（1-9）
   - 選手名
   - 府県
   - 級班（SS, S1, S2, A1, A2, A3, L1）
   - 脚質（逃, 追, 差）※可能なら

4. 競走得点は級班から以下のように推定:
   - SS級: 115-120点
   - S1級: 110-115点
   - S2級: 105-110点
   - A1級: 95-105点
   - A2級: 85-95点
   - A3級: 75-85点
   - L1級: 45-55点

5. データを以下の形式で記録:
   race_X_riders = [
       {'car_no': 1, 'name': '選手名', 'prefecture': '府県', 'grade': 'S1', 'style': '追', 'avg_score': 112.0},
       ...
   ]

6. このスクリプトに追加して実行
""")

print("\n次のステップ:")
print("  このスクリプトを編集して、収集したデータを追加してください")
print("="*80)

# 出力フォーマットのサンプル
print("\n【記述例】")
print("""
# レース2のデータ（例）
race_2_riders = [
    {'car_no': 1, 'name': '山田太郎', 'prefecture': '東京', 'grade': 'A3', 'style': '追', 'avg_score': 82.0},
    {'car_no': 2, 'name': '鈴木一郎', 'prefecture': '大阪', 'grade': 'A2', 'style': '逃', 'avg_score': 88.0},
    # ... 9人分
]

race_2_data = {
    'race_date': 20251004,
    'track': '函館',
    'race_no': '2R',
    'grade': 'F2',
    'category': 'Ａ級チャレンジ予選',
    'trifecta_payout': 9910.0,
    'riders': race_2_riders
}

collected_races.append(race_2_data)
""")
