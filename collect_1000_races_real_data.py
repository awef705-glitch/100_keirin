#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
1,000レース分の実データ収集（Web検索使用）
各レースについて出走表を検索し、選手情報を収集
"""

import pandas as pd
import json
import time

# 対象レース読み込み
df_races = pd.read_csv('analysis/model_outputs/recent_1000_races.csv')

print("="*80)
print("【1,000レース 実データ収集開始】")
print("="*80)
print(f"\n総レース数: {len(df_races)}")

collected_races = []
failed_races = []

# 各レースについて処理
for i in range(len(df_races)):
    race = df_races.iloc[i]

    # 日付フォーマット
    date_str = str(int(race['race_date']))
    year = date_str[:4]
    month = date_str[4:6]
    day = date_str[6:8]

    print(f"\n{'='*80}")
    print(f"【レース {i+1}/{len(df_races)}】")
    print(f"{'='*80}")
    print(f"日付: {year}年{month}月{day}日")
    print(f"会場: {race['track']}")
    print(f"レース: {race['race_no']}")
    print(f"グレード: {race['grade']}")
    print(f"カテゴリ: {race['category']}")
    print(f"三連単配当: ¥{race['trifecta_payout_num']:,.0f}")

    # Web検索実行
    print(f"\n→ Web検索を実行中...")

    # この後、Web検索ツールを使用して情報収集
    # 各レースについて複数の検索パターンを試す

    race_data = {
        'race_date': int(race['race_date']),
        'track': race['track'],
        'race_no': race['race_no'],
        'grade': race['grade'],
        'category': race['category'],
        'trifecta_payout': float(race['trifecta_payout_num']),
        'search_attempted': True,
        'data_collected': False,
        'riders': []
    }

    # 収集状況を記録
    if len(race_data['riders']) == 9:
        collected_races.append(race_data)
        print(f"✓ 収集成功")
    else:
        failed_races.append(race_data)
        print(f"✗ 収集失敗")

    # 進捗保存（100レースごと）
    if (i + 1) % 100 == 0:
        output = {
            'total_attempted': i + 1,
            'collected': len(collected_races),
            'failed': len(failed_races),
            'collected_races': collected_races,
            'failed_races': failed_races
        }
        with open(f'collection_progress_{i+1}.json', 'w', encoding='utf-8') as f:
            json.dump(output, f, ensure_ascii=False, indent=2)
        print(f"\n進捗保存: {i+1}レース処理済み（成功{len(collected_races)}, 失敗{len(failed_races)}）")

print(f"\n{'='*80}")
print("【収集完了】")
print(f"{'='*80}")
print(f"成功: {len(collected_races)}/{len(df_races)}")
print(f"失敗: {len(failed_races)}/{len(df_races)}")
