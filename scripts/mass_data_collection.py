#!/usr/bin/env python3
"""
大量データ収集 - 2023-2024年の全G1レースを収集してモデル強化
"""

import sys
import json
from pathlib import Path

# 2024年 G1レース一覧 (既知情報から)
G1_RACES_2024 = [
    {
        'name': 'KEIRINグランプリ2024',
        'date': '20241230',
        'venue': '静岡',
        'winner': '古性優作',
        'status': 'collected',  # 既に収集済み
    },
    {
        'name': '競輪祭2024',
        'date': '20241124',
        'venue': '小倉',
        'winner': '脇本雄太',
        'status': 'collected',
    },
    {
        'name': '寬仁親王牌2024',
        'date': '20241020',
        'venue': '弥彦',
        'winner': '古性優作',
        'status': 'partial',  # 結果のみ、詳細データ必要
        'search_queries': [
            '寬仁親王牌 2024 決勝 出走表 競走得点',
            '"古性優作" "小原太樹" "河端朋之" 競走得点 2024年10月',
            '弥彦競輪 2024年10月20日 決勝 三連単 配当'
        ]
    },
    {
        'name': 'オールスター競輪2024',
        'date': '20240818',
        'venue': '平塚',
        'winner': '古性優作',
        'status': 'collected',
    },
    {
        'name': '高松宮記念杯2024',
        'date': '20240616',
        'venue': '岸和田',
        'winner': '北井佑季',
        'status': 'collected',
    },
    {
        'name': '日本選手権競輪2024',
        'date': '20240505',
        'venue': 'いわき平',
        'winner': '平原康多',
        'status': 'need_data',
        'search_queries': [
            '日本選手権競輪 2024 決勝 出走表 平原康多',
            '"平原康多" "吉田拓矢" 競走得点 2024年5月',
            'いわき平競輪 2024年5月5日 決勝 三連単 配当'
        ]
    },
    {
        'name': '全日本選抜競輪2024',
        'date': '20240212',
        'venue': '岐阜',
        'winner': '郡司浩平',
        'status': 'need_data',
        'search_queries': [
            '全日本選抜競輪 2024 決勝 出走表 郡司浩平',
            '"郡司浩平" "北井佑季" 競走得点 2024年2月',
            '岐阜競輪 2024年2月12日 決勝 三連単 配当'
        ]
    },
]

# 2023年 G1レース一覧
G1_RACES_2023 = [
    {
        'name': 'KEIRINグランプリ2023',
        'date': '20231230',
        'venue': '立川',
        'winner': '松浦悠士',
        'status': 'need_data',
        'search_queries': [
            'KEIRINグランプリ 2023 決勝 出走表 松浦悠士',
            '"松浦悠士" 競走得点 2023年12月',
            '立川競輪 2023年12月30日 グランプリ 三連単 配当'
        ]
    },
    {
        'name': '競輪祭2023',
        'date': '20231119',
        'venue': '小倉',
        'winner': '松浦悠士',
        'status': 'need_data',
    },
    {
        'name': '寬仁親王牌2023',
        'date': '20231022',
        'venue': '弥彦',
        'winner': '古性優作',
        'status': 'need_data',
    },
]

def print_collection_plan():
    """収集計画を表示"""
    print("="*70)
    print("大量データ収集計画")
    print("="*70)
    print()

    print("【2024年 G1レース】")
    collected = [r for r in G1_RACES_2024 if r['status'] == 'collected']
    partial = [r for r in G1_RACES_2024 if r['status'] == 'partial']
    need = [r for r in G1_RACES_2024 if r['status'] == 'need_data']

    print(f"✓ 収集済み: {len(collected)}レース")
    for r in collected:
        print(f"  - {r['name']} ({r['date']}, {r['venue']})")

    print(f"△ 部分的: {len(partial)}レース")
    for r in partial:
        print(f"  - {r['name']} ({r['date']}, {r['venue']})")

    print(f"✗ 未収集: {len(need)}レース")
    for r in need:
        print(f"  - {r['name']} ({r['date']}, {r['venue']})")

    print()
    print("【2023年 G1レース】")
    print(f"✗ 未収集: {len(G1_RACES_2023)}レース")
    for r in G1_RACES_2023:
        print(f"  - {r['name']} ({r['date']}, {r['venue']})")

    print()
    print("="*70)
    print(f"合計目標: {len(G1_RACES_2024) + len(G1_RACES_2023)}レース")
    print(f"現在: {len(collected)}レース収集済み")
    print(f"追加収集必要: {len(partial) + len(need) + len(G1_RACES_2023)}レース")
    print("="*70)
    print()

def generate_search_queries():
    """未収集レースの検索クエリを生成"""
    print("検索クエリ生成")
    print("="*70)
    print()

    all_races = G1_RACES_2024 + G1_RACES_2023
    need_collection = [r for r in all_races if r['status'] in ['need_data', 'partial']]

    for race in need_collection:
        print(f"\n【{race['name']}】")
        if 'search_queries' in race:
            for i, query in enumerate(race['search_queries'], 1):
                print(f"  {i}. {query}")
        else:
            # 自動生成
            print(f"  1. {race['name']} 決勝 出走表 競走得点")
            print(f"  2. \"{race['winner']}\" 競走得点 {race['date'][:6]}")
            print(f"  3. {race['venue']}競輪 {race['date'][:8]} 決勝 三連単 配当")

def save_collection_plan():
    """収集計画をJSONで保存"""
    plan = {
        '2024_G1': G1_RACES_2024,
        '2023_G1': G1_RACES_2023,
        'summary': {
            'total_target': len(G1_RACES_2024) + len(G1_RACES_2023),
            'collected': len([r for r in G1_RACES_2024 if r['status'] == 'collected']),
            'need_collection': len([r for r in G1_RACES_2024 + G1_RACES_2023
                                   if r['status'] in ['need_data', 'partial']])
        }
    }

    output_file = Path('data/collection_plan.json')
    output_file.parent.mkdir(exist_ok=True)

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(plan, f, ensure_ascii=False, indent=2)

    print(f"\n収集計画を保存: {output_file}")

if __name__ == '__main__':
    print_collection_plan()
    generate_search_queries()
    save_collection_plan()

    print()
    print("="*70)
    print("次のステップ:")
    print("1. 上記の検索クエリでWeb検索")
    print("2. 選手の競走得点・配当金額を収集")
    print("3. data/web_search_races.csvに追加")
    print("4. モデルを再訓練")
    print("="*70)
