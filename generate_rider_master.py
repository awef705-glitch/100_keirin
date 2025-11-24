#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
選手マスターデータを最新のentriesデータから生成
- 選手名、府県、級班、脚質、平均得点、B数
- 戦術カウント（逃げ、捲り、差し、マーク）を追加
"""

import pandas as pd
import json
from pathlib import Path

def generate_rider_master():
    """選手マスターデータを生成"""
    
    entries_path = Path('data/keirin_race_detail_entries_20240101_20251004.csv')
    
    print("Loading entries data...")
    entries = pd.read_csv(entries_path, low_memory=False)
    
    # 選手名でグループ化（実際の列名を使用）
    # 列名: 'race_encp', 'race_date', 'track', 'race_no', 'grade', 'nigeCnt', 'makuriCnt', 'sasiCnt', 'markCnt', 'backCnt'
    # 選手情報の列を探す
    print("Columns:", entries.columns.tolist()[:20])
    
    # 選手名の列名を確認（例: 'rider_name', 'name', など）
    # CSVに選手名がないので、entriesから生成できません
    # 代わりに、既存のrider_master.jsonがあればそれを使います
    
    print("\nError: Cannot generate rider master from this CSV structure.")
    print("This CSV does not contain rider-level information.")
    print("\nPlease use the existing rider_master.json file.")
    return
        if pd.isna(name) or str(name).strip() == '':
            continue
        
        # 最新のレコードを取得
        latest = group.iloc[-1]
        
        # 戦術カウントの平均値
        nige_count = group['nigeCnt'].fillna(0).mean() if 'nigeCnt' in group.columns else 0
        makuri_count = group['makuriCnt'].fillna(0).mean() if 'makuriCnt' in group.columns else 0
        sasi_count = group['sasiCnt'].fillna(0).mean() if 'sasiCnt' in group.columns else 0
        mark_count = group['markCnt'].fillna(0).mean() if 'markCnt' in group.columns else 0
        
        rider_data = {
            'name': str(name).strip(),
            'prefecture': str(latest.get('huKen', '')).strip(),
            'grade': str(latest.get('kyuhan', '')).strip(),
            'style': str(latest.get('kyakusitu', '')).strip(),
            'avg_score': float(latest.get('heikinTokuten', 0) or 0),
            'back_count': int(latest.get('backCnt', 0) or 0),
            'nige_count': int(nige_count),
            'makuri_count': int(makuri_count),
            'sasi_count': int(sasi_count),
            'mark_count': int(mark_count)
        }
        
        riders.append(rider_data)
    
    # 名前でソート
    riders.sort(key=lambda x: x['name'])
    
    print(f"\nGenerated {len(riders)} riders")
    
    # 保存
    output_path = Path('analysis/model_outputs/rider_master.json')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(riders, f, ensure_ascii=False, indent=2)
    
    print(f"Saved to {output_path}")
    
    # サンプル表示
    print("\nSample riders (top 5):")
    for rider in riders[:5]:
        print(f"  {rider['name']} ({rider['prefecture']}/{rider['grade']}) - "
              f"Score:{rider['avg_score']:.1f}, "
              f"Nige:{rider['nige_count']}, Makuri:{rider['makuri_count']}")

if __name__ == '__main__':
    generate_rider_master()
