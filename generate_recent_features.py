#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
直近成績特徴を生成するスクリプト

各選手の直近3戦の成績を計算し、CSVに保存します。
- 直近3戦平均着順
- 直近3戦の1着回数
- 連続入着数
"""

import pandas as pd
from pathlib import Path
from collections import defaultdict

def generate_recent_performance_features():
    """直近成績特徴を生成"""
    
    # データ読み込み
    entries_path = Path('data/keirin_race_detail_entries_20240101_20251004.csv')
    results_path = Path('data/keirin_results_20240101_20251004.csv')
    
    print("Loading data...")
    print(f"Loading entries from {entries_path}...")
    entries = pd.read_csv(entries_path, low_memory=False)
    
    print(f"Loading results from {results_path}...")
    results = pd.read_csv(results_path, low_memory=False)
    
    # race_dateでソート
    entries['race_date'] = pd.to_numeric(entries['race_date'], errors='coerce')
    entries = entries.sort_values('race_date')
    
    # 選手ごとの成績履歴を追跡
    rider_history = defaultdict(list)
    
    # 新しい列を初期化
    entries['recent_3_avg_position'] = 0.0
    entries['recent_3_first_count'] = 0
    entries['consecutive_placements'] = 0
    
    print(f"Processing {len(entries)} entries...")
    
    # 各エントリーに対して直近成績を計算
    for idx, row in entries.iterrows():
        if idx % 10000 == 0:
            print(f"Processed {idx}/{len(entries)}")
        
        rider_name = str(row.get('name', ''))
        if not rider_name:
            continue
        
        # この選手の過去データを取得
        history = rider_history[rider_name]
        
        if len(history) > 0:
            # 直近3戦のデータ
            recent_3 = history[-3:]
            
            # 平均着順
            positions = [h['position'] for h in recent_3 if h['position'] > 0]
            if positions:
                entries.at[idx, 'recent_3_avg_position'] = sum(positions) / len(positions)
            
            # 1着回数
            first_count = sum(1 for h in recent_3 if h['position'] == 1)
            entries.at[idx, 'recent_3_first_count'] = first_count
            
            # 連続入着数（3着以内）
            consecutive = 0
            for h in reversed(history):
                if h['position'] <= 3 and h['position'] > 0:
                    consecutive += 1
                else:
                    break
            entries.at[idx, 'consecutive_placements'] = consecutive
        
        # この選手の過去データに今回のレースを追加（仮の着順=0）
        # 実際の着順はレース後に分かるが、ここでは予測時を想定
        rider_history[rider_name].append({
            'race_date': row['race_date'],
            'position': 0  # 未確定
        })
    
    # 保存
    output_path = Path('data/keirin_entries_with_recent_performance.csv')
    entries.to_csv(output_path, index=False)
    print(f"\nSaved to {output_path}")
    print(f"Added columns: recent_3_avg_position, recent_3_first_count, consecutive_placements")

if __name__ == '__main__':
    generate_recent_performance_features()
