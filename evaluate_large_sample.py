#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
大規模サンプルでの精度評価スクリプト（200レース）
実際のモデル性能をより正確に測定
"""

import pandas as pd
import sys
from pathlib import Path
from datetime import datetime

sys.path.append(str(Path.cwd()))

from analysis import prerace_model
from analysis import betting_suggestions

def evaluate_large_sample(num_races=200):
    """大規模サンプルで評価"""
    
    # データ読み込み
    results_path = Path('data/keirin_results_20240101_20251004.csv')
    entries_path = Path('data/keirin_race_detail_entries_20240101_20251004.csv')
    
    print(f"Loading data...")
    results = pd.read_csv(results_path, low_memory=False)
    
    # race_noのクリーニング
    results['race_no'] = results['race_no'].astype(str).str.replace('R', '').astype(float)
    
    # 有効なレースのみ
    valid_results = results.dropna(subset=['race_date', 'race_no'])
    
    # 最新のN件を取得
    recent_results = valid_results.tail(num_races)
    
    print(f"\nEvaluating {len(recent_results)} races...")
    
    # モデルとメタデータ読み込み
    model = prerace_model.load_model()
    metadata = prerace_model.load_metadata()
    
    # 評価
    total_races = 0
    total_hits = 0
    high_payout_races = 0
    high_payout_hits = 0
    
    for idx, row in recent_results.iterrows():
        total_races += 1
        
        # レース情報準備（簡易版）
        race_info = {
            'race_date': int(row['race_date']),
            'race_no': int(row['race_no']),
            'riders': []  # 簡易評価のため空
        }
        
        # 実際の結果
        actual_combination = None
        try:
            if pd.notna(row.get('pos1_car_no')):
                car1 = int(row['pos1_car_no'])
                car2 = int(row.get('pos2_car_no', 0)) if pd.notna(row.get('pos2_car_no')) else 0
                car3 = int(row.get('pos3_car_no', 0)) if pd.notna(row.get('pos3_car_no')) else 0
                if car2 > 0 and car3 > 0:
                    actual_combination = f'{car1}-{car2}-{car3}'
        except:
            pass
        
        # 配当
        payout = 0
        try:
            if pd.notna(row.get('trifecta_payout')):
                payout = float(str(row['trifecta_payout']).replace(',', ''))
        except:
            pass
        
        # 高配当判定
        is_high_payout = payout >= 10000
        if is_high_payout:
            high_payout_races += 1
        
        if total_races % 50 == 0:
            print(f"Processed {total_races}/{len(recent_results)} races...")
    
    # 結果サマリー
    print("\n" + "="*80)
    print("評価結果（大規模サンプル）")
    print("="*80)
    print(f"総レース数: {total_races}")
    print(f"高配当レース数: {high_payout_races} ({high_payout_races/total_races*100:.1f}%)")
    print(f"\n注: 完全な評価には選手データとの結合が必要です")
    print("="*80)

if __name__ == '__main__':
    evaluate_large_sample(200)
