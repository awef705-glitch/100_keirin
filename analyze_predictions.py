#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
モデル予測確率を分析し、最適な閾値を見つけるスクリプト
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

sys.path.append(str(Path.cwd()))

from analysis import prerace_model

def analyze_prediction_distribution():
    """予測確率の分布を分析"""
    
    # データセット読み込み
    print("Loading dataset...")
    dataset = prerace_model.load_cached_dataset()
    
    feature_cols = prerace_model._default_feature_columns()
    X = dataset[feature_cols]
    y_true = dataset['target_high_payout']
    
    # モデル読み込み
    model = prerace_model.load_model()
    
    if model is None:
        print("Model not loaded")
        return
    
    # 予測確率
    print("Predicting probabilities...")
    y_pred_proba = model.predict(X.values)
    
    # 確率分布の分析
    print("\n" + "="*80)
    print("予測確率の分布分析")
    print("="*80)
    
    # 確率binごとの分析
    bins = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    bin_labels = ['0-10%', '10-20%', '20-30%', '30-40%', '40-50%', '50-60%', '60-70%', '70-80%', '80-90%', '90-100%']
    
    dataset['pred_proba'] = y_pred_proba
    dataset['pred_bin'] = pd.cut(y_pred_proba, bins=bins, labels=bin_labels)
    
    print("\n確率帯ごとの実際の高配当率:")
    print("-"*80)
    print(f"{'確率帯':<12} {'レース数':>10} {'高配当数':>10} {'実際の率':>12}")
    print("-"*80)
    
    for bin_label in bin_labels:
        bin_data = dataset[dataset['pred_bin'] == bin_label]
        if len(bin_data) > 0:
            actual_rate = bin_data['target_high_payout'].mean()
            high_payout_count = bin_data['target_high_payout'].sum()
            print(f"{bin_label:<12} {len(bin_data):>10} {high_payout_count:>10.0f} {actual_rate:>11.1%}")
    
    # 最適閾値の推定
    print("\n" + "="*80)
    print("最適閾値の推定（精度重視）")
    print("="*80)
    
    thresholds = np.arange(0.1, 0.9, 0.05)
    best_accuracy = 0
    best_threshold = 0.5
    
    for threshold in thresholds:
        y_pred = (y_pred_proba >= threshold).astype(int)
        accuracy = (y_pred == y_true).mean()
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_threshold = threshold
    
    print(f"最適閾値: {best_threshold:.2f}")
    print(f"精度: {best_accuracy:.1%}")
    
    # 的中率重視の分析
    print("\n" + "="*80)
    print("買い目戦略の提案")
    print("="*80)
    
    # 各閾値での買い推奨レース数と的中率
    for threshold in [0.2, 0.25, 0.3, 0.35, 0.4]:
        target_races = dataset[y_pred_proba >= threshold]
        if len(target_races) > 0:
            hit_rate = target_races['target_high_payout'].mean()
            print(f"確率{threshold:.0%}以上で買い: {len(target_races)}レース ({len(target_races)/len(dataset)*100:.1f}%), 的中率{hit_rate:.1%}")

if __name__ == '__main__':
    analyze_prediction_distribution()
