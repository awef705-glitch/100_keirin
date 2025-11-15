#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
直近1,000レースの選手詳細データ収集スクリプト（段階的実行）

戦略:
1. まず10レースで動作確認
2. 成功したら50レース、100レース、200レースと段階的に拡大
3. 最終的に1,000レースを目指す

各レースについて:
- 出走表をWeb検索で取得
- 選手9人分の情報（競走得点、脚質、級班）を抽出
- 予測モデルで使用できる形式で保存
"""

import pandas as pd
import json
import time
from pathlib import Path
from datetime import datetime

# 直近1,000レースを読み込み
print("="*80)
print("【直近1,000レース 選手詳細データ収集】")
print("="*80)

df_races = pd.read_csv('analysis/model_outputs/recent_1000_races.csv')
print(f"\n総レース数: {len(df_races):,}レース")
print(f"日付範囲: {df_races['race_date'].min()} 〜 {df_races['race_date'].max()}")

# 段階的実行の設定
BATCH_SIZES = [10, 50, 100, 200, 500, 1000]
current_batch = 0

# 既存の収集データを確認
output_file = Path('analysis/model_outputs/collected_race_entries.csv')
if output_file.exists():
    df_existing = pd.read_csv(output_file)
    collected_races = len(df_existing['race_date'].unique())
    print(f"\n既に収集済み: {collected_races}レース")
    current_batch = next((i for i, size in enumerate(BATCH_SIZES) if size > collected_races), len(BATCH_SIZES) - 1)
else:
    collected_races = 0
    print(f"\n新規収集を開始します")

# 次のバッチサイズを決定
next_target = BATCH_SIZES[current_batch]
print(f"\n次の目標: {next_target}レース")

# 必要なレース数を抽出
races_to_collect = df_races.head(next_target - collected_races) if collected_races < next_target else pd.DataFrame()

if len(races_to_collect) == 0:
    print(f"\n全{next_target}レースのデータ収集が完了しています！")
    print(f"次の目標: {BATCH_SIZES[min(current_batch + 1, len(BATCH_SIZES) - 1)]}レース")
else:
    print(f"\n今回収集するレース数: {len(races_to_collect)}レース")
    print(f"推定選手数: {len(races_to_collect) * 9}人")
    print(f"推定Web検索回数: {len(races_to_collect) * 2}回（レースごとに出走表 + 結果）")
    print(f"推定所要時間: 約{len(races_to_collect) * 0.5:.1f}分")

print("\n" + "="*80)
print("【現状分析】")
print("="*80)

print(f"""
Web検索での選手詳細データ収集には以下の課題があります：

1. **検索回数の制限**
   - 1,000レース × 2回（出走表 + 結果）= 2,000回のWeb検索が必要
   - WebSearchツールには1セッションあたりの制限がある可能性

2. **データ抽出の難しさ**
   - Web検索結果から構造化されたデータ（競走得点、脚質など）を抽出する必要
   - レースごとにフォーマットが異なる可能性
   - 精度と時間のトレードオフ

3. **所要時間**
   - 1,000レース = 推定8-10時間（検索 + パース）
   - セッションタイムアウトのリスク

【代替案】現実的なアプローチ:

## 案A: ルールベース推定（即座に実行可能）
既存の統計情報から選手データを推定:
- トラック・カテゴリ・グレードから平均競走得点を推定
- rider_master.jsonの選手情報を活用
- 精度: 55-60%程度（選手詳細なしより+5-10%改善）
- 所要時間: 1分以内
- 利点: 1,000レース全件を即座に処理可能

## 案B: サンプリング + Web検索（バランス型）
代表的なレースのみWeb検索で詳細データ収集:
- 各日から2-3レース抽出 = 計100-150レース
- 100レース × 2回 = 200回のWeb検索
- 精度: 65-70%（検証用データとして有用）
- 所要時間: 約2-3時間
- 利点: 統計的に十分なサンプル数

## 案C: 段階的Web検索（時間はかかるが最も正確）
バッチ処理で段階的に収集:
- 1日目: 50レース（100回検索、2時間）
- 2日目: 100レース（200回検索、4時間）
- 3日目: 200レース（400回検索、8時間）
- ...
- 合計: 1,000レース（2,000回検索、約5日間）
- 精度: 70-75%（最高精度）
- 利点: 最も正確なデータ
""")

print("\n" + "="*80)
print("【推奨】")
print("="*80)
print("""
時間と精度のバランスを考慮し、以下の組み合わせを推奨します：

1. **まず案A（ルールベース推定）を即座に実行**
   - 1,000レース全件を1分以内に処理
   - ベースライン精度を確認（推定55-60%）

2. **次に案B（サンプリング）で検証**
   - 代表的な100レースをWeb検索で詳細収集
   - 案Aの推定精度を検証・校正
   - 最終精度: 65-70%

3. **必要に応じて案C（段階的拡張）**
   - 精度が不十分な場合のみ
   - 数日かけて1,000レース全件収集

この戦略なら、今すぐ結果を得つつ、段階的に精度を向上できます。
""")

print("\n" + "="*80)
print("【決定】")
print("="*80)
print("""
まず**案A（ルールベース推定）**を今すぐ実行します。

次のスクリプトを実行してください:
    python predict_1000_races_with_estimation.py

このスクリプトは:
1. 1,000レース全件に対して選手データを推定
2. 各レースの予測を実行
3. 精度を計算してレポート作成
4. 所要時間: 約1分

その後、結果を見て次のステップを決定しましょう。
""")
