#!/usr/bin/env python3
"""
選手ごとの過去成績を集計
レース前予測用の統計データを作成
"""
import json
import re
from pathlib import Path
from collections import defaultdict

import pandas as pd
import numpy as np


def parse_payout(value: str) -> float:
    """配当金文字列を数値に変換"""
    if pd.isna(value) or value == "":
        return np.nan
    digits = re.sub(r"[^0-9]", "", str(value))
    return float(digits) if digits else np.nan


def calculate_player_stats(csv_path: Path) -> dict:
    """選手ごとの過去成績を集計"""

    print("=" * 60)
    print("選手統計データの作成")
    print("=" * 60)

    # データ読み込み
    print("\n[1/3] データを読み込み中...")
    df = pd.read_csv(csv_path)
    print(f"  総レース数: {len(df):,}")

    # 配当金を数値化
    df["trifecta_payout_value"] = df["trifecta_payout"].apply(parse_payout)
    df = df.dropna(subset=["trifecta_payout_value"])

    # 選手統計を集計
    print("\n[2/3] 選手ごとの統計を集計中...")
    player_stats = defaultdict(lambda: {
        "races": 0,
        "wins": 0,
        "places_2": 0,
        "places_3": 0,
        "total_payout": 0,
        "high_payout_races": 0,
    })

    for _, row in df.iterrows():
        # 1着選手
        name1 = row.get("pos1_name", "")
        if pd.notna(name1) and name1:
            player_stats[name1]["races"] += 1
            player_stats[name1]["wins"] += 1
            player_stats[name1]["total_payout"] += row["trifecta_payout_value"]
            if row["trifecta_payout_value"] >= 10000:
                player_stats[name1]["high_payout_races"] += 1

        # 2着選手
        name2 = row.get("pos2_name", "")
        if pd.notna(name2) and name2:
            player_stats[name2]["races"] += 1
            player_stats[name2]["places_2"] += 1
            player_stats[name2]["total_payout"] += row["trifecta_payout_value"]
            if row["trifecta_payout_value"] >= 10000:
                player_stats[name2]["high_payout_races"] += 1

        # 3着選手
        name3 = row.get("pos3_name", "")
        if pd.notna(name3) and name3:
            player_stats[name3]["races"] += 1
            player_stats[name3]["places_3"] += 1
            player_stats[name3]["total_payout"] += row["trifecta_payout_value"]
            if row["trifecta_payout_value"] >= 10000:
                player_stats[name3]["high_payout_races"] += 1

    # 統計を計算
    print("\n[3/3] 勝率・連対率などを計算中...")
    player_final_stats = {}

    for name, stats in player_stats.items():
        if stats["races"] < 5:  # 最低5レース以上
            continue

        races = stats["races"]
        player_final_stats[name] = {
            "races": races,
            "win_rate": stats["wins"] / races,
            "place_2_rate": stats["places_2"] / races,
            "place_3_rate": stats["places_3"] / races,
            "top3_rate": (stats["wins"] + stats["places_2"] + stats["places_3"]) / races,
            "avg_payout": stats["total_payout"] / races,
            "high_payout_rate": stats["high_payout_races"] / races,
        }

    print(f"\n  集計完了: {len(player_final_stats):,} 選手")

    # Top 10選手を表示
    print("\n  勝率Top 10:")
    sorted_players = sorted(
        player_final_stats.items(),
        key=lambda x: x[1]["win_rate"],
        reverse=True
    )[:10]

    for i, (name, stats) in enumerate(sorted_players, 1):
        print(f"    {i:2d}. {name:15s} 勝率:{stats['win_rate']*100:5.1f}% "
              f"連対率:{(stats['win_rate'] + stats['place_2_rate'])*100:5.1f}% "
              f"({stats['races']}走)")

    return player_final_stats


def main():
    # データファイル
    csv_path = Path("data/keirin_results_20240101_20251004.csv")
    if not csv_path.exists():
        raise SystemExit(f"データファイルが見つかりません: {csv_path}")

    # 選手統計を計算
    player_stats = calculate_player_stats(csv_path)

    # 保存
    output_dir = Path("backend/models")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "player_stats.json"

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(player_stats, f, ensure_ascii=False, indent=2)

    print(f"\n✅ 選手統計を保存しました: {output_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
