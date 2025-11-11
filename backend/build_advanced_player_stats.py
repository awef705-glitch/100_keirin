#!/usr/bin/env python3
"""
詳細な選手統計を作成（場所別、決まり手別など）
"""
import json
from pathlib import Path
from collections import defaultdict

import pandas as pd
import numpy as np


def build_advanced_player_stats(csv_path: str) -> dict:
    """詳細な選手統計を構築"""

    print("=" * 60)
    print("詳細な選手統計データの作成")
    print("=" * 60)

    # データ読み込み
    print("\n[1/5] データを読み込み中...")
    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    print(f"  総レース数: {len(df):,}")

    # 選手ごとの詳細統計
    print("\n[2/5] 選手ごとの詳細統計を集計中...")
    player_stats = defaultdict(lambda: {
        "races": 0,
        "wins": 0,
        "place_2": 0,
        "place_3": 0,
        "total_payout": 0,
        "high_payout": 0,
        "by_track": defaultdict(lambda: {"races": 0, "wins": 0}),
        "by_grade": defaultdict(lambda: {"races": 0, "wins": 0}),
        "by_category": defaultdict(lambda: {"races": 0, "wins": 0}),
        "by_decision": defaultdict(int),
        "recent_performance": [],  # 最近のパフォーマンス
    })

    # 各レースを処理
    for idx, row in df.iterrows():
        if idx % 10000 == 0 and idx > 0:
            print(f"  処理中: {idx:,}/{len(df):,} レース")

        track = row.get("track", "不明")
        grade = row.get("grade", "不明")
        category = row.get("category", "不明")
        trifecta_payout = row.get("trifecta_payout", "0円")

        # 配当を数値に変換
        try:
            payout = int(str(trifecta_payout).replace("円", "").replace(",", ""))
        except:
            payout = 0

        is_high_payout = payout >= 10000

        # 1着選手
        pos1_name = row.get("pos1_name")
        pos1_decision = row.get("pos1_decision", "不明")
        if pd.notna(pos1_name):
            player_stats[pos1_name]["races"] += 1
            player_stats[pos1_name]["wins"] += 1
            player_stats[pos1_name]["total_payout"] += payout
            if is_high_payout:
                player_stats[pos1_name]["high_payout"] += 1

            player_stats[pos1_name]["by_track"][track]["races"] += 1
            player_stats[pos1_name]["by_track"][track]["wins"] += 1
            player_stats[pos1_name]["by_grade"][grade]["races"] += 1
            player_stats[pos1_name]["by_grade"][grade]["wins"] += 1
            player_stats[pos1_name]["by_category"][category]["races"] += 1
            player_stats[pos1_name]["by_category"][category]["wins"] += 1
            player_stats[pos1_name]["by_decision"][pos1_decision] += 1
            player_stats[pos1_name]["recent_performance"].append({
                "position": 1,
                "payout": payout,
                "track": track,
                "grade": grade
            })

        # 2着選手
        pos2_name = row.get("pos2_name")
        if pd.notna(pos2_name):
            player_stats[pos2_name]["races"] += 1
            player_stats[pos2_name]["place_2"] += 1
            player_stats[pos2_name]["total_payout"] += payout
            if is_high_payout:
                player_stats[pos2_name]["high_payout"] += 1

            player_stats[pos2_name]["by_track"][track]["races"] += 1
            player_stats[pos2_name]["by_grade"][grade]["races"] += 1
            player_stats[pos2_name]["by_category"][category]["races"] += 1
            player_stats[pos2_name]["recent_performance"].append({
                "position": 2,
                "payout": payout,
                "track": track,
                "grade": grade
            })

        # 3着選手
        pos3_name = row.get("pos3_name")
        if pd.notna(pos3_name):
            player_stats[pos3_name]["races"] += 1
            player_stats[pos3_name]["place_3"] += 1
            player_stats[pos3_name]["total_payout"] += payout
            if is_high_payout:
                player_stats[pos3_name]["high_payout"] += 1

            player_stats[pos3_name]["by_track"][track]["races"] += 1
            player_stats[pos3_name]["by_grade"][grade]["races"] += 1
            player_stats[pos3_name]["by_category"][category]["races"] += 1
            player_stats[pos3_name]["recent_performance"].append({
                "position": 3,
                "payout": payout,
                "track": track,
                "grade": grade
            })

    print(f"  完了: {len(df):,} レース")

    # 統計を計算
    print("\n[3/5] 各種統計を計算中...")
    result = {}

    for player_name, stats in player_stats.items():
        races = stats["races"]
        if races == 0:
            continue

        wins = stats["wins"]
        place_2 = stats["place_2"]
        place_3 = stats["place_3"]
        top3 = wins + place_2 + place_3

        # 基本統計
        result[player_name] = {
            "races": races,
            "wins": wins,
            "place_2": place_2,
            "place_3": place_3,
            "win_rate": wins / races,
            "place_2_rate": place_2 / races,
            "place_3_rate": place_3 / races,
            "top3_rate": top3 / races,
            "avg_payout": stats["total_payout"] / races,
            "high_payout_rate": stats["high_payout"] / races,
        }

        # 場所別勝率
        track_stats = {}
        for track, track_data in stats["by_track"].items():
            if track_data["races"] >= 3:  # 最低3レース
                track_stats[track] = {
                    "races": track_data["races"],
                    "win_rate": track_data["wins"] / track_data["races"]
                }
        result[player_name]["by_track"] = track_stats

        # グレード別勝率
        grade_stats = {}
        for grade, grade_data in stats["by_grade"].items():
            if grade_data["races"] >= 3:
                grade_stats[grade] = {
                    "races": grade_data["races"],
                    "win_rate": grade_data["wins"] / grade_data["races"]
                }
        result[player_name]["by_grade"] = grade_stats

        # カテゴリー別勝率
        category_stats = {}
        for category, category_data in stats["by_category"].items():
            if category_data["races"] >= 3:
                category_stats[category] = {
                    "races": category_data["races"],
                    "win_rate": category_data["wins"] / category_data["races"]
                }
        result[player_name]["by_category"] = category_stats

        # 決まり手の傾向
        if stats["by_decision"]:
            total_decisions = sum(stats["by_decision"].values())
            decision_dist = {
                k: v / total_decisions
                for k, v in stats["by_decision"].items()
            }
            result[player_name]["decision_distribution"] = decision_dist

        # 最近のパフォーマンス（直近20レース）
        recent = stats["recent_performance"][-20:]
        if recent:
            result[player_name]["recent_win_rate"] = sum(1 for r in recent if r["position"] == 1) / len(recent)
            result[player_name]["recent_top3_rate"] = sum(1 for r in recent if r["position"] <= 3) / len(recent)
            result[player_name]["recent_avg_payout"] = np.mean([r["payout"] for r in recent])

    print(f"  集計完了: {len(result):,} 選手")

    # Top 10選手を表示
    print("\n[4/5] 勝率Top 10:")
    sorted_players = sorted(
        result.items(),
        key=lambda x: (x[1]["win_rate"], x[1]["races"]),
        reverse=True
    )
    for i, (name, stats) in enumerate(sorted_players[:10], 1):
        print(f"    {i:2d}. {name:20s} 勝率: {stats['win_rate']*100:5.1f}% "
              f"連対率: {(stats['win_rate'] + stats['place_2_rate'])*100:5.1f}% "
              f"({stats['races']}走)")

    return result


def main():
    csv_path = Path(__file__).parent.parent / "data" / "keirin_results_20240101_20251004.csv"
    output_path = Path(__file__).parent / "models" / "player_stats_advanced.json"

    # 統計を作成
    player_stats = build_advanced_player_stats(str(csv_path))

    # 保存
    print("\n[5/5] 保存中...")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(player_stats, f, ensure_ascii=False, indent=2)

    print(f"\n✅ 詳細選手統計を保存しました: {output_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
