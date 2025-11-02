#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
簡易トップK出力ツール（事前データモデル）

利用例:
  python easy_predict.py
  python easy_predict.py --date 20241025
  python easy_predict.py --start-date 20241001 --end-date 20241031 --top-k 50
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from analysis import prerace_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="LightGBM 事前予測モデルから高配当候補 Top-K を出力します。",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--date", type=int, help="特定の日付 (YYYYMMDD)。指定するとその日だけ表示。")
    parser.add_argument("--start-date", type=int, help="開始日 (YYYYMMDD)")
    parser.add_argument("--end-date", type=int, help="終了日 (YYYYMMDD)")
    parser.add_argument("--min-score", type=float, help="このスコア以上のみ表示")
    parser.add_argument("--top-k", type=int, default=100, help="出力件数")
    parser.add_argument(
        "--output",
        type=Path,
        help="CSV に保存する場合のパス",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset = prerace_model.load_cached_dataset()
    metadata = prerace_model.load_metadata()
    model = prerace_model.load_model()

    feature_frame = dataset[metadata["feature_columns"]].astype(float)
    scores = model.predict(feature_frame)

    df = dataset[
        [
            "race_date",
            "keirin_cd",
            "race_no",
            "track",
            "grade",
            "target_high_payout",
        ]
    ].copy()
    df["prediction"] = scores

    if args.date:
        df = df[df["race_date"] == args.date]
    if args.start_date:
        df = df[df["race_date"] >= args.start_date]
    if args.end_date:
        df = df[df["race_date"] <= args.end_date]
    if args.min_score is not None:
        df = df[df["prediction"] >= args.min_score]

    df = df.sort_values("prediction", ascending=False).reset_index(drop=True)
    top_k = min(args.top_k, len(df))
    df = df.head(top_k)

    if df.empty:
        print("指定条件に一致するレースがありません。")
        return

    hit_rate = df["target_high_payout"].mean() if "target_high_payout" in df.columns else float("nan")

    print("\n" + "=" * 70)
    print("高配当候補 Top-{} (事前モデル)".format(top_k))
    print("=" * 70)
    print(
        f"対象レース数: {len(df):,} / 高配当率: "
        f"{hit_rate * 100:.1f}% (訓練期間内の確認値)"
    )
    print("-" * 70)
    print(f"{'順位':<4} {'日付':<10} {'場':<4} {'R':<2} {'グレード':<6} {'スコア':<8} {'実績':<4}")
    print("-" * 70)
    for idx, row in df.iterrows():
        print(
            f"{idx + 1:<4} "
            f"{str(int(row['race_date'])):<10} "
            f"{str(row['keirin_cd']).zfill(2):<4} "
            f"{int(row['race_no']):<2} "
            f"{str(row.get('grade', '')):<6} "
            f"{row['prediction']:.3f}   "
            f"{'◎' if row.get('target_high_payout', 0) else '-'}"
        )

    if args.output:
        df.to_csv(args.output, index=False)
        print(f"\nCSV に保存しました: {args.output}")


if __name__ == "__main__":
    main()

