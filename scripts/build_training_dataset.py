import argparse
import json
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd


def read_csv(path: Path, **kwargs) -> pd.DataFrame:
    df = pd.read_csv(path, **kwargs)
    df.columns = [col.strip() for col in df.columns]
    return df


def simplify_payout(series: pd.Series) -> pd.Series:
    cleaned = (
        series.fillna("")
        .astype(str)
        .str.replace("[^0-9.]", "", regex=True)
        .replace("", pd.NA)
    )
    return pd.to_numeric(cleaned, errors="coerce")


def add_result_targets(results: pd.DataFrame) -> pd.DataFrame:
    df = results.copy()
    df["trifecta_payout_value"] = simplify_payout(df["trifecta_payout"])
    df["high_payout_flag"] = (df["trifecta_payout_value"] >= 10000).astype("Int64")
    for idx in range(1, 4):
        name_col = f"pos{idx}_name"
        car_col = f"pos{idx}_car_no"
        df[name_col] = df[name_col].fillna("").astype(str)
        df[car_col] = pd.to_numeric(df[car_col], errors="coerce").astype("Int64")
    return df


def build_recent_features(entries_df: pd.DataFrame, results_df: pd.DataFrame) -> pd.DataFrame:
    finishes: List[pd.DataFrame] = []
    for idx, pos_col in enumerate(["pos1_name", "pos2_name", "pos3_name"], start=1):
        temp = results_df[["race_id", pos_col]].rename(columns={pos_col: "sensyuName"})
        temp["finish_pos"] = idx
        finishes.append(temp)
    finish_df = pd.concat(finishes, ignore_index=True)
    merged = entries_df.merge(finish_df, how="left", on=["race_id", "sensyuName"])
    merged["finish_pos"] = merged["finish_pos"].fillna(10)
    merged["race_date"] = pd.to_datetime(merged["race_date_x"], format="%Y%m%d", errors="coerce")
    merged = merged.sort_values(["sensyuRegistNo", "race_date"])
    merged["recent_finish_avg"] = merged.groupby("sensyuRegistNo")["finish_pos"].transform(
        lambda s: s.rolling(window=5, min_periods=1).mean()
    )
    merged["recent_finish_last"] = merged.groupby("sensyuRegistNo")["finish_pos"].shift(1)
    return merged


def add_race_level_features(df: pd.DataFrame) -> pd.DataFrame:
    grouped = df.groupby("race_id")
    df["race_entry_count"] = grouped["sensyuRegistNo"].transform("count")
    score_mean = grouped["feature_score"].transform("mean")
    score_std = grouped["feature_score"].transform("std").fillna(0)
    df["race_score_mean"] = score_mean
    df["race_score_std"] = score_std
    df["score_diff"] = df["feature_score"] - score_mean
    df["score_z"] = df["score_diff"] / score_std.replace(0, 1)
    df["score_rank"] = grouped["feature_score"].rank(ascending=False, method="first")
    df["score_percentile"] = df["score_rank"] / df["race_entry_count"].replace(0, np.nan)
    df["recent_finish_rank"] = grouped["recent_finish_avg"].rank(method="min")
    df["recent_finish_rel"] = df["recent_finish_avg"] - grouped["recent_finish_avg"].transform("mean")

    df["prefecture_count_in_race"] = (
        df.groupby(["race_id", "prefecture"])["prefecture"].transform("count")
    )
    df["style_count_in_race"] = df.groupby(["race_id", "kyakusitu"])["kyakusitu"].transform("count")
    df["same_prefecture_ratio"] = (
        df["prefecture_count_in_race"] / df["race_entry_count"].replace(0, np.nan)
    )
    df["same_style_ratio"] = (
        df["style_count_in_race"] / df["race_entry_count"].replace(0, np.nan)
    )

    df["is_escape_style"] = (df["kyakusitu"] == "逃").astype(int)
    df["is_chaser_style"] = (df["kyakusitu"] == "追").astype(int)
    df["syaban"] = pd.to_numeric(df["syaban"], errors="coerce")
    df["car_no_norm"] = df.groupby("race_id")["syaban"].transform(
        lambda s: (s - s.min()) / (s.max() - s.min() + 1e-6)
    )
    return df


def main():
    parser = argparse.ArgumentParser(description="Build merged training dataset for keirin model")
    parser.add_argument("--race", default="data/keirin_race_detail_20240101_20240331_race_full.csv")
    parser.add_argument("--entries", default="data/keirin_race_detail_20240101_20240331_entries_full.csv")
    parser.add_argument("--results", default="data/keirin_results_20240101_20240331_full.csv")
    parser.add_argument("--profiles", default="data/keirin_rider_profiles_20240101_20240331.csv")
    parser.add_argument("--output", default="data/keirin_training_dataset_20240101_20240331.csv")
    args = parser.parse_args()

    race_df = read_csv(Path(args.race))
    entries_df = read_csv(Path(args.entries), dtype={"sensyuRegistNo": str})
    results_df = read_csv(Path(args.results))
    profiles_df = read_csv(Path(args.profiles), dtype={"rider_id": str})

    results_df = add_result_targets(results_df)
    race_results = race_df.merge(
        results_df[
            ["race_id", "trifecta_payout_value", "high_payout_flag", "category", "meeting_icon"]
        ],
        on="race_id",
        how="left",
    )

    entries_df = entries_df.rename(columns={"huKen": "entry_prefecture"})
    entries_recent = build_recent_features(entries_df, results_df)
    merged = entries_recent.merge(race_results, on="race_id", how="left", suffixes=("", "_race"))

    profiles_df = profiles_df.rename(columns={"rider_id": "sensyuRegistNo"})
    merged = merged.merge(
        profiles_df[
            [
                "sensyuRegistNo",
                "prefecture",
                "region",
                "period",
                "current_grade",
                "style_profile",
                "current_official_score",
                "experience_years",
            ]
        ],
        on="sensyuRegistNo",
        how="left",
    )

    merged["heikinTokuten"] = pd.to_numeric(merged["heikinTokuten"], errors="coerce")
    merged["current_official_score"] = pd.to_numeric(merged["current_official_score"], errors="coerce")
    merged["sensyuRegistNo"] = merged["sensyuRegistNo"].fillna("000000")
    merged["kyakusitu"] = merged["kyakusitu"].fillna(merged["style_profile"])
    merged["prefecture"] = merged["prefecture"].fillna(merged["entry_prefecture"])
    merged["region"] = merged["region"].fillna("Unknown")
    merged["feature_score"] = merged[["heikinTokuten", "current_official_score"]].mean(axis=1)
    merged["recent_finish_last"] = merged["recent_finish_last"].fillna(merged["recent_finish_avg"])
    merged["experience_years"] = merged["experience_years"].fillna(0)

    merged = add_race_level_features(merged)

    output_path = Path(args.output)
    merged.to_csv(output_path, index=False, encoding="utf-8-sig")

    summary = {
        "rows": int(len(merged)),
        "columns": merged.columns.tolist(),
        "output_csv": str(output_path),
        "high_payout_defined": int(merged["high_payout_flag"].notna().sum()),
    }
    summary_path = output_path.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
