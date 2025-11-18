import argparse
import json
from pathlib import Path
from typing import List

import pandas as pd


def read_csvs(paths: List[Path]) -> pd.DataFrame:
    frames = []
    for path in paths:
        df = pd.read_csv(path)
        df["source_file"] = path.name
        df["source_mtime"] = path.stat().st_mtime
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def normalize_race_no(series: pd.Series) -> pd.Series:
    race_no = series.astype(str).str.replace(".0", "", regex=False)
    return race_no.str.zfill(2)


def ensure_date_str(series: pd.Series) -> pd.Series:
    values = series.astype(str).str.extract(r"(\d{8})")[0]
    return values.fillna(series.astype(str))


def main():
    parser = argparse.ArgumentParser(description="Combine SJ0315 race detail CSVs into unified datasets")
    parser.add_argument("--data-dir", default="data", help="Directory containing race detail CSVs")
    parser.add_argument("--prerace", default="data/keirin_prerace_20240101_20240331.csv", help="Pre-race CSV with keirin_cd")
    parser.add_argument("--race-pattern", default="keirin_race_detail_race_*.csv", help="Glob for race CSVs")
    parser.add_argument("--entry-pattern", default="keirin_race_detail_entries_*.csv", help="Glob for entry CSVs")
    parser.add_argument("--output-prefix", default="keirin_race_detail_20240101_20240331", help="Prefix for output files")
    parser.add_argument("--write-parquet", action="store_true", help="Also write Parquet outputs")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    race_paths = sorted(data_dir.glob(args.race_pattern))
    entry_paths = sorted(data_dir.glob(args.entry_pattern))
    if not race_paths or not entry_paths:
        raise SystemExit("Race or entry CSVs not found in data directory")

    race_df = read_csvs(race_paths)
    entry_df = read_csvs(entry_paths)

    if "race_encp" not in race_df.columns or "race_encp" not in entry_df.columns:
        raise SystemExit("race_encp column missing in inputs")

    race_df = race_df[race_df["race_encp"].notna()]
    entry_df = entry_df[entry_df["race_encp"].notna()]

    race_df = race_df.sort_values(["race_encp", "source_mtime", "source_file"]).drop_duplicates(subset=["race_encp"], keep="last")
    entry_df = entry_df.sort_values(["race_encp", "syaban", "source_mtime", "source_file"]).drop_duplicates(subset=["race_encp", "syaban"], keep="last")

    prerace_cols = ["race_encp", "keirin_cd"]
    prerace_df = pd.read_csv(args.prerace, usecols=[col for col in prerace_cols if col])
    prerace_df = prerace_df.drop_duplicates(subset=["race_encp"])
    race_df = race_df.merge(prerace_df, on="race_encp", how="left")

    race_df["race_date"] = ensure_date_str(race_df["race_date"]).astype(str)
    race_df["race_no_norm"] = normalize_race_no(race_df["race_no"])
    race_df["keirin_cd"] = race_df["keirin_cd"].fillna("UNK").astype(str)
    race_df["race_id"] = race_df["keirin_cd"] + "_" + race_df["race_date"] + "_" + race_df["race_no_norm"]
    race_df.drop(columns=["race_no_norm"], inplace=True)

    entry_df = entry_df.merge(race_df[["race_encp", "race_id", "race_date", "track", "grade"]], on="race_encp", how="left")

    race_df = race_df.drop(columns=["source_mtime"])
    entry_df = entry_df.drop(columns=["source_mtime"])

    output_prefix = data_dir / args.output_prefix
    race_out_csv = f"{output_prefix}_race_full.csv"
    entry_out_csv = f"{output_prefix}_entries_full.csv"

    race_df.to_csv(race_out_csv, index=False, encoding="utf-8-sig")
    entry_df.to_csv(entry_out_csv, index=False, encoding="utf-8-sig")

    outputs = {
        "race_csv": race_out_csv,
        "entry_csv": entry_out_csv,
        "race_rows": int(len(race_df)),
        "entry_rows": int(len(entry_df)),
        "race_sources": [p.name for p in race_paths],
        "entry_sources": [p.name for p in entry_paths],
    }

    if args.write_parquet:
        race_out_parquet = f"{output_prefix}_race_full.parquet"
        entry_out_parquet = f"{output_prefix}_entries_full.parquet"
        race_df.to_parquet(race_out_parquet, index=False)
        entry_df.to_parquet(entry_out_parquet, index=False)
        outputs["race_parquet"] = race_out_parquet
        outputs["entry_parquet"] = entry_out_parquet

    summary_path = f"{output_prefix}_summary.json"
    Path(summary_path).write_text(json.dumps(outputs, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(outputs, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
