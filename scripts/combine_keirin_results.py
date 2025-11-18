import argparse
import json
from pathlib import Path
from typing import List

import pandas as pd


def read_result_csvs(paths: List[Path]) -> pd.DataFrame:
    frames = []
    for path in paths:
        df = pd.read_csv(path, dtype={"race_date": str}, encoding_errors="ignore")
        df["source_file"] = path.name
        df["source_mtime"] = path.stat().st_mtime
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def normalize_race_no(series: pd.Series) -> pd.Series:
    digits = series.astype(str).str.extract(r"(\d+)")[0]
    return digits.fillna("").replace("", pd.NA).astype("Int64")


def main():
    parser = argparse.ArgumentParser(description="Combine keirin result CSV files into a single dataset")
    parser.add_argument("--data-dir", default="data", help="Directory containing keirin_results_*.csv files")
    parser.add_argument("--pattern", default="keirin_results_2024*.csv", help="Glob pattern for input CSVs")
    parser.add_argument("--output-prefix", default="keirin_results_20240101_20240331_full", help="Base name for output files")
    parser.add_argument("--write-parquet", action="store_true", help="Also write Parquet outputs (requires pyarrow/fastparquet)")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    paths = sorted(p for p in data_dir.glob(args.pattern) if args.output_prefix not in p.name)
    if not paths:
        raise SystemExit(f"No files matching {args.pattern} in {data_dir}")

    df = read_result_csvs(paths)
    if df.empty:
        raise SystemExit("Input CSVs contain no rows")

    df["race_date"] = df["race_date"].astype(str).str.extract(r"(\d{8})")[0].fillna(df["race_date"].astype(str))
    df["race_no_num"] = normalize_race_no(df["race_no"])
    df["keirin_cd"] = pd.to_numeric(df["keirin_cd"], errors="coerce").astype("Float64")
    df["keirin_cd_str"] = df["keirin_cd"].fillna(-1).astype(int).astype(str).str.zfill(2)
    df["race_no_str"] = df["race_no_num"].fillna(-1).astype(int).astype(str).str.zfill(2)
    df["race_id"] = df["keirin_cd_str"] + "_" + df["race_date"] + "_" + df["race_no_str"]
    df = df.drop(columns=["keirin_cd_str", "race_no_str"])

    df = df.sort_values(["race_id", "source_mtime", "source_file"]).drop_duplicates(subset=["race_id"], keep="last")
    df = df.drop(columns=["source_mtime"])

    output_base = data_dir / args.output_prefix
    csv_path = f"{output_base}.csv"
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")

    outputs = {
        "csv": csv_path,
        "rows": int(len(df)),
        "sources": [p.name for p in paths],
    }

    if args.write_parquet:
        parquet_path = f"{output_base}.parquet"
        df.to_parquet(parquet_path, index=False)
        outputs["parquet"] = parquet_path

    summary_path = f"{output_base}_summary.json"
    Path(summary_path).write_text(json.dumps(outputs, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(outputs, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
