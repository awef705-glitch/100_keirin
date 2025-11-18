import argparse
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED

DATA_PATTERNS = [
    "keirin_prerace_*.csv",
    "keirin_prerace_*_summary.json",
    "keirin_race_detail_race_*.csv",
    "keirin_race_detail_entries_*.csv",
    "keirin_race_detail_summary_*.json",
    "keirin_results_*.csv",
    "keirin_results_*.json",
    "keirin_rider_profiles_*.csv",
    "keirin_rider_profiles_*.summary.json",
    "keirin_training_dataset_*.csv",
    "keirin_training_dataset_*.summary.json",
]


def find_files(data_dir: Path) -> list[Path]:
    files: list[Path] = []
    for pattern in DATA_PATTERNS:
        files.extend(sorted(data_dir.glob(pattern)))
    return files


def main() -> None:
    parser = argparse.ArgumentParser(description="Package keirin data files into a zip archive")
    parser.add_argument("--data-dir", default="data", help="Directory that contains keirin CSV/JSON files")
    parser.add_argument("--output", default="dist/keirin_dataset_bundle.zip", help="Zip archive output path")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        raise SystemExit(f"Data directory not found: {data_dir}")

    files = find_files(data_dir)
    if not files:
        raise SystemExit("No keirin data files found to package.")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with ZipFile(output_path, "w", compression=ZIP_DEFLATED) as zf:
        for file_path in files:
            arcname = file_path.relative_to(data_dir.parent)
            zf.write(file_path, arcname)

    print(f"Packaged {len(files)} files into {output_path}")


if __name__ == "__main__":
    main()
