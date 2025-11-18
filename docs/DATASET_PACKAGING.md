# Keirin Dataset Packaging Guide

Large CSVs under `data/` are intentionally kept out of Git to avoid bloating the repository.  
Use `scripts/package_keirin_dataset.py` to create a portable ZIP that bundles every generated dataset (pre-race, SJ0315 race detail, SJ0306 results, rider profiles, training dataset, and summary JSON files).

## Requirements

```powershell
pip install -r requirements.txt
```

## Create the ZIP archive

```powershell
python scripts/package_keirin_dataset.py --output dist/keirin_dataset_bundle.zip
```

### Arguments

| Argument | Default | Description |
| --- | --- | --- |
| `--data-dir` | `data` | Root directory that stores CSV/JSON files. |
| `--output` | `dist/keirin_dataset_bundle.zip` | Path of the resulting archive. |

The script automatically adds the following patterns if they exist:
- `keirin_prerace_*.csv`, `keirin_prerace_*_summary.json`
- `keirin_race_detail_race_*.csv`, `keirin_race_detail_entries_*.csv`, `keirin_race_detail_summary_*.json`
- `keirin_results_*.csv`, `keirin_results_*.json`
- `keirin_rider_profiles_*.csv`, `keirin_rider_profiles_*.summary.json`
- `keirin_training_dataset_*.csv`, `keirin_training_dataset_*.summary.json`

## Share with teammates

1. Run the script to generate the ZIP.
2. Upload `dist/keirin_dataset_bundle.zip` to your preferred storage (S3, Drive, etc.).
3. Share the link with teammates. They can unzip it into the repository root and immediately have access to the same datasets.
