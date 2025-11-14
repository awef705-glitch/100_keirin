#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Build rider master data from race entries for autocomplete."""

import json
from pathlib import Path
import pandas as pd


def build_rider_master(entries_path: Path, output_path: Path) -> None:
    """Create rider master JSON from entries CSV."""
    print(f"Loading entries from {entries_path}...")
    entries = pd.read_csv(entries_path)

    # Sort by date to get most recent info
    entries['race_date'] = entries['race_date'].astype(int)
    entries_sorted = entries.sort_values('race_date', ascending=False)

    # Group by rider name and take most recent entry
    print("Aggregating rider information...")
    rider_master = entries_sorted.groupby('simei').first().reset_index()

    # Select relevant columns
    rider_data = rider_master[['simei', 'huKen', 'kyuhan', 'kyakusitu', 'heikinTokuten']].copy()
    rider_data.columns = ['name', 'prefecture', 'grade', 'style', 'avg_score']

    # Clean data
    rider_data = rider_data.dropna(subset=['name'])
    rider_data['name'] = rider_data['name'].str.strip()
    rider_data = rider_data[rider_data['name'].str.len() > 0]

    # Remove invalid scores
    rider_data['avg_score'] = pd.to_numeric(rider_data['avg_score'], errors='coerce')

    # Convert to list of dicts
    riders_list = []
    for _, row in rider_data.iterrows():
        rider = {
            'name': str(row['name']),
            'prefecture': str(row['prefecture']) if pd.notna(row['prefecture']) else '',
            'grade': str(row['grade']) if pd.notna(row['grade']) else '',
            'style': str(row['style']) if pd.notna(row['style']) else '',
            'avg_score': float(row['avg_score']) if pd.notna(row['avg_score']) else None
        }
        riders_list.append(rider)

    # Sort by name
    riders_list.sort(key=lambda x: x['name'])

    # Save as JSON
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(riders_list, f, ensure_ascii=False, indent=2)

    print(f"Created rider master data: {len(riders_list)} riders")
    print(f"Saved to: {output_path}")
    if riders_list:
        print(f"Sample: {riders_list[0]}")


def build_track_master(results_path: Path, output_path: Path) -> None:
    """Create track master JSON from results CSV."""
    print(f"Loading results from {results_path}...")
    results = pd.read_csv(results_path)

    # Get unique tracks with their codes
    tracks = results[['keirin_cd', 'track']].drop_duplicates()
    tracks = tracks.dropna()
    tracks['keirin_cd'] = tracks['keirin_cd'].astype(str).str.zfill(2)
    tracks['track'] = tracks['track'].str.strip()

    # Sort by code
    tracks = tracks.sort_values('keirin_cd')

    # Convert to list of dicts
    tracks_list = [
        {'code': row['keirin_cd'], 'name': row['track']}
        for _, row in tracks.iterrows()
    ]

    # Save as JSON
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(tracks_list, f, ensure_ascii=False, indent=2)

    print(f"Created track master data: {len(tracks_list)} tracks")
    print(f"Saved to: {output_path}")
    if tracks_list:
        print(f"Sample: {tracks_list[0]}")


if __name__ == "__main__":
    data_dir = Path("data")
    output_dir = Path("analysis/model_outputs")

    # Build rider master
    entries_path = data_dir / "keirin_race_detail_entries_20240101_20251004.csv"
    rider_output = output_dir / "rider_master.json"
    build_rider_master(entries_path, rider_output)

    print()

    # Build track master
    results_path = data_dir / "keirin_results_20240101_20251004.csv"
    track_output = output_dir / "track_master.json"
    build_track_master(results_path, track_output)
