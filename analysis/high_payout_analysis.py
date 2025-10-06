import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, classification_report,
                             roc_auc_score)
from sklearn.model_selection import train_test_split


def parse_payout(value: str) -> float:
    if pd.isna(value) or value == "":
        return np.nan
    digits = re.sub(r"[^0-9]", "", str(value))
    return float(digits) if digits else np.nan


def build_dataset(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df["trifecta_payout_value"] = df["trifecta_payout"].apply(parse_payout)
    df = df.dropna(subset=["trifecta_payout_value"])
    df["high_payout"] = (df["trifecta_payout_value"] >= 10000).astype(int)

    # Basic numeric descriptors from車番
    for pos in (1, 2, 3):
        df[f"pos{pos}_car_no"] = pd.to_numeric(df[f"pos{pos}_car_no"], errors="coerce")
    df["car_sum"] = df[["pos1_car_no", "pos2_car_no", "pos3_car_no"]].sum(axis=1)
    df["car_std"] = df[["pos1_car_no", "pos2_car_no", "pos3_car_no"]].std(axis=1)
    df["car_range"] = df[["pos1_car_no", "pos2_car_no", "pos3_car_no"]].max(axis=1) - df[["pos1_car_no", "pos2_car_no", "pos3_car_no"]].min(axis=1)
    df["car_median"] = df[["pos1_car_no", "pos2_car_no", "pos3_car_no"]].median(axis=1)
    df["car_min"] = df[["pos1_car_no", "pos2_car_no", "pos3_car_no"]].min(axis=1)
    df["car_max"] = df[["pos1_car_no", "pos2_car_no", "pos3_car_no"]].max(axis=1)

    # Clean categorical columns (fill NA with label)
    cat_cols = [
        "grade",
        "track",
        "category",
        "race_no",
        "meeting_icon",
        "pos1_decision",
        "pos2_decision",
        "pos3_decision",
        "pos1_name",
        "pos2_name",
        "pos3_name",
    ]
    for col in cat_cols:
        df[col] = df[col].fillna("(欠損)").astype(str)

    # Normalise race number to remove trailing 'R'
    df["race_no"] = df["race_no"].str.upper().str.replace("R", "", regex=False)

    numeric_cols = ["car_sum", "car_std", "car_range", "car_median", "car_min", "car_max"]
    for col in numeric_cols:
        mean = df[col].mean()
        std = df[col].std(ddof=0)
        if std == 0:
            df[col] = 0
        else:
            df[col] = (df[col] - mean) / std

    X_cat = pd.get_dummies(df[cat_cols], prefix=cat_cols, dummy_na=False)
    X_num = df[numeric_cols]
    X = pd.concat([X_cat, X_num], axis=1)
    y = df["high_payout"].values
    return X, y, df


def train_model(X: pd.DataFrame, y: np.ndarray):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    model = LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        n_jobs=None,
    )
    model.fit(X_train, y_train)
    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= 0.5).astype(int)
    metrics = {
        "auc": roc_auc_score(y_test, y_proba),
        "average_precision": average_precision_score(y_test, y_proba),
        "classification_report": classification_report(y_test, y_pred, output_dict=True),
        "positive_rate": float(y.mean()),
    }
    return model, metrics, X.columns


def aggregate_importance(model: LogisticRegression, feature_names: pd.Index) -> pd.DataFrame:
    coef = np.abs(model.coef_[0])
    groups = []
    for name in feature_names:
        if name in {"car_sum", "car_std", "car_range", "car_median", "car_min", "car_max"}:
            groups.append(name)
        else:
            base = name.split("_", 1)[0]
            groups.append(base)
    importance = pd.DataFrame({
        "feature": feature_names,
        "group": groups,
        "importance": coef,
    })
    grouped = importance.groupby("group")["importance"].sum().sort_values(ascending=False)
    total = grouped.sum()
    if total == 0:
        grouped_percent = grouped * 0
    else:
        grouped_percent = grouped / total * 100
    result = pd.DataFrame({
        "group": grouped_percent.index,
        "importance_percent": grouped_percent.values,
        "raw_importance": grouped.loc[grouped_percent.index].values,
    })
    return result


def main():
    parser = argparse.ArgumentParser(description="Analyse high-payout patterns from keirin results")
    parser.add_argument("--input", default="data/keirin_results_20240101_20251004.csv", help="CSV file to analyse")
    parser.add_argument("--output", default="analysis/high_payout_scoring.json", help="Path to write summary JSON")
    args = parser.parse_args()

    csv_path = Path(args.input)
    if not csv_path.exists():
        raise SystemExit(f"Input file not found: {csv_path}")

    X, y, df = build_dataset(csv_path)
    model, metrics, feature_names = train_model(X, y)
    importance_df = aggregate_importance(model, feature_names)

    summary = {
        "input_file": str(csv_path),
        "num_records": int(df.shape[0]),
        "positive_count": int(df["high_payout"].sum()),
        "metrics": metrics,
        "group_importance": importance_df.head(20).to_dict(orient="records"),
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
