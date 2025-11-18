import argparse
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split


def load_dataset(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["keirin_cd"] = df["keirin_cd"].fillna(0).astype(int).astype(str).str.zfill(2)
    numeric_cols = [
        "feature_score",
        "recent_finish_avg",
        "recent_finish_last",
        "experience_years",
        "nigeCnt",
        "makuriCnt",
        "sasiCnt",
        "markCnt",
        "backCnt",
        "raceResult1Syaban",
        "raceResult2Syaban",
        "backCnt1Syaban",
        "backCnt2Syaban",
        "race_entry_count",
        "race_score_mean",
        "race_score_std",
        "score_diff",
        "score_z",
        "score_rank",
        "score_percentile",
        "recent_finish_rank",
        "recent_finish_rel",
        "same_prefecture_ratio",
        "same_style_ratio",
        "is_escape_style",
        "is_chaser_style",
        "car_no_norm",
    ]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["high_payout_flag"] = pd.to_numeric(df["high_payout_flag"], errors="coerce")
    fill_zero_cols = [
        "recent_finish_last",
        "experience_years",
        "nigeCnt",
        "makuriCnt",
        "sasiCnt",
        "markCnt",
        "backCnt",
        "raceResult1Syaban",
        "raceResult2Syaban",
        "backCnt1Syaban",
        "backCnt2Syaban",
        "race_entry_count",
        "race_score_mean",
        "race_score_std",
        "score_diff",
        "score_z",
        "score_rank",
        "score_percentile",
        "recent_finish_rank",
        "recent_finish_rel",
        "same_prefecture_ratio",
        "same_style_ratio",
        "is_escape_style",
        "is_chaser_style",
        "car_no_norm",
    ]
    for col in fill_zero_cols:
        df[col] = df[col].fillna(0)
    df["prefecture"] = df["prefecture"].fillna("Unknown")
    df["region"] = df["region"].fillna("Unknown")
    df["style_profile"] = df["style_profile"].fillna("Unknown")
    df["kyakusitu"] = df["kyakusitu"].fillna("Unknown")
    df["grade"] = df["grade"].fillna("Unknown")
    df["category"] = df["category"].fillna("Unknown")
    df["track"] = df["track"].fillna(df["track_x"])
    df["track"] = df["track"].fillna("Unknown")
    df["syumoku"] = df["syumoku"].fillna("Unknown")
    df["feature_score"] = df["feature_score"].fillna(df["heikinTokuten"])
    df["recent_finish_last"] = df["recent_finish_last"].fillna(df["recent_finish_avg"])
    df["experience_years"] = df["experience_years"].fillna(0)
    df["nigeCnt"] = df["nigeCnt"].fillna(0)
    df["makuriCnt"] = df["makuriCnt"].fillna(0)
    df["sasiCnt"] = df["sasiCnt"].fillna(0)
    df["markCnt"] = df["markCnt"].fillna(0)
    df["backCnt"] = df["backCnt"].fillna(0)
    df["raceResult1Syaban"] = df["raceResult1Syaban"].fillna(0)
    df["raceResult2Syaban"] = df["raceResult2Syaban"].fillna(0)
    df["backCnt1Syaban"] = df["backCnt1Syaban"].fillna(0)
    df["backCnt2Syaban"] = df["backCnt2Syaban"].fillna(0)
    return df


def main():
    parser = argparse.ArgumentParser(description="Train LightGBM model on keirin dataset")
    parser.add_argument("--dataset", default="data/keirin_training_dataset_20240101_20240331.csv")
    parser.add_argument("--model-output", default="analysis/model_outputs/prerace_model_lgbm_new.txt")
    parser.add_argument("--summary-output", default="analysis/model_outputs/prerace_model_summary_new.json")
    parser.add_argument("--test-size", type=float, default=0.2)
    args = parser.parse_args()

    df = load_dataset(Path(args.dataset))
    feature_cols = [
        "feature_score",
        "recent_finish_avg",
        "recent_finish_last",
        "experience_years",
        "nigeCnt",
        "makuriCnt",
        "sasiCnt",
        "markCnt",
        "backCnt",
        "raceResult1Syaban",
        "raceResult2Syaban",
        "backCnt1Syaban",
        "backCnt2Syaban",
        "race_entry_count",
        "race_score_mean",
        "race_score_std",
        "score_diff",
        "score_z",
        "score_rank",
        "score_percentile",
        "recent_finish_rank",
        "recent_finish_rel",
        "same_prefecture_ratio",
        "same_style_ratio",
        "is_escape_style",
        "is_chaser_style",
        "car_no_norm",
    ]
    categorical_cols = ["grade", "kyakusitu", "prefecture", "region", "style_profile", "category", "track", "syumoku"]

    X = df[feature_cols + categorical_cols].copy()
    for col in categorical_cols:
        X[col] = X[col].astype("category")
    y = df["high_payout_flag"].fillna(0).astype(int).to_numpy()

    X_train, X_valid, y_train, y_valid = train_test_split(
        X, y, test_size=args.test_size, random_state=42, stratify=y
    )

    train_data = lgb.Dataset(X_train, label=y_train, categorical_feature=categorical_cols, free_raw_data=False)
    valid_data = lgb.Dataset(X_valid, label=y_valid, categorical_feature=categorical_cols, reference=train_data)

    params = {
        "objective": "binary",
        "metric": ["auc", "average_precision"],
        "learning_rate": 0.05,
        "num_leaves": 64,
        "feature_pre_filter": False,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "feature_fraction": 0.8,
        "min_data_in_leaf": 50,
    }

    model = lgb.train(
        params,
        train_data,
        num_boost_round=400,
        valid_sets=[train_data, valid_data],
        valid_names=["train", "valid"],
        callbacks=[lgb.log_evaluation(period=50)],
    )

    preds = model.predict(X_valid)
    auc = roc_auc_score(y_valid, preds)
    ap = average_precision_score(y_valid, preds)

    model_path = Path(args.model_output)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model.save_model(str(model_path))

    feature_importance = {
        name: int(value)
        for name, value in zip(feature_cols + categorical_cols, model.feature_importance())
    }
    summary = {
        "dataset": str(args.dataset),
        "model": str(model_path),
        "valid_auc": auc,
        "valid_average_precision": ap,
        "features": feature_cols + categorical_cols,
        "categorical_features": categorical_cols,
        "feature_importance": feature_importance,
    }
    summary_path = Path(args.summary_output)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
