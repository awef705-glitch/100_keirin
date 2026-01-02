import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, precision_score, recall_score, log_loss, brier_score_loss
from pathlib import Path

def evaluate_performance():
    oof_path = Path("analysis/model_outputs/prerace_model_oof.csv")
    if not oof_path.exists():
        print("OOF file not found.")
        return

    df = pd.read_csv(oof_path)
    
    # 1. Overall Metrics
    auc = roc_auc_score(df['label'], df['prediction'])
    logloss = log_loss(df['label'], df['prediction'])
    brier = brier_score_loss(df['label'], df['prediction'])
    
    print(f"Overall AUC: {auc:.4f}")
    print(f"Log Loss: {logloss:.4f}")
    print(f"Brier Score: {brier:.4f}")
    
    # 2. Calibration Analysis (Binning)
    # Group predictions into bins (0-10%, 10-20%, etc.)
    bins = np.linspace(0, 1, 11)
    df['prob_bin'] = pd.cut(df['prediction'], bins=bins, labels=False)
    
    calibration = df.groupby('prob_bin').agg({
        'prediction': 'mean',
        'label': 'mean',
        'race_date': 'count'
    }).rename(columns={'race_date': 'count', 'prediction': 'avg_pred', 'label': 'actual_rate'})
    
    print("\nCalibration Analysis (Reliability):")
    print(calibration)
    
    # 3. Precision at Top-K (Simulating betting on top confidence races)
    # Sort by prediction confidence
    df_sorted = df.sort_values('prediction', ascending=False)
    
    thresholds = [100, 500, 1000, 5000, 10000]
    print("\nPrecision at Top-K Races:")
    for k in thresholds:
        if k > len(df):
            break
        top_k = df_sorted.head(k)
        precision = top_k['label'].mean()
        print(f"Top {k} races: Precision = {precision:.2%} (Avg Pred: {top_k['prediction'].mean():.2%})")

    # 4. Threshold Analysis
    # If we set a threshold of X%, what is the precision/recall?
    print("\nThreshold Analysis:")
    for thresh in [0.3, 0.4, 0.5, 0.6, 0.7]:
        pred_binary = (df['prediction'] >= thresh).astype(int)
        if pred_binary.sum() == 0:
            continue
        prec = precision_score(df['label'], pred_binary)
        rec = recall_score(df['label'], pred_binary)
        count = pred_binary.sum()
        print(f"Threshold >= {thresh:.1f}: Count={count}, Precision={prec:.2%}, Recall={rec:.2%}")

    # 5. Save Report
    with open("analysis/model_outputs/evaluation_report.txt", "w", encoding="utf-8") as f:
        f.write(f"Overall AUC: {auc:.4f}\n")
        f.write(f"Log Loss: {logloss:.4f}\n")
        f.write("\nCalibration Analysis:\n")
        f.write(calibration.to_string())
        f.write("\n\nPrecision at Top-K:\n")
        for k in thresholds:
            if k <= len(df):
                top_k = df_sorted.head(k)
                f.write(f"Top {k}: {top_k['label'].mean():.2%}\n")

if __name__ == "__main__":
    evaluate_performance()
