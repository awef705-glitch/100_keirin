#!/bin/bash
# モデルを最新のWeb検索データで再訓練

echo "=================================="
echo "RETRAINING MODEL WITH WEB DATA"
echo "=================================="
echo ""

# Backup old model
if [ -f "analysis/model_outputs/high_payout_model_lgbm.txt" ]; then
    echo "Backing up old model..."
    cp analysis/model_outputs/high_payout_model_lgbm.txt \
       analysis/model_outputs/high_payout_model_lgbm_backup_$(date +%Y%m%d).txt
fi

# Check if we have the required data files
if [ ! -f "data/keirin_results_20240101_20251004.csv" ]; then
    echo "ERROR: Missing results file"
    exit 1
fi

# Train with CV
echo "Training model with time-series cross-validation..."
echo ""

python analysis/train_high_payout_model_cv.py \
    --results data/keirin_results_20240101_20251004.csv \
    --prerace data/keirin_prerace_20240101_20251004.csv \
    --entries data/keirin_race_detail_entries_20240101_20251004.csv \
    --threshold 10000 \
    --folds 5

echo ""
echo "=================================="
echo "Training complete!"
echo "=================================="
echo ""

# Show metrics
if [ -f "analysis/model_outputs/high_payout_model_lgbm_metrics.json" ]; then
    echo "Model metrics:"
    cat analysis/model_outputs/high_payout_model_lgbm_metrics.json | python -m json.tool | head -30
fi
