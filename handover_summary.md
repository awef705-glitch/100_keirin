# KEIRIN High-Payout Project Handover Memo

## Goal
- Integrate 2024/01-2025/10 race data (results, entries, rider profiles, odds, external info) and build a model that predicts high-payouts (e.g., trifecta ≥10,000 JPY).

## Progress So Far
1. **Race detail (SJ0315) dataset completed**
   - Collected missing races for 2024/05-2025/10 with scripts/fetch_keirin_race_detail.py.
   - Consolidated into master files (data/keirin_race_detail_{race|entries}_20240101_20251004.*) with 48,758 races / 346,013 entries and no gaps.
   - Re-fetched the missing 2024/01/01, 2024/04/11, 2024/04/24 races individually.

2. **Mobile JSON investigation & scripts**
   - Confirmed access to https://keirin.jp/sp/json JST069 (schedule) and JST019/JST020 (odds screens).
   - Created scripts/fetch_keirin_odds_sample.py and scripts/fetch_keirin_odds_batch.py; sample outputs stored in analysis/odds_samples/ and analysis/odds_payloads_test/.
   - Responses include betting availability and rider lists, but numeric odds are absent (searchOzz parameter variations do not return the numbers).

3. **Baseline high-payout model**
   - analysis/train_high_payout_model.py merges results, prerace, and entries into features; trained HistGradientBoostingClassifier for trifecta ≥10,000 JPY.
   - Hold-out (20%) performance: ROC-AUC 0.995, accuracy 0.966. Model/metrics saved under analysis/model_outputs/.

4. **Playwright network capture**
   - analysis/capture_odds_network.py drives Chromium (Pixel 5 emulation) through /sp/oddsselect; captures JST060 and /sp/odds requests (parameters saved in analysis/playwright_logs/).
   - Extracted POST parameters (disp, bkcd, kday, rnum, kake, mode, searchOzz, hoji). Response HTML remains a template (mainOzzData empty) with no odds numbers.
   - Saved hidden-input snapshots and screenshots for reference.

5. **Feature expansion & LightGBM model**
   - Added derived features and reusable column selection (`select_feature_columns`) to train_high_payout_model.py.
   - Built analysis/train_high_payout_model_cv.py for LightGBM + TimeSeriesSplit (5 folds). Cross-val: ROC-AUC 0.8406, Average Precision 0.8632 (metrics in analysis/model_outputs/high_payout_model_lgbm_metrics.json).
   - Trained final LightGBM on full data; model saved to analysis/model_outputs/high_payout_model_lgbm.txt.
6. **Dataset builder & inference pipeline**
   - analysis/build_training_dataset.py generates the feature dataset (analysis/model_outputs/training_dataset.csv).
   - analysis/predict_high_payout.py loads the LightGBM model and outputs top-K race recommendations (CSV).
   - analysis/inference_service.py exposes a FastAPI endpoint for mobile/JSON clients.
7. **Miscellaneous**
   - Investigation artifacts (analysis/sample_odds*.json, etc.) remain for reference.
   - Progress log maintained in progress_log.md.

## Next Actions
1. **Pin down the odds numbers**
   - Use Playwright captures to identify additional requests triggered after /sp/odds (likely authenticated or internal APIs). Re-run DevTools capture on a real browser to map hidden field values.

2. **If the above fails**
   - Re-check PC Data Plaza (keirin.jp/pc/dfw/dataplaza/...) for any public odds endpoints or licensed data feeds.

3. **Post-odds pipeline**
   - Document betting parameter specs (bkcd/kday/rnum/kake/mode/searchOzz) and design a batch crawler.
   - Define storage schema and matching keys (race_date + bkeirin_cd + race_no) to merge odds with existing datasets.

## References
- Detailed log: progress_log.md (2025-10-13/14/19 entries).
- Investigation scripts: analysis/tmp_odds_request.py, analysis/tmp_fetch_board.py, analysis/tmp_post_odds.py.
- Playwright logs: analysis/playwright_logs/network_requests.txt, hidden_inputs.json, odds_response_*.html.
- Models: analysis/model_outputs/high_payout_model.joblib (HistGB) and high_payout_model_lgbm.txt (LightGBM), with respective metrics JSON files.
