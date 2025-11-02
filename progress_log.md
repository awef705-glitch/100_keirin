# 作業ログ

## 2025-10-13

### データ取得（SJ0315）
- 2024/05〜2025/10 の欠損レースを約5万件バッチ取得。
- scripts/fetch_keirin_race_detail.py を改良し、出走・結果を keirin_race_detail_{race|entries}_YYYYMMDD_YYYYMMDD.csv に出力。
- 取得データをマスター (data/keirin_race_detail_{race|entries}_20240101_20251004.*) に統合し、欠損ゼロを確認。
- 取りこぼしていた 2024/01/01・2024/04/11・2024/04/24 の 3 レースを個別再取得。
- data/keirin_race_detail_summary_20240101_20251004.json を更新し、レース 48,758 件／エントリ 346,013 件を確認。

### オッズ API 調査
- スマホ版 JSON (https://keirin.jp/sp/json) の JST069（開催情報）、JST019/JST020（オッズ画面用）にアクセスできることを確認。
- scripts/fetch_keirin_odds_sample.py を作成し、1 レース分のメタデータとオッズ payload を取得できることを確認。
  - 例: `python scripts/fetch_keirin_odds_sample.py 20251013 47 1 3 --output-dir analysis/odds_samples`
- scripts/fetch_keirin_odds_batch.py を追加し、prerace CSV を基に巡回取得できるようにした。
  - サンプルは 2025/10/04 分を analysis/odds_payloads_test/ に保存。
- 現状 JST020 の payload には賭式可否・選手リストは含まれるが、オッズ数値は含まれていない。searchOzz パラメータの指定有無で JSON 形状は変化するが、数値は返ってこない。

### その他
- 調査用 HTML (analysis/tmp_odds*.html) を保存し、API との対応関係をメモ。
- 必要なログは progress_log.md に追記。
- 課題: searchOzz の扱いやオッズ取得バッチに向けた準備を整理。

## 2025-10-14

### オッズ取得フローの調査継続
- JST020 リクエストで searchOzz を指定しても JSON スキーマは変わるが、オッズ数値は依然欠損。
- oddsselect → odds の画面遷移時、フォーム POST (ST0101) を再現しようとしたが EC0001E のエラーが発生。
- javax.faces.ViewState や hidden フィールドの値が動的に設定されるため、再現には追加のパラメータが必要と判断。
- 暫定スクリプト analysis/tmp_odds_request.py, analysis/tmp_fetch_board.py, analysis/tmp_post_odds.py を作成し、取得フローの切り出しを開始。

## 2025-10-19

### 高配当モデルのベースライン構築
- analysis/train_high_payout_model.py を実装し、結果CSV・出走表・選手明細を結合して特徴量を生成。
- HistGradientBoostingClassifier で三連単 10,000 円以上を陽性とした分類モデルを学習。
- ROC-AUC 0.995／Accuracy 0.966（テスト 20%）。モデルと指標は analysis/model_outputs/ に保存。

### オッズ POST 調査メモ
- 20251004 函館 1R（三連単）を対象に requests.Session で以下を実行。
  1. GET /sp/
  2. POST /sp/oddsselect（encp=8vdVlO5IxwzRzvGxziu8NmbAWj4uzUACOR2D9THOgWE）
  3. GET /sp/json?type=JST069 ...
  4. GET /sp/json?type=JST019 ...
  5. POST /sp/odds (disp=ST0101, bkcd=11, kday=20251004, rnum=1, kake=3, mode=1 …)
- 手順を踏んでもレスポンスは EC0001E。kimHdnNotSS や encp, hoji を付与しても変化なし。
- oddsselect の hidden 値を抽出した結果、UNQ_hidden_04/05/06・UNQ_hojiScreenId は初期状態では空。実ブラウザでのボタンクリックにより埋まる想定のため、開発者ツールでの再取得が必要。

### Playwright による通信キャプチャ
- Playwright（Chromium／Pixel 5 エミュレーション）で `/sp/oddsselect` を自動操作するスクリプト `analysis/capture_odds_network.py` を作成。
- `btnKeirinjyo` → `raceNo` → `賭式（三連単）` → `表示` のクリックを強制し、`/sp/json?type=JST060` および `/sp/odds` の通信（POST パラメータ含む）を `analysis/playwright_logs/network_requests.txt` に保存。
- `/sp/odds` のレスポンス HTML はテンプレートのみで、オッズ数値が含まれない（`mainOzzData=''`）。JST020 も賭式可否・選手リストのみでオッズ数値は取得できず、追加の非公開 API もしくは会員専用通信が存在する可能性が高い。
- hidden フィールドのスナップショット（hidden_inputs.json）とスクリーンショット（oddsselect_page.png）を保存し、JS が設定する値を確認可能にした。

### 特徴量拡張と LightGBM モデル
- analysis/train_high_payout_model.py に派生特徴量生成を追加し、平均得点レンジや脚質比率、出走数比などを計算可能にした。
- 同スクリプトの特徴量選択ロジックを `select_feature_columns` として切り出し、他モデルでも再利用。
- 新規スクリプト `analysis/train_high_payout_model_cv.py` を作成。LightGBM＋TimeSeriesSplit（5分割）で交差検証を実施し、ROC-AUC 0.8406／Average Precision 0.8632 を確認。詳細指標は `analysis/model_outputs/high_payout_model_lgbm_metrics.json` に保存。
- 交差検証後、全データで LightGBM を再学習し、モデルファイル `analysis/model_outputs/high_payout_model_lgbm.txt` を出力。

### データセット整備と推論スクリプト
- analysis/build_training_dataset.py で派生特徴量付きデータセットを一括生成（analysis/model_outputs/training_dataset.csv）。
- analysis/predict_high_payout.py で LightGBM モデルからトップ候補をCSV出力できるようにした。トップK精度や日付フィルタにも対応済み。
- FastAPI サービス `analysis/inference_service.py` を追加。`uvicorn analysis.inference_service:app --reload` で起動し、iPhone 等から JSON POST で推論可能。

