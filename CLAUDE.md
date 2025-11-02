# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## プロジェクト概要

競輪の三連単で**10,000円以上の高配当を予測する機械学習プロジェクト**です。LightGBMと時系列クロスバリデーションを使用し、ROC-AUC 0.841、Precision@Top100 = 1.0を達成しています。

**データ期間**: 2024年1月1日〜2025年10月4日（48,700レース）
**言語**: 日本語・英語混在（ドキュメントは日本語、コードは英語）

## よく使うコマンド

### 予測の実行（最も使用頻度が高い）

```bash
# シンプルな予測 - トップ100の高配当レースを表示
python easy_predict.py

# 日付でフィルタ
python easy_predict.py --date 20241025

# スコアでフィルタしてCSVに保存
python easy_predict.py --min-score 0.9 --top-k 50 --output predictions.csv
```

### モデルの訓練

```bash
# メインのLightGBMモデルをクロスバリデーション付きで訓練（推奨）
python analysis/train_high_payout_model_cv.py \
  --results data/keirin_results_20240101_20251004.csv \
  --prerace data/keirin_prerace_20240101_20251004.csv \
  --entries data/keirin_race_detail_entries_20240101_20251004.csv \
  --threshold 10000 \
  --folds 5

# ハイパーパラメータのグリッドサーチを含む
python analysis/train_high_payout_model_cv.py --grid-search

# 訓練用データセットの構築（parquet/csvを作成）
python analysis/build_training_dataset.py
```

### データ収集

```bash
# レース結果の取得（実際の配当情報）
python scripts/fetch_keirin_results.py --start 20240101 --end 20241231

# レース前情報の取得
python scripts/fetch_keirin_prerace.py --start 20240101 --end 20241231

# レース詳細エントリデータの取得
python scripts/fetch_keirin_race_detail.py --start 20240101 --end 20241231
```

### APIサービス

```bash
# FastAPI推論サービスの起動
uvicorn analysis.inference_service:app --reload

# http://localhost:8000 でアクセス
# POST /predict エンドポイントで予測
```

### Webアプリ

```bash
# スマホ最適化Webアプリの起動
python web_app.py

# http://localhost:8000 でアクセス
```

### 対話的な予測

```bash
# 対話モード
python predict_race.py --interactive

# JSONファイルから読み込み
python predict_race.py --file sample_race.json
```

## アーキテクチャ

### データパイプラインの流れ

1. **データ収集** (`scripts/`)
   - `fetch_keirin_results.py` → レース結果と配当をスクレイピング
   - `fetch_keirin_prerace.py` → レース前情報をスクレイピング
   - `fetch_keirin_race_detail.py` → 詳細エントリデータをスクレイピング
   - すべて `data/` ディレクトリに日付範囲を含むファイル名で出力

2. **データセット構築** (`analysis/build_training_dataset.py`)
   - results、prerace、entriesデータをマージ
   - 派生特徴量を作成（選手統計のCV、std、mean）
   - `analysis/model_outputs/training_dataset.parquet` に出力

3. **モデル訓練** (`analysis/train_high_payout_model_cv.py`)
   - TimeSeriesSplit（5分割）を使用してデータリーケージを防止
   - バランスクラスウェイト付きでLightGBMを訓練
   - `analysis/model_outputs/high_payout_model_lgbm.txt` にモデルを出力
   - `analysis/model_outputs/high_payout_model_lgbm_metrics.json` にメトリクスを出力
   - `analysis/model_outputs/high_payout_model_lgbm_oof.csv` にOOF予測を出力

4. **推論** （複数のインターフェース）
   - `easy_predict.py` → OOF予測を読み込んで高速分析
   - `predict_race.py` → 手動入力からルールベース予測
   - `analysis/inference_service.py` → 訓練済みモデルを使ったFastAPIサービス
   - `web_app.py` → スマホ対応Webインターフェース

### 主要なデータ構造

**レース識別情報**:
- `race_date` (int): YYYYMMDD形式（例: 20241025）
- `keirin_cd` (str): 2桁の会場コード（例: "27" は京王閣）
- `race_no` または `race_no_int` (int): レース番号 1-12

**特徴量**（重要度順）:
1. `trifecta_popularity` - 圧倒的に最重要（三連単の人気順位）
2. `heikinTokuten_cv` - 選手平均得点の変動係数
3. `category` - レースカテゴリ（S級、A級など）
4. `track` - トラックの種類/会場
5. `nigeCnt_cv`, `makuriCnt_cv` など - 脚質のばらつき

**目的変数**:
- `target_high_payout` - 二値（配当 ≥ 閾値なら1、それ以外は0）

### モジュール依存関係

```
train_high_payout_model.py (ベース)
    ↓ (インポートされる)
train_high_payout_model_cv.py
    ↓ (使用される)
build_training_dataset.py

inference_utils.py
    ↓ (インポートされる)
inference_service.py

predict_race.py
    ↓ (インポートされる)
web_app.py
```

## 重要な実装ノート

### 時系列データの取り扱い

**重要**: クロスバリデーションには必ず `TimeSeriesSplit` を使用してください。このデータセットは `race_date` で時系列順に並んでいます。通常のK-Foldを使うと未来のデータが訓練に漏れてしまいます。

```python
from sklearn.model_selection import TimeSeriesSplit
tscv = TimeSeriesSplit(n_splits=5)
```

### 特徴量エンジニアリングのパターン

すべての特徴量エンジニアリングは `train_high_payout_model.py:add_derived_features()` で行われます:
- 選手統計の集約（mean、std、cv）
- track、category、gradeのカテゴリカルエンコーディング
- 欠損値の処理（数値は0で埋める、カテゴリカルは"unknown"）

新しい特徴量を追加する場合:
1. `add_derived_features()` にロジックを追加
2. `select_feature_columns()` で新しい特徴量を追加
3. `train_high_payout_model_cv.py` でモデルを再訓練

### モデル出力ファイル

訓練パイプラインは `analysis/model_outputs/` に以下を生成します:
- `high_payout_model_lgbm.txt` - 訓練済みLightGBMモデル（`lgb.Booster(model_file=...)` で読み込み可能）
- `high_payout_model_lgbm_metrics.json` - 性能指標とハイパーパラメータ
- `high_payout_model_lgbm_oof.csv` - すべての訓練データのOut-of-fold予測
- `high_payout_model_lgbm_feature_importance.csv` - 特徴量重要度ランキング

`easy_predict.py` スクリプトは過去データの高速予測のため、OOFファイルを直接読み込みます。

### カテゴリカル特徴量

LightGBMのネイティブカテゴリカルサポートを使用します。カテゴリカル特徴量は必ず `category` dtypeに設定してください:

```python
for col in categorical_features:
    X[col] = X[col].astype("category")
```

カテゴリカル特徴量: `track`, `category`, `grade`, `meeting_icon`

### データファイルの命名規則

ファイルは `{タイプ}_{開始日}_{終了日}.{拡張子}` のパターンに従います:
- 例: `keirin_results_20240101_20251004.csv`
- 日付はYYYYMMDD形式の範囲（両端含む）

## プロジェクト固有の規約

### 言語の使い分け
- コードコメント: 一貫性のため英語が望ましい
- ドキュメント（README、ANALYSIS_REPORT）: 日本語（対象ユーザー向け）
- 変数名: 英語 + 日本語ドメイン用語（例: `heikinTokuten`, `nigeCnt`）

### モデルのバージョニング
- モデルファイルにはバージョン番号を含めない
- 実験時は異なるファイル名で保存（例: `_v2`, `_experimental`）
- 「本番」モデルは常に `high_payout_model_lgbm.txt`

### テストのアプローチ
- 正式なユニットテストはなし（研究・個人プロジェクトのため）
- クロスバリデーションメトリクスで検証
- 既知の日付で `easy_predict.py` を使った手動テスト

## よくあるワークフロー

### 新しい特徴量の追加

1. `analysis/train_high_payout_model.py:add_derived_features()` を編集
2. `select_feature_columns()` で `numeric_features` または `categorical_features` に追加
3. 再訓練: `python analysis/train_high_payout_model_cv.py`
4. `high_payout_model_lgbm_feature_importance.csv` で影響を確認

### 予測精度が悪い場合のデバッグ

1. `analysis/model_outputs/high_payout_model_lgbm_metrics.json` で各foldのメトリクスを確認
2. `high_payout_model_lgbm_oof.csv` を読み込み、偽陽性/偽陰性を調査
3. `easy_predict.py --date YYYYMMDD` で特定の日付をチェック
4. データセット内の特徴量の値が破損していないか確認

### データの更新

1. 新しい日付範囲でfetchスクリプトを実行
2. `data/` のCSVファイルを連結または置換
3. データセット再構築: `python analysis/build_training_dataset.py`
4. 再訓練: `python analysis/train_high_payout_model_cv.py`

## 既知の制限事項

- **オッズデータなし**: モデルは人気順位をプロキシとして使用しているが、実際のオッズがあれば精度向上が期待できる
- **天候データなし**: 天候はレース結果に影響するが、現在キャプチャされていない
- **並び（"narabi"）詳細なし**: 並び情報は存在するが、完全には統合されていない
- **ルールベースのフォールバック**: `predict_race.py` はヒューリスティックを使用（訓練済みモデルは使わない）（軽量デモのための意図的な設計）

## 依存パッケージ

コアパッケージ（pipでインストール）:
```
pandas numpy scikit-learn lightgbm
fastapi uvicorn pydantic  # API用
jinja2  # Webアプリ用
```

オプション:
```
pyarrow  # parquetサポート用
joblib   # モデルシリアライゼーション（LightGBMのネイティブsaveの代替）
```
