# Render デプロイ作業まとめ（2025-11-02）

## 実施したこと
- `web_app.py` に LightGBM オプション化・起動時のアクセス案内強化・テンプレート欠落時の警告を追加。
- 入力フォーム (`templates/index.html`) をモバイル向けに整理し、任意項目を折りたたみ表示に変更。
- 結果画面 (`templates/result.html`) とエラー画面 (`templates/error.html`) をヒューリスティック利用時でも分かりやすい文言に更新。
- `analysis/prerace_model.py` の脚質正規化ロジックを改善し、UI からの入力を確実にマッピング。
- Render の Web Service を作成し、GitHub リポジトリから自動デプロイする設定を構築。

## Render 側の設定
- **Service 名**: keirin-web（任意）
- **Runtime**: Python 3
- **Build Command**: `pip install -r requirements.txt`
- **Start Command**: `uvicorn web_app:app --host 0.0.0.0 --port $PORT`
- **環境変数**
  - `PYTHON_VERSION = 3.11.7`
  - 必要に応じて `KEIRIN_ENABLE_LIGHTGBM = 1`（モデルを使う場合）
- **リポジトリ**: `https://github.com/awef705-glitch/100_keirin` の `main` ブランチ
- デプロイ完了後に付与された公開 URL を iPhone の Safari で開き、「ホーム画面に追加」で常駐可能。

## 現状
- Render 側でビルド／起動が成功し、Web アプリが公開状態。
- `git status` 上ではテンプレート関連・スクリプト修正が追跡対象になっているため、ローカル変更をコミットして GitHub と Render の内容を同期させた。

## 今後のメンテ
- モデルを更新した場合は `analysis/model_outputs/` 配下のファイルもコミットして再デプロイ。
- 無料プランは 15 分以上アクセスが無いとスリープする点に注意（有料プランで常時稼働が可能）。
- 環境変数の変更や追加デプロイが必要になった際は Render ダッシュボードの「Environment」タブから編集。

