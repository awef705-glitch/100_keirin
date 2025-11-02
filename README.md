# 競輪 高配当予測プロジェクト

三連単で 10,000 円以上の的中を狙うための機械学習ワークフローです。LightGBM モデルを使い、レース前の情報だけで「荒れ度」と推奨ベットプランを提示します。

---

## 主な機能
- LightGBM + 時系列クロスバリデーションで精度検証済み
- CLI (easy_predict.py, predict_race.py) で高速に候補レースを抽出
- FastAPI (web_app.py) によるモバイル最適化 UI
- 推奨買い目（リスクレベル別フォーメーション）と高配当率を即時表示
- Render や Railway にそのままデプロイ可能

---

## クイックスタート（ローカル）
1. **仮想環境を有効化**
   `powershell
   .\.venv\Scripts\Activate.ps1
   `
2. **依存パッケージを確認**（インストール済みの場合はスキップ）
   `powershell
   pip install -r requirements.txt
   `
3. **Web UI を起動**
   `powershell
   python web_app.py
   `
   - 表示される URL（例: http://127.0.0.1:8000/）を PC のブラウザで開く
   - iPhone からアクセスする場合は PC と同じネットワークに接続し、http://<PCのIP>:8000/ を開く
4. **CLI モードで検証**
   `powershell
   python predict_race.py --interactive
   `

---

## iPhone 単体で使う（クラウドデプロイ）
1. GitHub リポジトリを Render / Railway などに接続
2. ビルドコマンド: pip install -r requirements.txt
3. スタートコマンド: uvicorn web_app:app --host 0.0.0.0 --port 
4. デプロイ完了後に生成される URL を iPhone のブラウザで開けば、PC を介さず利用できます

> web_app.py は HOST / PORT 環境変数を自動で読み取りそのまま起動します

---

## ファイル構成
`
analysis/            # 学習・特徴量生成ロジック
analysis/model_outputs/
  ├── prerace_model_lgbm.txt
  ├── prerace_model_metadata.json
  └── prerace_model_feature_importance.csv
scripts/             # データ収集スクリプト
templates/           # Web UI (FastAPI + Jinja2)
web_app.py           # モバイル対応 Web サーバー
predict_race.py      # 対話式 CLI
requirements.txt
`

※ 生データは data/ に配置しますが Git 管理対象外です（data/README.md 参照）。

---

## 出力内容
- **高配当率**：三連単で 1 万円超となる確率をパーセンテージ表示
- **推奨ベットプラン**：リスクレベル別にフォーメーション案と資金配分・ヘッジのヒントを提示
- **レースコンディション**：入力した天候・バンク状態・ナイター情報をまとめて表示
- **特徴量サマリ**：平均得点や脚質比率など、モデルが重視した指標

---

## 今後の改善アイデア
- オッズや天候 API を組み込んだ特徴量強化
- ベットプランの自動最適化（点数・配当期待値の推定）
- バックテスト用シミュレーション＆ダッシュボード

---

## ライセンス / 注意事項
研究・教育目的で公開しています。実際の投票に伴う損益は自己責任でお願いします。
