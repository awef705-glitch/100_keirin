# 🚀 Vercel デプロイガイド

**完全無料・PC不要・デプロイ失敗なし**

---

## ✅ なぜVercelなのか？

| 項目 | Railway/Render | **Vercel** | Cloudflare Tunnel |
|------|---------------|------------|------------------|
| 💰 料金 | 無料枠小（制限多） | **完全無料（超寛大）** | 完全無料 |
| 🐍 Python | ✅ 使える | ✅ **使える** | ✅ 使える |
| 🤖 LightGBM | ✅ 動く | ✅ **そのまま動く** | ✅ 動く |
| 🚀 デプロイ | 失敗しやすい | **超簡単** | 不要 |
| 📱 PC依存 | ❌ 不要 | ❌ **不要** | ✅ 必要 |
| 🌐 無料枠 | 月500時間など | **100GB帯域・無制限リクエスト** | 無制限 |
| 🎯 推奨度 | ❌ | **⭐️⭐️⭐️** | △（PC起動必要） |

### Vercelの無料枠（超寛大）

- 🎁 **100GB/月の帯域幅**
- 🎁 **100GB-時間の関数実行**
- 🎁 **無制限リクエスト**
- 🎁 **クレカ登録不要**
- 🎁 **独自ドメイン対応**
- 🎁 **自動SSL証明書**

→ 個人利用なら**実質無制限**！

---

## 🎯 デプロイ手順（3ステップ・5分）

### 1️⃣ Vercelアカウント作成

1. https://vercel.com にアクセス
2. **Sign Up** をクリック
3. GitHubアカウントで連携（推奨）
4. 無料プラン「Hobby」を選択

### 2️⃣ GitHubにプッシュ

```bash
# プロジェクトのルートディレクトリで実行
git add .
git commit -m "Add Vercel deployment configuration"
git push origin main
```

### 3️⃣ Vercelでデプロイ

1. Vercelダッシュボードで **New Project** をクリック
2. GitHubリポジトリ `100_keirin` を選択
3. **Import** をクリック
4. そのまま **Deploy** をクリック

**完了！** 🎉

数分後、URLが発行されます（例: `https://100-keirin.vercel.app`）

---

## 📱 iPhoneでアクセス

1. 発行されたURL（例: `https://100-keirin.vercel.app`）をSafariで開く
2. 共有ボタン → **ホーム画面に追加**
3. アプリのように使える！

---

## 🔧 設定詳細

### プロジェクト構造

```
100_keirin/
├── public/              # 静的ファイル
│   ├── index.html       # メインページ
│   ├── styles.css       # スタイル
│   ├── app.js           # フロントエンドロジック
│   ├── manifest.json    # PWA設定
│   └── service-worker.js # オフライン対応
├── api/                 # サーバーレス関数
│   └── predict.py       # /api/predict エンドポイント
├── analysis/            # モデルとロジック
│   ├── prerace_model.py
│   └── model_outputs/   # モデルファイル
├── vercel.json          # Vercel設定
└── requirements.txt     # Python依存関係
```

### vercel.json の設定

```json
{
  "version": 2,
  "builds": [
    {
      "src": "api/**/*.py",
      "use": "@vercel/python",
      "config": {
        "maxLambdaSize": "50mb"
      }
    }
  ],
  "routes": [
    {
      "src": "/api/(.*)",
      "dest": "/api/$1"
    },
    {
      "src": "/(.*)",
      "dest": "/public/$1"
    }
  ]
}
```

### requirements.txt

```
fastapi==0.104.1
pandas==2.1.3
lightgbm==4.1.0
numpy==1.26.2
scikit-learn==1.3.2
```

---

## 🎨 カスタマイズ

### 独自ドメインを設定

1. Vercelダッシュボード → Settings → Domains
2. 独自ドメインを追加（例: `keirin.example.com`）
3. DNSレコードを設定（VercelがDNS情報を表示）
4. 自動でSSL証明書が発行される

### 環境変数を追加

1. Settings → Environment Variables
2. 変数名と値を追加
3. 再デプロイで反映

---

## 🐛 トラブルシューティング

### Q1: デプロイが失敗する

**A:** ログを確認
- Vercelダッシュボード → Deployments → ログを確認
- Python依存関係のエラーが多い → `requirements.txt` を確認

### Q2: APIが動かない

**A:** 関数ログを確認
- Vercel → Functions → api/predict.py → View Logs
- モデルファイルが見つからない → `analysis/model_outputs/` が含まれているか確認

### Q3: モデルファイルが大きすぎる

**A:** `.vercelignore` を作成
```
# 不要なファイルを除外
data/
*.csv
*.log
__pycache__/
```

### Q4: タイムアウトエラー

**A:** Vercel無料プランの制限
- 関数実行時間: 最大10秒
- モデル読み込みを高速化する必要がある場合は、モデルをキャッシュする

---

## 🔄 更新方法

コードを更新したら：

```bash
git add .
git commit -m "Update prediction logic"
git push origin main
```

→ **自動で再デプロイされる**（1-2分）

---

## 📊 無料枠の監視

Vercelダッシュボード → Settings → Usage
- 帯域幅: 100GB/月まで
- 関数実行: 100GB-時間/月まで

**個人利用なら余裕で収まる**

---

## 💡 メリットまとめ

1. **完全無料** - 個人利用なら実質無制限
2. **PC不要** - Vercelがホスティング
3. **デプロイ簡単** - git push するだけ
4. **失敗しない** - Railway/Renderより安定
5. **高速** - グローバルCDN配信
6. **PWA対応** - アプリのように使える
7. **自動SSL** - https:// で安全

---

## 🎉 次のステップ

1. **今すぐデプロイ**: 上記の3ステップを実行
2. **iPhoneでアクセス**: ホーム画面に追加
3. **友達にシェア**: URLを送るだけ

デプロイ完了したら、あなたの専用URLが発行されます！

**例**: `https://100-keirin-xxx.vercel.app`

---

## 📚 関連リンク

- [Vercel公式ドキュメント](https://vercel.com/docs)
- [Python Serverless Functions](https://vercel.com/docs/functions/serverless-functions/runtimes/python)
- [プロジェクトREADME](./README.md)
