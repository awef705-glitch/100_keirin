# 🚀 iPhone単体で使えるようにデプロイする

このガイドでは、PCなしでiPhone単体で使えるように、無料クラウドサービス（Render.com）にアプリをデプロイする方法を説明します。

## 📱 完成後の使い方

デプロイが完了すると：
- **PCは不要！**
- **iPhoneのSafariで直接アクセス**
- **どこでも使える**（Wi-Fi、モバイルデータ通信）
- **完全無料**

## 🌐 デプロイ方法

### オプション1: Render.com（推奨・最も簡単）

#### 前提条件
- GitHubアカウント
- Render.comアカウント（無料）

#### 手順

1. **GitHubにコードをプッシュ**（既に完了している場合はスキップ）

```bash
git add -A
git commit -m "Add Render deployment config"
git push origin main
```

2. **Render.comにアクセス**

https://render.com にアクセスしてアカウント作成（GitHubアカウントで登録が簡単）

3. **新しいWeb Serviceを作成**

- ダッシュボードで「New +」→「Web Service」をクリック
- GitHubリポジトリを接続
- リポジトリ「100_keirin」を選択

4. **設定を入力**

以下の設定を入力：

```
Name: keirin-predictor（任意の名前）
Region: Singapore（またはお好みのリージョン）
Branch: main（またはメインブランチ名）
Runtime: Python 3
Build Command: pip install -r requirements.txt
Start Command: gunicorn --bind 0.0.0.0:$PORT --workers 1 --timeout 120 backend.app:app
```

**Environment Variables:**
```
PYTHON_VERSION = 3.11.0
```

**Instance Type:**
- Free を選択

5. **デプロイ**

「Create Web Service」をクリック

初回デプロイには5-10分かかります。

6. **URLを取得**

デプロイが完了すると、以下のようなURLが表示されます：
```
https://keirin-predictor.onrender.com
```

7. **iPhoneでアクセス！**

このURLをiPhoneのSafariで開くだけ！

### オプション2: Railway.app

#### 手順

1. **Railway.appにアクセス**

https://railway.app にアクセスしてアカウント作成

2. **プロジェクトを作成**

- 「New Project」をクリック
- 「Deploy from GitHub repo」を選択
- リポジトリ「100_keirin」を選択

3. **設定**

Railway.appは自動的に検出しますが、必要に応じて：

```
Start Command: gunicorn --bind 0.0.0.0:$PORT --workers 1 --timeout 120 backend.app:app
```

4. **ドメインを生成**

- Settingsタブで「Generate Domain」をクリック

5. **iPhoneでアクセス！**

生成されたURLをiPhoneのSafariで開く

### オプション3: Fly.io

#### 手順

1. **Fly.ioにアカウント作成**

https://fly.io でアカウント作成（クレジットカード登録が必要だが無料枠あり）

2. **flyctlをインストール**（PCで実行）

```bash
# Mac
brew install flyctl

# Linux
curl -L https://fly.io/install.sh | sh

# Windows
powershell -Command "iwr https://fly.io/install.ps1 -useb | iex"
```

3. **ログイン**

```bash
fly auth login
```

4. **Fly.io設定ファイルを作成**

プロジェクトルートで：

```bash
fly launch
```

質問に答える：
- App name: keirin-predictor（任意）
- Region: Tokyo（または近いリージョン）
- PostgreSQL: No
- Redis: No

5. **デプロイ**

```bash
fly deploy
```

6. **URLを取得**

```bash
fly status
```

表示されたURLをiPhoneで開く

## 🎯 PWAとしてホーム画面に追加

デプロイ後、iPhoneで以下の手順：

1. **Safariでアプリを開く**
2. **共有ボタン（□↑）をタップ**
3. **「ホーム画面に追加」を選択**
4. **「追加」をタップ**

これで、ネイティブアプリのようにホーム画面から起動できます！

## 📊 無料プランの制限

### Render.com
- ✅ 完全無料
- ⚠️ 15分間アクセスがないとスリープ（次回起動時に数秒待つ）
- ✅ 月間750時間まで無料
- ✅ カスタムドメイン可能

### Railway.app
- ✅ 月$5分の無料クレジット
- ✅ スリープなし
- ✅ 高速

### Fly.io
- ✅ 月間無料枠あり
- ✅ スリープなし
- ⚠️ クレジットカード登録必要

## 🔧 トラブルシューティング

### デプロイが失敗する

**原因1: モデルファイルが見つからない**

backend/models/ディレクトリがGitに含まれているか確認：

```bash
git add backend/models/
git commit -m "Add model files"
git push
```

**原因2: 依存関係のエラー**

requirements.txtが正しいか確認：

```bash
pip install -r requirements.txt
```

エラーがなければOK

**原因3: タイムアウト**

初回のモデルロードに時間がかかる場合があります。
Start Commandに `--timeout 120` が含まれているか確認。

### アプリが起動しない

1. **ログを確認**

Render.comの場合：
- ダッシュボード → サービス → Logs

2. **モデルファイルの確認**

ログに「モデルとパラメータをロードしました」と表示されるか確認

3. **再デプロイ**

「Manual Deploy」→「Clear build cache & deploy」を試す

### iPhoneで表示されない

1. **HTTPSか確認**

デプロイされたURLは必ずHTTPSで始まる必要があります

2. **キャッシュをクリア**

Safariの設定でキャッシュをクリア

3. **別のブラウザで試す**

Chromeなど別のブラウザでも試してみる

## 🌟 カスタムドメイン（オプション）

独自ドメインを使いたい場合：

### Render.com

1. ドメインを取得（Namecheap、Google Domainsなど）
2. Render.comのSettings → Custom Domains
3. ドメインを追加
4. DNS設定を更新（CNAMEレコード）

## 🎉 完成！

これで、iPhoneだけで競輪予測アプリが使えます！

**アクセス方法:**
1. iPhoneのSafariでデプロイしたURLを開く
2. ホーム画面に追加
3. いつでもどこでも予測！

## 📝 デプロイ後のメンテナンス

### モデルの更新

新しいデータでモデルを再学習した場合：

```bash
# モデルを再学習
python backend/train_model.py

# Gitにプッシュ
git add backend/models/
git commit -m "Update model"
git push
```

Render.comは自動的に再デプロイされます。

### コードの更新

フロントエンドやバックエンドを更新した場合：

```bash
git add -A
git commit -m "Update app"
git push
```

自動的に再デプロイされます。

## 🔒 セキュリティ

- HTTPS通信で暗号化
- データは外部に送信されません
- モデルはサーバー側で実行
- プライバシー保護

## 💡 ヒント

### スリープ対策（Render.com）

15分間アクセスがないとスリープするため、以下の方法でスリープを防げます：

1. **UptimeRobot**を使う（無料）
   - https://uptimerobot.com
   - 5分ごとにアクセスしてスリープを防ぐ

2. **Cron-jobを使う**（無料）
   - https://cron-job.org
   - 定期的にアクセス

## 🎊 おめでとうございます！

これで、PCなしでiPhone単体で使える競輪予測アプリが完成しました！
