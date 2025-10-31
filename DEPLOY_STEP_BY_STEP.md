# 🚀 Render.comで今すぐデプロイする手順

## ステップ1: Render.comにアクセス

1. **ブラウザで以下にアクセス**
   ```
   https://render.com
   ```

2. **アカウント作成**
   - 右上の「Get Started」または「Sign Up」をクリック
   - 「Sign up with GitHub」を選択（一番簡単）
   - GitHubでログイン・認証

## ステップ2: 新しいWeb Serviceを作成

1. **ダッシュボードに移動**
   - ログイン後、自動的にダッシュボードが表示される

2. **「New +」ボタンをクリック**
   - 画面上部にある青いボタン

3. **「Web Service」を選択**
   - メニューから「Web Service」をクリック

## ステップ3: GitHubリポジトリを接続

1. **「Connect a repository」画面が表示される**

2. **GitHubリポジトリを選択**
   - リストに「100_keirin」が表示されていればそれをクリック

   **表示されない場合:**
   - 「Configure account」をクリック
   - GitHub認証画面で「100_keirin」リポジトリへのアクセスを許可
   - 戻って再度リストを確認

3. **「Connect」ボタンをクリック**

## ステップ4: サービスを設定

以下の画面が表示されます。各項目を入力：

### 📝 基本設定

```
Name（名前）:
→ keirin-predictor
  （好きな名前でOK、URLに使われます）

Region（地域）:
→ Singapore
  （日本に近いので速い、他の地域でもOK）

Branch（ブランチ）:
→ main
  （または claude/read-repository-contents-011CUT8cpEivmfcT2RoxTot7）

Runtime（実行環境）:
→ Python 3
  （自動検出されるはず）
```

### ⚙️ ビルド設定

```
Build Command（ビルドコマンド）:
→ pip install --upgrade pip && pip install -r requirements.txt

Start Command（起動コマンド）:
→ gunicorn --bind 0.0.0.0:$PORT --workers 1 --timeout 120 backend.app:app
```

### 🔧 環境変数（Environment Variables）

**「Add Environment Variable」をクリック**

```
Key（キー）: PYTHON_VERSION
Value（値）: 3.11.0
```

### 💰 プラン選択

```
Instance Type:
→ Free
  （無料プランを選択）
```

## ステップ5: デプロイ開始

1. **画面下部の「Create Web Service」ボタンをクリック**
   - 青い大きなボタン

2. **デプロイが自動的に開始**
   - ログが表示される
   - 5-10分かかります（初回）

## ステップ6: デプロイ完了を確認

### ✅ 成功の確認

ログの最後に以下のようなメッセージが表示されれば成功：

```
==> Build succeeded 🎉
==> Deploying...
==> Starting service with 'gunicorn...'
==> Your service is live 🎉
   https://keirin-predictor.onrender.com
```

### 画面上部にURLが表示

```
https://keirin-predictor-xxxx.onrender.com
```

このURLをコピー！

## ステップ7: iPhoneでアクセス

1. **iPhoneのSafariを開く**

2. **コピーしたURLを貼り付けて開く**
   ```
   https://keirin-predictor-xxxx.onrender.com
   ```

3. **アプリが表示される！**

4. **ホーム画面に追加（推奨）**
   - 画面下部の「共有」ボタン（□に↑のマーク）をタップ
   - 下にスクロールして「ホーム画面に追加」をタップ
   - 「追加」をタップ

## 🎉 完成！

これでiPhone単体で競輪予測アプリが使えます！

---

## 🐛 エラーが出た場合

### エラー: Build failed

**ログを確認:**
- 画面左側の「Logs」タブをクリック
- エラーメッセージを確認

**よくあるエラーと対処法:**

#### 1. Python version エラー
```
Environment Variables に以下を追加:
Key: PYTHON_VERSION
Value: 3.11.0
```

#### 2. モデルファイルが見つからない
```
GitHubに backend/models/ がプッシュされているか確認
```

#### 3. タイムアウトエラー
```
Start Command に --timeout 120 が含まれているか確認
```

### 再デプロイする方法

1. 画面右上の「Manual Deploy」をクリック
2. 「Clear build cache & deploy」を選択
3. 数分待つ

---

## 📱 今後の使い方

### URLをブックマーク

```
https://keirin-predictor-xxxx.onrender.com
```

このURLをiPhoneのSafariでブックマークするか、ホーム画面に追加すれば、いつでもアクセスできます。

### 初回アクセス時の注意

**無料プランの場合:**
- 15分間アクセスがないとスリープ状態になります
- 次回アクセス時に数秒〜30秒ほど待つ必要があります
- 「Loading...」と表示されたら、そのまま待ってください

---

## 💡 ヒント

### スリープを防ぐ（オプション）

**UptimeRobot を使う（無料）:**

1. https://uptimerobot.com にアクセス
2. アカウント作成
3. 「Add New Monitor」をクリック
4. Monitor Type: HTTP(s)
5. Friendly Name: Keirin App
6. URL: あなたのRender.comのURL
7. Monitoring Interval: 5 minutes
8. 「Create Monitor」をクリック

これで5分ごとにアクセスしてスリープを防ぎます。

---

## 🔍 トラブルシューティング

### Q: GitHubリポジトリが表示されない

**A:** Render.comにGitHubリポジトリへのアクセス権限を与える

1. Render.comで「Configure account」をクリック
2. GitHubの認証画面が開く
3. 「100_keirin」リポジトリにチェックを入れる
4. 「Save」をクリック

### Q: ビルドに時間がかかる

**A:** 初回は5-10分かかります。気長に待ちましょう。

### Q: デプロイ後にアクセスできない

**A:** 以下を確認：

1. URLが正しいか
2. HTTPSで始まっているか（HTTPは使えません）
3. デプロイが完了しているか（ログに「Your service is live」と表示）
4. 少し待ってから再度アクセス

### Q: アプリが表示されない

**A:** ブラウザのキャッシュをクリア

1. Safariの設定を開く
2. Safari → 詳細 → Webサイトデータ
3. 「全Webサイトデータを削除」

---

## 📞 サポート

それでも解決しない場合は、Render.comのログをコピーして、GitHubのIssuesに貼り付けてください。
