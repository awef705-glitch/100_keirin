# Claude Code - Claude Pro 恒久設定ガイド

このドキュメントは、VSCodeでClaude CodeをClaude Pro（Web版）で恒久的に使用するための設定手順を記載しています。

## 問題の背景

Claude Code APIとClaude Pro（Web版）の認証が混在すると、VSCodeを再起動するたびにAPIに戻ってしまう問題が発生します。

## 解決策

VSCodeの**グローバル設定**と**ワークスペース設定**の両方で `claudeCode.authProvider` を `"web"` に設定します。

---

## 設定手順

### 1. 既存のAPI認証からログアウト（任意）

ターミナルで以下を実行：

```bash
npx @anthropic-ai/claude-code logout
```

### 2. グローバル設定の変更

**ファイルパス**: `C:\Users\<ユーザー名>\AppData\Roaming\Code\User\settings.json`

以下のように設定：

```json
{
  "claudeCode.authProvider": "web",
  "claudeCode.interface": "terminal"
}
```

**PowerShellスクリプトで自動更新**:

```powershell
$settingsPath = "$env:APPDATA\Code\User\settings.json"
$content = Get-Content $settingsPath -Raw
$content = $content -replace '"claudeCode\.authProvider": "anthropic"', '"claudeCode.authProvider": "web"'
Set-Content -Path $settingsPath -Value $content -NoNewline
Write-Host "Successfully updated claudeCode.authProvider to 'web'"
```

### 3. ワークスペース設定の変更（プロジェクトごと）

**ファイルパス**: `<プロジェクトルート>\.vscode\settings.json`

以下のように設定：

```json
{
  "claudeCode.useTerminal": true,
  "claudeCode.authProvider": "web",
  "claudeCode.interface": "terminal"
}
```

**PowerShellスクリプトで自動更新**:

```powershell
$settingsPath = ".\.vscode\settings.json"

$content = @"
{
  "claudeCode.useTerminal": true,
  "claudeCode.authProvider": "web",
  "claudeCode.interface": "terminal"
}
"@

Set-Content -Path $settingsPath -Value $content
Write-Host "Successfully updated workspace settings"
```

---

## 設定の確認

### VSCodeで確認

1. VSCodeを開く
2. `Ctrl + ,` で設定を開く
3. 検索バーに `claudeCode.authProvider` を入力
4. 値が `web` になっていることを確認

### ファイルで確認

```powershell
# グローバル設定
cat $env:APPDATA\Code\User\settings.json | Select-String "claudeCode"

# ワークスペース設定
cat .\.vscode\settings.json
```

---

## 重要な注意事項

### 設定の優先順位

VSCodeの設定には以下の優先順位があります：

1. **ワークスペース設定** (`.vscode/settings.json`) - 最優先
2. **グローバル設定** (`AppData/Roaming/Code/User/settings.json`)
3. **デフォルト設定**

**ワークスペース設定が最優先**されるため、プロジェクトごとに`.vscode/settings.json`を作成している場合は、そちらも必ず`"web"`に設定してください。

### 全てのフォルダで適用するには

全てのプロジェクトで自動的にClaude Proを使用したい場合：

1. **グローバル設定**で `"claudeCode.authProvider": "web"` を設定
2. **ワークスペース設定がある場合**は、そちらも`"web"`に設定（ワークスペース設定が優先されるため）

---

## トラブルシューティング

### VSCode再起動後にAPIに戻ってしまう

**原因**: ワークスペース設定で`"anthropic"`が設定されている可能性があります。

**対処法**:
```powershell
# ワークスペース設定を確認
cat .\.vscode\settings.json

# "anthropic"が含まれていたら"web"に変更
```

### "ログインしてください"と表示される

**対処法**:
1. VSCodeでClaude Code拡張機能を開く
2. "Sign in with Claude Pro"を選択
3. ブラウザでログイン
4. VSCodeに戻る

### どの設定が適用されているか確認する方法

VSCode内で:
1. `Ctrl + Shift + P` でコマンドパレットを開く
2. "Preferences: Open Settings (JSON)"を入力
3. **ユーザー設定**と**ワークスペース設定**の両方を確認

---

## 参考：API版の設定（過去の設定）

過去にAPI版を使用していた場合の設定：

```json
{
  "claudeCode.authProvider": "anthropic",
  "claudeCode.interface": "terminal"
}
```

この設定を`"web"`に変更することで、Claude Proを使用できます。

---

## 補足：既存のWSL連携設定との関係

過去にWSL連携でClaude Codeを設定していた場合（`claude-wsl`スクリプトなど）、それらの設定は**認証方法には影響しません**。

WSL連携はあくまで**コマンド実行環境**の設定であり、認証プロバイダーは別途VSCode拡張機能の設定で管理されます。

---

## 設定完了後の確認

1. VSCodeを完全に終了する（全てのウィンドウを閉じる）
2. VSCodeを再起動
3. Claude Code拡張機能を開く
4. "Claude Pro"でログインされていることを確認
5. ターミナルで`claude`コマンドを実行してテスト

---

## まとめ

以下の2つのファイルで `"claudeCode.authProvider": "web"` を設定してください：

1. **グローバル設定**: `C:\Users\<user>\AppData\Roaming\Code\User\settings.json`
2. **ワークスペース設定**: `<プロジェクト>\.vscode\settings.json`

これで、VSCodeを再起動してもClaude Proが恒久的に使用されます。
