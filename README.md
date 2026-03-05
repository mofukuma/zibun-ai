# Claude AIチャットプログラム

ブラウザ上でClaude AIと日本語で対話できるウェブアプリケーションです。

## 機能

- ブラウザ上でClaude AIとリアルタイムで対話
- 会話履歴の自動保存と表示
- シンプルで使いやすいチャット画面
- レスポンシブデザイン対応

## 必要な環境

- Python 3.7以上
- インターネット接続(Claude APIとの通信に必要)

## セットアップ方法

### 1. venv環境を作成(推奨)

プロジェクト専用のPython環境を作成します:

```bash
# venv環境を作成
python3 -m venv venv

# venv環境を有効化
# Mac/Linuxの場合:
source venv/bin/activate

# Windowsの場合:
venv\Scripts\activate
```

### 2. 必要なライブラリをインストール

```bash
# requirements.txtからインストール(推奨)
pip install -r requirements.txt

# または個別にインストール
pip install anthropic flask
```

### 3. APIキーの取得と設定

1. [Anthropic公式サイト](https://console.anthropic.com/)でアカウントを作成
2. APIキーを取得
3. 環境変数に設定:

**Mac/Linuxの場合:**
```bash
export ANTHROPIC_API_KEY='your-api-key-here'
```

**Windowsの場合(コマンドプロンプト):**
```cmd
set ANTHROPIC_API_KEY=your-api-key-here
```

**Windowsの場合(PowerShell):**
```powershell
$env:ANTHROPIC_API_KEY="your-api-key-here"
```

## 使い方

### 1. venv環境を有効化

毎回プログラムを実行する前に、venv環境を有効化してください:

```bash
# Mac/Linuxの場合:
source venv/bin/activate

# Windowsの場合:
venv\Scripts\activate
```

### 2. プログラムを起動

```bash
python chat_1.py
```

### 3. ブラウザでアクセス

プログラムを起動したら、以下のURLにアクセス:

```
http://localhost:5000
```

### 4. チャット開始

- 画面下部の入力欄にメッセージを入力
- 「送信」ボタンをクリック、またはEnterキーを押す
- AIからの返答が表示されます

### 5. 終了方法

ターミナルで `Ctrl+C` を押すとプログラムが終了します。

venv環境を終了する場合:
```bash
deactivate
```

## ファイル構成

```
chat_ai/
├── chat_1.py           # メインプログラム(すべての機能を含む1ファイル)
├── 計画書.md            # 開発計画書
├── README.md           # このファイル
├── requirements.txt    # 必要なライブラリのリスト
└── venv/               # Python仮想環境(作成後)
```

## プログラムの特徴

### 1つのファイルに全機能を実装
- chat_1.pyに全ての機能が含まれています
- Python、HTML、CSS、JavaScriptすべてが1ファイルで完結

### 日本語コメント付き
- ファイルの先頭に説明
- 各関数の前に機能説明
- 複雑な処理の前に詳しいコメント

### わかりやすい構造
- 初心者でも理解しやすいシンプルな設計
- 専門用語を避けた日本語コメント

## トラブルシューティング

### APIキーのエラーが出る場合
- 環境変数が正しく設定されているか確認
- APIキーが有効か確認

### ポート5000が使用中の場合
chat_1.pyの最後の行を変更:
```python
app.run(debug=True, host='0.0.0.0', port=8000)  # 5000を別の番号に変更
```

### ライブラリが見つからないエラー
```bash
pip install --upgrade anthropic flask
```

## 注意事項

- APIキーは外部に公開しないでください
- API利用には料金がかかる場合があります
- インターネット接続が必要です
- 会話履歴はブラウザを閉じると消えます

## ライセンス

このプログラムは学習・個人利用目的で自由に使用できます。
