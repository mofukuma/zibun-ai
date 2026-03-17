"""
Claude AIチャットプログラム
ブラウザ上でClaude AIと対話できるウェブアプリケーション

必要なライブラリ:
  pip install anthropic flask

使い方:
  1. APIキーを環境変数ANTHROPIC_API_KEYに設定する
  2. python chat_1.py を実行
  3. ブラウザで http://localhost:5000 にアクセス
"""

# 必要なライブラリの読み込み
from flask import Flask, render_template_string, request, jsonify, session
from anthropic import Anthropic
import os
from datetime import datetime

# Flaskアプリケーションの初期化
app = Flask(__name__)
# セッション管理のための秘密鍵を設定(会話履歴を保持するために必要)
app.secret_key = os.urandom(24)

# Claude APIクライアントの初期化
# 環境変数からAPIキーを取得する
# APIキーは ANTHROPIC_API_KEY という名前で環境変数に設定しておく必要がある
anthropic_client = Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

# HTMLテンプレート(画面のデザイン)
# 1つのファイルにまとめるため、HTML/CSS/JavaScriptを文字列として定義
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="ja">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Claude AIチャット</title>
    <style>
        /* 画面全体のスタイル設定 */
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            height: 100vh;
            display: flex;
            justify-content: center;
            align-items: center;
            padding: 20px;
        }

        /* チャット画面全体のコンテナ */
        .chat-container {
            background: white;
            border-radius: 15px;
            box-shadow: 0 10px 40px rgba(0,0,0,0.2);
            width: 100%;
            max-width: 800px;
            height: 90vh;
            display: flex;
            flex-direction: column;
            overflow: hidden;
        }

        /* ヘッダー部分(タイトル表示エリア) */
        .chat-header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            text-align: center;
            font-size: 24px;
            font-weight: bold;
        }

        /* メッセージ表示エリア(会話履歴が表示される場所) */
        .chat-messages {
            flex: 1;
            overflow-y: auto;
            padding: 20px;
            background: #f5f5f5;
        }

        /* 個々のメッセージのスタイル */
        .message {
            margin-bottom: 15px;
            padding: 12px 16px;
            border-radius: 10px;
            max-width: 80%;
            word-wrap: break-word;
            animation: fadeIn 0.3s;
        }

        /* メッセージが表示される時のアニメーション */
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(10px); }
            to { opacity: 1; transform: translateY(0); }
        }

        /* ユーザーのメッセージのスタイル */
        .user-message {
            background: #667eea;
            color: white;
            margin-left: auto;
            text-align: right;
        }

        /* AIの返答メッセージのスタイル */
        .ai-message {
            background: white;
            color: #333;
            margin-right: auto;
            border: 1px solid #e0e0e0;
        }

        /* 送信者名の表示スタイル */
        .message-sender {
            font-size: 12px;
            font-weight: bold;
            margin-bottom: 5px;
            opacity: 0.8;
        }

        /* メッセージ本文のスタイル */
        .message-content {
            line-height: 1.5;
            white-space: pre-wrap;
        }

        /* 入力エリア全体(メッセージ入力欄と送信ボタン) */
        .chat-input-area {
            padding: 20px;
            background: white;
            border-top: 1px solid #e0e0e0;
            display: flex;
            gap: 10px;
        }

        /* メッセージ入力欄のスタイル */
        .chat-input {
            flex: 1;
            padding: 12px;
            border: 2px solid #e0e0e0;
            border-radius: 25px;
            font-size: 14px;
            outline: none;
            transition: border-color 0.3s;
        }

        /* 入力欄にフォーカスした時のスタイル */
        .chat-input:focus {
            border-color: #667eea;
        }

        /* 送信ボタンのスタイル */
        .send-button {
            padding: 12px 30px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            border-radius: 25px;
            font-size: 14px;
            font-weight: bold;
            cursor: pointer;
            transition: transform 0.2s, box-shadow 0.2s;
        }

        /* 送信ボタンにカーソルを乗せた時 */
        .send-button:hover {
            transform: translateY(-2px);
            box-shadow: 0 5px 15px rgba(102, 126, 234, 0.4);
        }

        /* 送信ボタンが無効の時(送信中など) */
        .send-button:disabled {
            background: #ccc;
            cursor: not-allowed;
            transform: none;
        }

        /* ローディング表示(AIが考え中の時) */
        .loading {
            display: none;
            text-align: center;
            padding: 10px;
            color: #667eea;
            font-style: italic;
        }

        .loading.active {
            display: block;
        }
    </style>
</head>
<body>
    <div class="chat-container">
        <!-- ヘッダー部分 -->
        <div class="chat-header">
            Claude AIチャット
        </div>

        <!-- メッセージ表示エリア -->
        <div class="chat-messages" id="chatMessages">
            <!-- ここに会話履歴が動的に追加されます -->
        </div>

        <!-- ローディング表示 -->
        <div class="loading" id="loading">AIが考え中...</div>

        <!-- 入力エリア -->
        <div class="chat-input-area">
            <input
                type="text"
                class="chat-input"
                id="messageInput"
                placeholder="メッセージを入力してください..."
                autocomplete="off"
            >
            <button class="send-button" id="sendButton" onclick="sendMessage()">送信</button>
        </div>
    </div>

    <script>
        // ページ読み込み時に会話履歴を取得して表示する
        window.onload = function() {
            loadChatHistory();
        };

        // Enterキーでメッセージ送信できるようにする
        document.getElementById('messageInput').addEventListener('keypress', function(e) {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                sendMessage();
            }
        });

        // メッセージを送信する関数
        async function sendMessage() {
            const input = document.getElementById('messageInput');
            const message = input.value.trim();

            // 空のメッセージは送信しない
            if (!message) return;

            // 送信ボタンを無効化(連続送信を防ぐ)
            const sendButton = document.getElementById('sendButton');
            sendButton.disabled = true;

            // ユーザーのメッセージを画面に表示
            addMessageToChat('user', message);

            // 入力欄をクリア
            input.value = '';

            // ローディング表示を表示
            document.getElementById('loading').classList.add('active');

            try {
                // サーバーにメッセージを送信
                const response = await fetch('/send_message', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({ message: message })
                });

                const data = await response.json();

                // エラーチェック
                if (data.error) {
                    alert('エラーが発生しました: ' + data.error);
                    return;
                }

                // AIの返答を画面に表示
                addMessageToChat('assistant', data.response);

            } catch (error) {
                alert('通信エラーが発生しました: ' + error.message);
            } finally {
                // ローディング表示を非表示
                document.getElementById('loading').classList.remove('active');
                // 送信ボタンを有効化
                sendButton.disabled = false;
                // 入力欄にフォーカスを戻す
                input.focus();
            }
        }

        // メッセージを画面に追加する関数
        function addMessageToChat(role, content) {
            const messagesDiv = document.getElementById('chatMessages');
            const messageDiv = document.createElement('div');

            // ユーザーかAIかで表示スタイルを変える
            if (role === 'user') {
                messageDiv.className = 'message user-message';
                messageDiv.innerHTML = `
                    <div class="message-sender">あなた</div>
                    <div class="message-content">${escapeHtml(content)}</div>
                `;
            } else {
                messageDiv.className = 'message ai-message';
                messageDiv.innerHTML = `
                    <div class="message-sender">Claude AI</div>
                    <div class="message-content">${escapeHtml(content)}</div>
                `;
            }

            messagesDiv.appendChild(messageDiv);

            // 最新のメッセージまで自動スクロール
            messagesDiv.scrollTop = messagesDiv.scrollHeight;
        }

        // 会話履歴を読み込んで表示する関数
        async function loadChatHistory() {
            try {
                const response = await fetch('/get_history');
                const data = await response.json();

                // 会話履歴を順番に表示
                data.history.forEach(msg => {
                    addMessageToChat(msg.role, msg.content);
                });
            } catch (error) {
                console.error('履歴の読み込みに失敗しました:', error);
            }
        }

        // HTMLエスケープ関数(セキュリティ対策)
        function escapeHtml(text) {
            const div = document.createElement('div');
            div.textContent = text;
            return div.innerHTML;
        }
    </script>
</body>
</html>
"""


# トップページを表示する関数
@app.route('/')
def index():
    """
    ブラウザでアクセスした時に最初に表示されるページ
    会話履歴を初期化してチャット画面を表示する
    """
    # セッションに会話履歴がない場合は初期化
    if 'messages' not in session:
        session['messages'] = []

    # HTMLテンプレートを返す
    return render_template_string(HTML_TEMPLATE)


# 会話履歴を取得する関数
@app.route('/get_history')
def get_history():
    """
    保存されている会話履歴を返す
    ページを更新した時などに、過去の会話を表示するために使用
    """
    messages = session.get('messages', [])
    return jsonify({'history': messages})


# メッセージを送信してAIから返答を受け取る関数
@app.route('/send_message', methods=['POST'])
def send_message():
    """
    ユーザーからのメッセージを受け取り、Claude APIに送信して返答を取得する

    処理の流れ:
    1. ユーザーのメッセージを受け取る
    2. 会話履歴に追加
    3. Claude APIに送信
    4. AIの返答を受け取る
    5. 会話履歴に追加
    6. ブラウザに返答を返す
    """
    try:
        # リクエストからメッセージを取得
        data = request.get_json()
        user_message = data.get('message', '')

        if not user_message:
            return jsonify({'error': 'メッセージが空です'}), 400

        # セッションから会話履歴を取得
        if 'messages' not in session:
            session['messages'] = []

        # ユーザーのメッセージを会話履歴に追加
        session['messages'].append({
            'role': 'user',
            'content': user_message
        })

        # Claude APIにメッセージを送信して返答を取得
        ai_response = get_claude_response(session['messages'])

        # AIの返答を会話履歴に追加
        session['messages'].append({
            'role': 'assistant',
            'content': ai_response
        })

        # セッションを保存
        session.modified = True

        # 返答を返す
        return jsonify({
            'response': ai_response,
            'success': True
        })

    except Exception as e:
        # エラーが発生した場合
        print(f"エラーが発生しました: {str(e)}")
        return jsonify({
            'error': f'エラーが発生しました: {str(e)}',
            'success': False
        }), 500


# Claude APIと通信して返答を取得する関数
def get_claude_response(messages):
    """
    Claude APIにメッセージを送信して、AIからの返答を取得する

    引数:
        messages: 会話履歴のリスト

    戻り値:
        AIからの返答文字列
    """
    try:
        # Claude APIにリクエストを送信
        # model: 使用するAIモデル(claude-sonnet-4-5が最新の高性能モデル)
        # max_tokens: 返答の最大長
        # messages: これまでの会話履歴
        response = anthropic_client.messages.create(
            model="claude-sonnet-4-5",
            max_tokens=8096,
            messages=messages
        )

        # 返答からテキスト部分を取り出して返す
        return response.content[0].text

    except Exception as e:
        # エラーが発生した場合はエラーメッセージを返す
        return f"申し訳ございません。エラーが発生しました: {str(e)}"


# プログラムのメイン部分(ここから実行が始まる)
if __name__ == '__main__':
    """
    プログラムを直接実行した時にこの部分が実行される
    Flaskサーバーを起動してブラウザからのアクセスを待ち受ける
    """
    print("=" * 50)
    print("Claude AIチャットプログラムを起動します")
    print("=" * 50)
    print()

    # APIキーが設定されているかチェック
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("警告: ANTHROPIC_API_KEY環境変数が設定されていません")
        print("以下のコマンドでAPIキーを設定してください:")
        print("export ANTHROPIC_API_KEY='your-api-key-here'")
        print()

    print("サーバーを起動しています...")
    print("ブラウザで以下のURLにアクセスしてください:")
    print("http://localhost:5000")
    print()
    print("終了するには Ctrl+C を押してください")
    print("=" * 50)
    print()

    # Flaskサーバーを起動
    # debug=True: 開発モード(エラーが詳しく表示される)
    # host='0.0.0.0': すべてのネットワークインターフェースからアクセス可能
    # port=5000: ポート番号5000で起動
    app.run(debug=True, host='0.0.0.0', port=5000)
