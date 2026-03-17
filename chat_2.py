"""
Claude AIチャットプログラム(Streamlit版)
Streamlitを使ってブラウザ上でClaude AIと対話できるアプリケーション

特徴:
  - Streamlitのチャット部品を使ったシンプルな画面
  - 会話の記憶機能(AIが過去のやり取りを覚えている)
  - APIキーは.envファイルで安全に管理

使い方:
  1. .envファイルにAPIキーを設定する
  2. streamlit run chat_2.py を実行
  3. ブラウザが自動で開き、チャット画面が表示される
"""

# 必要な部品の読み込み
import streamlit as st
from anthropic import Anthropic
from dotenv import load_dotenv
import os

# .envファイルからAPIキーを読み込む
load_dotenv()

# Claude APIクライアントの準備
# .envファイルに書いたANTHROPIC_API_KEYを使ってAIと通信する準備をする
client = Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))


def get_claude_response(messages):
    """
    Claude APIにメッセージを送信して、AIからの返答を取得する関数

    引数:
        messages: これまでの会話履歴のリスト

    戻り値:
        AIからの返答テキスト
    """
    try:
        # Claude APIにリクエストを送信
        # model: 使用するAIモデル
        # max_tokens: 返答の最大文字数
        # messages: これまでの会話履歴(これを送ることでAIが会話の流れを覚える)
        response = client.messages.create(
            model="claude-sonnet-4-5",
            max_tokens=8096,
            messages=messages,
        )

        # 返答からテキスト部分を取り出して返す
        return response.content[0].text

    except Exception as e:
        # エラーが発生した場合はエラーメッセージを返す
        return f"エラーが発生しました: {str(e)}"


# ページの設定(タイトルやアイコンなど)
st.set_page_config(
    page_title="Claude AIチャット",
    page_icon="🤖",
)

# ページタイトルの表示
st.title("Claude AIチャット")

# 会話履歴の初期化
# session_stateはStreamlitの一時的な記憶領域で、画面が更新されてもデータが残る
# 初回アクセス時だけ空のリストを作成する
if "messages" not in st.session_state:
    st.session_state.messages = []

# 過去のメッセージを画面に表示する処理
# 保存されている会話履歴を1つずつ取り出して、吹き出しとして表示する
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# ユーザーの入力を受け取る処理
# 画面の下にメッセージ入力欄を表示し、入力があった場合に処理を行う
if prompt := st.chat_input("メッセージを入力してください"):

    # ユーザーのメッセージを会話履歴に追加
    st.session_state.messages.append({"role": "user", "content": prompt})

    # ユーザーのメッセージを吹き出しで表示
    with st.chat_message("user"):
        st.markdown(prompt)

    # AIの返答を取得して表示する処理
    with st.chat_message("assistant"):
        # 「考え中...」の表示を出しながらAIからの返答を待つ
        with st.spinner("考え中..."):
            # 会話履歴をまとめてClaude APIに送信(これが「記憶」の仕組み)
            response = get_claude_response(st.session_state.messages)

        # AIの返答を画面に表示
        st.markdown(response)

    # AIの返答を会話履歴に追加(次の質問の時にAIがこの返答も覚えている)
    st.session_state.messages.append({"role": "assistant", "content": response})
