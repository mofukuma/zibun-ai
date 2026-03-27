# =============================================================================
# カフェ巡りAI - 喫茶店情報チャットアプリ
# - 全国の喫茶店の店舗データ(CSV)をRAGで検索し、
#   Claude sonnet がチャット形式で回答するStreamlitアプリ
# - Tavily Searchによるインターネット検索機能付き（ON/OFF切り替え可能）
# =============================================================================

import csv
import os

import streamlit as st
from dotenv import load_dotenv
from llama_index.core import Document, Settings, VectorStoreIndex
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.anthropic import Anthropic
from tavily import TavilyClient

# .envファイルからAPIキーを読み込む
load_dotenv()

# CSVファイルのパス
CSV_PATH = os.path.join(os.path.dirname(__file__), "komeda_list_tokyo.csv")

# Tavilyクライアントを初期化する
tavily_client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))


# -----------------------------------------------------------------------------
# CSVを読み込み、各行を「項目名: 値」形式のテキストに変換してDocumentリストを作る
# -----------------------------------------------------------------------------
def load_csv_as_documents(csv_path: str) -> list[Document]:
    documents = []
    with open(csv_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # 各行を「項目名: 値」の形にして1つのテキストにまとめる
            text = ", ".join(f"{col}: {val}" for col, val in row.items())
            documents.append(Document(text=text))
    return documents


# -----------------------------------------------------------------------------
# LlamaIndexの設定を初期化し、検索用インデックスを作成する
# （Streamlitのキャッシュで、アプリ再読み込み時に再作成しないようにする）
# -----------------------------------------------------------------------------
@st.cache_resource(show_spinner="店舗データを読み込んでいます...")
def build_index():
    # 埋め込みモデルにHuggingFaceのmultilingual-e5-baseを使う
    embed_model = HuggingFaceEmbedding(model_name="intfloat/multilingual-e5-base")
    Settings.embed_model = embed_model

    # AIモデルにClaude sonnet を使う
    llm = Anthropic(model="claude-sonnet-4-5", max_tokens=4096)
    Settings.llm = llm

    # CSVからドキュメントを作り、検索用インデックスを構築する
    documents = load_csv_as_documents(CSV_PATH)
    index = VectorStoreIndex.from_documents(documents)
    return index


# -----------------------------------------------------------------------------
# Tavilyでインターネット検索を行い、検索結果とソース情報を返す
# -----------------------------------------------------------------------------
def search_web(query: str) -> tuple[str, list[dict]]:
    # Tavilyに質問文を送って検索結果を取得する
    response = tavily_client.search(query=query, max_results=5)

    # 検索結果からテキストとソース情報を取り出す
    web_text = ""
    sources = []
    for result in response.get("results", []):
        title = result.get("title", "")
        url = result.get("url", "")
        content = result.get("content", "")
        web_text += f"【{title}】\n{content}\n\n"
        sources.append({"title": title, "url": url})

    return web_text, sources


# -----------------------------------------------------------------------------
# 会話履歴を含めたプロンプトを組み立てて、RAG検索＋AI回答を得る
# web_searchがTrueのとき、インターネット検索の結果も含める
# -----------------------------------------------------------------------------
def ask_with_history(
    index: VectorStoreIndex,
    user_message: str,
    chat_history: list,
    web_search: bool = False,
) -> tuple[str, list[dict]]:
    # 検索エンジンを作成（関連度の高い店舗データを上位10件取得）
    retriever = index.as_retriever(similarity_top_k=10)

    # ユーザーの質問に関連する店舗データを検索する
    nodes = retriever.retrieve(user_message)
    context = "\n".join(node.get_content() for node in nodes)

    # インターネット検索がONのとき、Tavilyで検索して結果を追加する
    web_context = ""
    sources = []
    if web_search:
        web_context, sources = search_web(user_message)

    # 過去の会話をテキストにまとめる
    history_text = ""
    for msg in chat_history:
        role = "ユーザー" if msg["role"] == "user" else "アシスタント"
        history_text += f"{role}: {msg['content']}\n"

    # システムプロンプト：AIの役割と回答ルールを指定する
    if web_search:
        # インターネット検索ONのとき：Webの情報も活用して回答する
        system_prompt = (
            "あなたは全国の喫茶店・カフェに詳しいアシスタントです。\n"
            "以下の店舗データ、インターネット検索結果、会話履歴をもとに、"
            "ユーザーの質問に日本語で丁寧に回答してください。\n"
            "店舗データとインターネット検索結果の両方を活用して、"
            "できるだけ正確で役立つ情報を提供してください。\n"
        )
    else:
        # インターネット検索OFFのとき：CSVデータだけで回答する
        system_prompt = (
            "あなたは全国の喫茶店・カフェに詳しいアシスタントです。\n"
            "以下の店舗データと会話履歴をもとに、ユーザーの質問に日本語で丁寧に回答してください。\n"
            "店舗データにない情報については「データにありません」と正直に答えてください。\n"
        )

    # 最終的にAIに渡すプロンプトを組み立てる
    full_prompt = f"{system_prompt}\n【店舗データ】\n{context}\n\n"
    if web_search and web_context:
        full_prompt += f"【インターネット検索結果】\n{web_context}\n"
    full_prompt += (
        f"【会話履歴】\n{history_text}\nユーザー: {user_message}\nアシスタント:"
    )

    # Claude sonnet に問い合わせて回答を得る
    llm = Settings.llm
    response = llm.complete(full_prompt)
    return str(response), sources


# =============================================================================
# Streamlit アプリ本体
# =============================================================================
def main():
    st.set_page_config(page_title="カフェ巡りAI", page_icon="☕")
    st.title("☕ カフェ巡りAI")
    st.caption("全国の喫茶店・カフェについて質問できます")

    # サイドバーにインターネット検索のON/OFFスイッチを置く
    with st.sidebar:
        st.header("設定")
        web_search = st.toggle("インターネット検索", value=False)
        if web_search:
            st.info("ONにすると、Webの情報も含めて回答します")
        else:
            st.info("OFFのときは、店舗データのみで回答します")

    # 検索用インデックスを構築する（初回のみ実行される）
    index = build_index()

    # 会話履歴をセッションに保存する（ページ再読み込みまで保持）
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # これまでの会話を画面に表示する
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            # 情報源がある場合はリンクも再表示する
            if message.get("sources"):
                with st.expander("参考にした情報源"):
                    for src in message["sources"]:
                        st.markdown(f"- [{src['title']}]({src['url']})")

    # ユーザーの入力を受け取る
    if prompt := st.chat_input(
        "質問を入力してください（例：渋谷でおすすめの喫茶店は？）"
    ):
        # ユーザーのメッセージを会話履歴に追加して画面に表示する
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # AIに質問して回答を得る
        with st.chat_message("assistant"):
            with st.spinner("回答を生成中..."):
                response, sources = ask_with_history(
                    index, prompt, st.session_state.messages[:-1], web_search
                )
            st.markdown(response)

            # インターネット検索ONで情報源がある場合、リンクを表示する
            if sources:
                with st.expander("参考にした情報源"):
                    for src in sources:
                        st.markdown(f"- [{src['title']}]({src['url']})")

        # AIの回答を会話履歴に追加する（情報源も一緒に保存する）
        st.session_state.messages.append(
            {"role": "assistant", "content": response, "sources": sources}
        )


if __name__ == "__main__":
    main()
