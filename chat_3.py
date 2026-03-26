import os
import streamlit as st
from dotenv import load_dotenv
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.llms.anthropic import Anthropic
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

load_dotenv()

# ページ設定
st.set_page_config(page_title="RAGチャットアプリ", page_icon="💬", layout="centered")

#--- モデル初期化関数 ---
@st.cache_resource(show_spinner=False)
def initialize_models():
   """モデル設定の初期化（サーバー起動中1回だけ実行される）"""
   api_key = os.getenv("ANTHROPIC_API_KEY")
   if not api_key:
       return False

   Settings.llm = Anthropic(model="claude-sonnet-4-5", api_key=api_key)
   Settings.embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")
   return True

@st.cache_resource(show_spinner=False)
def build_index():
   """インデックス作成（ファイル変更がない限り再実行されない）"""
   if not os.path.exists("knowledge.txt"):
       return None

   documents = SimpleDirectoryReader(input_files=["knowledge.txt"]).load_data()
   return VectorStoreIndex.from_documents(documents)

# --- 初期化処理 ---
if not initialize_models():
   st.error("APIキーが設定されていません")
   st.stop()

index = build_index()
if index is None:
   st.error("knowledge.txtが見つかりません")
   st.stop()

# --- セッションステートの初期化 ---
if "chat_engine" not in st.session_state:
   # システムプロンプトを定義
   SYSTEM_PROMPT = """
   あなたは知識ベースの情報を参考にして質問に答えるアシスタントです。
   知識ベースにキャラクター情報が含まれている場合でも、そのキャラクターになりきるのではなく、
   その情報について客観的に説明してください。
   """

   # contextモード: 検索結果(context)をプロンプトに含めて会話するモード
   st.session_state.chat_engine = index.as_chat_engine(
       chat_mode="context",
       system_prompt=SYSTEM_PROMPT,
       similarity_top_k=3
   )

if "messages" not in st.session_state:
   st.session_state.messages = []

# --- UI部分 ---
st.title("💬 RAGチャットアプリ")

# サイドバー
with st.sidebar:
   if st.button("🗑️ 会話履歴をクリア"):
       st.session_state.messages = []
       st.session_state.chat_engine.reset()
       st.rerun()

# 履歴表示
for msg in st.session_state.messages:
   with st.chat_message(msg["role"]):
       st.markdown(msg["content"])

# 入力処理
if prompt := st.chat_input("質問を入力してください"):
   st.session_state.messages.append({"role": "user", "content": prompt})
   with st.chat_message("user"):
       st.markdown(prompt)

   with st.chat_message("assistant"):
       # 履歴の文字列結合処理は不要。chatメソッドを呼ぶだけ。
       response = st.session_state.chat_engine.chat(prompt)
       st.markdown(str(response))

   st.session_state.messages.append({"role": "assistant", "content": str(response)})

