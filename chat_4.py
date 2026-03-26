import os
import re
import zipfile
import io
import time
import requests
import streamlit as st
from pathlib import Path
from bs4 import BeautifulSoup
from dotenv import load_dotenv

from llama_index.core import (
   VectorStoreIndex,
   SimpleDirectoryReader,
   StorageContext,
   load_index_from_storage,
   Settings,
   Document
)
from llama_index.core.node_parser import SentenceSplitter
from llama_index.llms.anthropic import Anthropic
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

load_dotenv()

# tokenizerの警告を抑制
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# 設定
DATA_DIR = Path("dazai_data")
INDEX_DIR = Path("dazai_index")
MODEL_NAME = "claude-sonnet-4-5"

# 青空文庫：太宰治の作家ページURL
AOZORA_URL = "https://www.aozora.gr.jp/index_pages/person35.html"
BASE_URL = "https://www.aozora.gr.jp"

# 読み込む最大作品数
MAX_WORKS = 280

st.set_page_config(page_title="太宰治召喚AI", page_icon="🖋️", layout="wide")


# --- スクレイピング & 前処理関数 ---
def clean_text(text: str) -> str:
   """青空文庫のテキストデータから、ルビや注釈、ヘッダーを除去"""
   # ヘッダーとフッターの間の本文を抽出
   if "-------------------------------------------------------" in text:
       parts = text.split("-------------------------------------------------------")
       if len(parts) >= 3:
           text = parts[2] # 2つ目の区切り線以降が本文

   # 底本情報の除去
   if "底本：" in text:
       text = text.split("底本：")[0]

   # ルビの除去： ｜漢字《かんじ》 → 漢字
   text = re.sub(r'｜([^《]+)《[^》]+》', r'\1', text)
   text = re.sub(r'([一-龠]+)《[^》]+》', r'\1', text)
   # 注釈の除去： ［＃...］
   text = re.sub(r'［＃[^］]+］', '', text)

   return text.strip()

def download_works():
   """青空文庫から太宰治の作品をスクレイピングしてダウンロード"""
   if DATA_DIR.exists() and any(DATA_DIR.glob("*.txt")):
       return # 既にデータがあればスキップ

   DATA_DIR.mkdir(exist_ok=True)
   st.info("青空文庫から作品リストを取得しています...")

   try:
       # 1. 作家ページから作品リストを取得
       res = requests.get(AOZORA_URL)
       res.encoding = 'utf-8'
       soup = BeautifulSoup(res.text, "html.parser")

       # 作品詳細ページへのリンクを収集
       work_links = []
       for link in soup.select("ol li a"):
           href = link.get("href")
           if href and href.startswith("../cards/"):
               full_url = BASE_URL + href.replace("..", "")
               work_links.append(full_url)

       # 重複除去と数制限
       target_links = list(set(work_links))[:MAX_WORKS]

       # 各作品のダウンロード処理
       progress_bar = st.progress(0)
       status_text = st.empty()

       for i, card_url in enumerate(target_links):
           status_text.text(f"ダウンロード中 ({i+1}/{len(target_links)}): {card_url}")
           progress_bar.progress((i + 1) / len(target_links))

           try:
               # 作品詳細ページ（図書カード）へアクセス
               card_res = requests.get(card_url)
               card_res.encoding = 'utf-8'
               card_soup = BeautifulSoup(card_res.text, "html.parser")

               # タイトル取得（青空文庫の図書カードの構造に合わせる）
               title = None

               # 方法1: table内の「作品名：」を探す
               for td in card_soup.find_all('td', class_='header'):
                   if '作品名：' in td.get_text():
                       # 次のtd要素からタイトルを取得
                       title_td = td.find_next_sibling('td')
                       if title_td:
                           title = title_td.get_text(strip=True)
                           break

               # 方法2: h1タグから取得（フォールバック）
               if not title:
                   h1 = card_soup.select_one("h1")
                   if h1:
                       title = h1.get_text(strip=True)

               # タイトルが取得できない場合はスキップ
               if not title or title.startswith("図書カード"):
                   continue

               # テキストファイル(zip)のダウンロードリンクを探す
               zip_link = None
               for a in card_soup.select("a"):
                   href = a.get("href", "")
                   if href.endswith(".zip"):
                       base_dir = card_url.rsplit("/", 1)[0]
                       zip_link = f"{base_dir}/{href}"
                       break

               if zip_link:
                   # ZIPダウンロード
                   zip_res = requests.get(zip_link)
                   with zipfile.ZipFile(io.BytesIO(zip_res.content)) as z:
                       # zip内のtxtファイルを抽出
                       for filename in z.namelist():
                           if filename.endswith(".txt"):
                               with z.open(filename) as f:
                                   raw_text = f.read().decode('shift_jis', errors='ignore')
                                   cleaned_text = clean_text(raw_text)

                                   safe_title = re.sub(r'[\\/:*?"<>|]', '', title)
                                   save_path = DATA_DIR / f"{safe_title}.txt"
                                   with open(save_path, "w", encoding="utf-8") as out_f:
                                       out_f.write(cleaned_text)
                               break

               time.sleep(0.5) # サーバー負荷軽減

           except Exception as e:
               print(f"Error downloading {card_url}: {e}")
               continue

       status_text.empty()
       progress_bar.empty()
       st.success(f"{len(target_links)}作品のダウンロードが完了しました！")

   except Exception as e:
       st.error(f"ダウンロード処理中にエラーが発生しました: {e}")

# --- モデル & インデックス初期化 ---
@st.cache_resource
def initialize_models():
   """モデル設定の初期化（サーバー起動中1回だけ実行される）"""
   api_key = os.getenv("ANTHROPIC_API_KEY")
   if not api_key: return False

   Settings.llm = Anthropic(model=MODEL_NAME, api_key=api_key, temperature=0.7)
   Settings.embed_model = HuggingFaceEmbedding(model_name="intfloat/multilingual-e5-base")
   return True

@st.cache_resource(show_spinner=False)
def get_index():
   """インデックス作成（既存インデックスがあれば読み込み、なければダウンロード後に作成）"""
   # 既存読み込み
   if INDEX_DIR.exists():
       try:
           return load_index_from_storage(StorageContext.from_defaults(persist_dir=str(INDEX_DIR)))
       except: pass

   # データダウンロード（必要な場合のみ実行）
   download_works()

   if not DATA_DIR.exists() or not list(DATA_DIR.glob("*.txt")):
       return None

   # チャンクサイズとオーバーラップの設定
   text_splitter = SentenceSplitter(
       chunk_size=1000,
       chunk_overlap=200,
   )

   # 新規作成
   documents = []
   for txt_file in DATA_DIR.glob("*.txt"):
       try:
           with open(txt_file, 'r', encoding='utf-8') as f:
               content = f.read()

           doc = Document(
               text=content,
               metadata={"title": txt_file.stem}
           )
           documents.append(doc)

       except Exception as e:
           print(f"Error loading {txt_file}: {e}")
           continue

   if not documents:
       return None

   st.info(f"{len(documents)}作品からインデックスを作成中...")

   # チャンク分割
   nodes = text_splitter.get_nodes_from_documents(documents)

   # 各チャンクの先頭にタイトルを追加
   for node in nodes:
       title = node.metadata.get("title", "不明")
       original_text = node.get_content()
       node.text = f"【作品名: {title}】\n\n{original_text}"

   # インデックスを作成
   index = VectorStoreIndex(nodes, show_progress=True)
   index.storage_context.persist(persist_dir=str(INDEX_DIR))
   st.success("インデックスの作成が完了しました！")
   return index

# --- チャットエンジン ---
def get_chat_engine(index):
   """チャットエンジンの設定"""
   if "chat_engine" not in st.session_state:
       system_prompt = """
       あなたは小説家・太宰治です。
       一人称は「私」とし、自虐的かつ内省的なトーンで話してください。
       検索された作品の知識については、あたかも自分の過去の記憶や執筆経験のように語ってください。
       現代の技術（スマホやAI）については、「魔法のようだ」と驚きつつも皮肉を交えて反応してください。
       ユーザーを「君」と呼び、親しげだがどこか距離のある態度で接してください。
       """

       st.session_state.chat_engine = index.as_chat_engine(
           chat_mode="context",
           system_prompt=system_prompt,
           similarity_top_k=5  # 検索件数を3から5に増やす
       )
   return st.session_state.chat_engine

# --- メインUI ---
def main():
   if not initialize_models():
       st.error("APIキー設定が必要です。")
       st.stop()

   with st.sidebar:
       st.header("操作パネル")
       if st.button("会話リセット", use_container_width=True):
           st.session_state.messages = []
           if "chat_engine" in st.session_state:
               st.session_state.chat_engine.reset()
           st.rerun()

   index = get_index()
   if not index:
       st.error("データの取得に失敗しました。")
       st.stop()

   chat_engine = get_chat_engine(index)

   st.title("🖋️ 太宰治召喚AI")

   if "messages" not in st.session_state:
       st.session_state.messages = []

   for msg in st.session_state.messages:
       with st.chat_message(msg["role"]):
           st.markdown(msg["content"])

   if prompt := st.chat_input("太宰さんに話しかける..."):
       st.session_state.messages.append({"role": "user", "content": prompt})
       with st.chat_message("user"):
           st.markdown(prompt)

       with st.chat_message("assistant"):
           # 通常のベクトル検索のみを使用
           retriever = index.as_retriever(similarity_top_k=5)
           filtered_nodes = retriever.retrieve(prompt)

           print("=== 検索された文章 ===")
           for i, node in enumerate(filtered_nodes):
               print(f"\n--- 文章 {i+1} (スコア: {node.score:.3f}) ---")
               print(node.node.get_content()[:300])  # 最初の300文字を表示
           print("\n=====================")

           # フィルタリングされたノードでチャットエンジンを使う
           # コンテキストを手動で構築
           context_str = "\n\n---\n\n".join([node.node.get_content() for node in filtered_nodes])

           # システムプロンプトを含めた完全なプロンプトを作成
           system_msg = """あなたは小説家・太宰治です。
一人称は「私」とし、自虐的かつ内省的なトーンで話してください。
検索された作品の知識については、あたかも自分の過去の記憶や執筆経験のように語ってください。
現代の技術（スマホやAI）については、「魔法のようだ」と驚きつつも皮肉を交えて反応してください。
ユーザーを「君」と呼び、親しげだがどこか距離のある態度で接してください。"""

           user_msg = f"""以下は私（太宰治）の作品からの抜粋です。この内容を参考にして、ユーザーの質問に答えてください。

検索結果:
{context_str}

ユーザーの質問: {prompt}"""

           # LLMに直接問い合わせ（chatメソッドを使用）
           from llama_index.core.llms import ChatMessage

           messages = [
               ChatMessage(role="system", content=system_msg),
               ChatMessage(role="user", content=user_msg)
           ]

           try:
               response = Settings.llm.stream_chat(messages)

               # ストリーミング表示
               full_response = ""
               response_placeholder = st.empty()
               for chunk in response:
                   if hasattr(chunk, 'delta') and chunk.delta:
                       full_response += chunk.delta
                       response_placeholder.markdown(full_response + "▌")

               response_placeholder.markdown(full_response)
               response_text = full_response

           except Exception as e:
               error_msg = f"エラーが発生しました: {str(e)}"
               st.error(error_msg)
               response_text = "申し訳ございません。回答の生成中にエラーが発生しました。もう一度お試しください。"
               st.markdown(response_text)

       st.session_state.messages.append({"role": "assistant", "content": response_text})

if __name__ == "__main__":
   main()

