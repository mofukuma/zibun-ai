"""
コメダ珈琲店情報RAGチャットアプリケーション

このアプリケーションは、東京都内のコメダ珈琲店の店舗情報をもとに、
ユーザーの質問に答えるチャット形式のアプリケーションです。

主な機能:
- チャット機能: ユーザーが店舗について質問できる会話型インターフェース
- 店舗情報検索: エリア、店名、設備(喫煙/電源/WiFi)に関する質問に回答
- 会話記憶: 過去のやり取りを覚えて、文脈を考慮した回答ができる
- 日本語対応: 完全に日本語での質問・回答に対応
- インターネット検索: Tavily Searchを使用した最新情報の検索（ON/OFF切り替え可能）
- Google Maps位置情報検索: Places APIで店舗詳細を取得、Routes APIで経路情報を取得（ON/OFF切り替え可能）

使用技術:
- Streamlit: Webアプリケーションの画面作成
- LlamaIndex: RAG(情報検索拡張生成)の実現
- Claude Sonnet 4: 高性能な言語モデル
- HuggingFace埋め込みモデル: 多言語対応のテキスト埋め込み
- Tavily Search: インターネット検索機能
- Google Maps Platform: 店舗の詳細情報とルート検索機能

実行方法:
    streamlit run chat_5.py
"""

import os
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from llama_index.core import Document, VectorStoreIndex, Settings
from llama_index.llms.anthropic import Anthropic
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from tavily import TavilyClient
import googlemaps


# 環境変数の読み込み
load_dotenv()


def search_internet(query):
    """
    インターネットから情報を検索する関数

    この関数は以下の処理を行います:
    1. Tavily APIキーを環境変数から取得
    2. 質問をインターネットで検索
    3. 検索結果を整理して返す

    Args:
        query (str): 検索したい質問文

    Returns:
        dict: 検索結果の辞書
            - success (bool): 検索が成功したかどうか
            - results (list): 検索結果のリスト（各要素はタイトル、URL、内容を含む）
            - error (str): エラーメッセージ（エラー時のみ）
    """
    try:
        # 環境変数からTavily APIキーを取得
        api_key = os.getenv("TAVILY_API_KEY")

        # APIキーが設定されているか確認
        if not api_key or api_key == "your_tavily_api_key_here":
            return {
                "success": False,
                "error": "TAVILY_API_KEYが設定されていません",
                "results": []
            }

        # Tavilyクライアントを初期化
        client = TavilyClient(api_key=api_key)

        # インターネット検索を実行
        # max_results: 最大5件の検索結果を取得
        # search_depth: basicモードで検索（高速・低コスト）
        response = client.search(
            query=query,
            max_results=5,
            search_depth="basic"
        )

        # 検索結果を整理
        results = []
        if "results" in response:
            for item in response["results"]:
                results.append({
                    "title": item.get("title", "タイトルなし"),
                    "url": item.get("url", ""),
                    "content": item.get("content", "")
                })

        return {
            "success": True,
            "results": results,
            "error": None
        }

    except Exception as e:
        # エラーが発生した場合
        return {
            "success": False,
            "error": f"検索エラー: {str(e)}",
            "results": []
        }


def check_google_maps_key():
    """
    Google Maps APIキーが設定されているか確認する関数

    この関数は以下の処理を行います:
    1. 環境変数からGoogle Maps APIキーを取得
    2. キーが存在し、デフォルト値でないことを確認

    Returns:
        bool: APIキーが正しく設定されていればTrue、そうでなければFalse
    """
    api_key = os.getenv("GOOGLE_MAPS_API_KEY")

    # APIキーが設定されていない、または初期値のままの場合はFalse
    if not api_key or api_key == "your-google-maps-api-key":
        return False

    return True


def search_places(query):
    """
    Google Maps Places APIを使って店舗を検索する関数

    この関数は以下の処理を行います:
    1. APIキーを環境変数から取得
    2. Google Mapsクライアントを初期化
    3. テキスト検索を実行
    4. 結果を整理して返す

    Args:
        query (str): 検索キーワード（例：「コメダ珈琲 渋谷」）

    Returns:
        dict: 検索結果
            - success (bool): 検索が成功したか
            - results (list): 店舗情報のリスト
                各要素は以下の情報を含む:
                - name (str): 店舗名
                - address (str): 住所
                - phone (str): 電話番号
                - opening_hours (list): 営業時間
                - rating (float): 評価（星の数）
                - user_ratings_total (int): レビュー数
                - place_id (str): 場所ID（ルート検索に使用）
                - location (dict): 緯度経度
            - error (str): エラーメッセージ（失敗時）
    """
    try:
        # 環境変数からGoogle Maps APIキーを取得
        api_key = os.getenv("GOOGLE_MAPS_API_KEY")

        # APIキーが設定されているか確認
        if not api_key or api_key == "your-google-maps-api-key":
            return {
                "success": False,
                "error": "GOOGLE_MAPS_API_KEYが設定されていません",
                "results": []
            }

        # Google Mapsクライアントを初期化
        gmaps = googlemaps.Client(key=api_key)

        # テキスト検索を実行
        # language='ja'で日本語で結果を取得
        places_result = gmaps.places(query=query, language='ja')

        # 検索結果を整理
        results = []
        if places_result.get('status') == 'OK' and 'results' in places_result:
            for place in places_result['results'][:5]:  # 最大5件まで取得
                # place_idを使って詳細情報を取得
                place_id = place.get('place_id')
                if place_id:
                    # 詳細情報取得用のフィールドを指定（料金節約のため必要な情報のみ）
                    place_details = gmaps.place(
                        place_id=place_id,
                        fields=[
                            'name',
                            'formatted_address',
                            'formatted_phone_number',
                            'opening_hours',
                            'rating',
                            'user_ratings_total',
                            'geometry/location'
                        ],
                        language='ja'
                    )

                    if place_details.get('status') == 'OK':
                        detail = place_details['result']

                        # 営業時間の取得
                        opening_hours = []
                        if 'opening_hours' in detail and 'weekday_text' in detail['opening_hours']:
                            opening_hours = detail['opening_hours']['weekday_text']

                        # 店舗情報を整理
                        results.append({
                            'name': detail.get('name', '名前不明'),
                            'address': detail.get('formatted_address', '住所不明'),
                            'phone': detail.get('formatted_phone_number', '電話番号不明'),
                            'opening_hours': opening_hours,
                            'rating': detail.get('rating', 0),
                            'user_ratings_total': detail.get('user_ratings_total', 0),
                            'place_id': place_id,
                            'location': detail.get('geometry', {}).get('location', {})
                        })

        return {
            "success": True,
            "results": results,
            "error": None
        }

    except Exception as e:
        # エラーが発生した場合
        return {
            "success": False,
            "error": f"Google Maps検索エラー: {str(e)}",
            "results": []
        }


def get_directions(origin, destination, mode='walking'):
    """
    Google Maps Directions API (Routes API)を使ってルート情報を取得する関数

    この関数は以下の処理を行います:
    1. APIキーを環境変数から取得
    2. 出発地と目的地でルート検索
    3. 所要時間と距離を計算
    4. わかりやすく整理して返す

    Args:
        origin (str): 出発地（住所や「渋谷駅」など）
        destination (str): 目的地（住所やplace_idなど）
        mode (str): 移動手段（'walking', 'driving', 'transit'のいずれか）
            - walking: 徒歩
            - driving: 車
            - transit: 公共交通機関

    Returns:
        dict: ルート情報
            - success (bool): 検索が成功したか
            - duration (str): 所要時間（例：「15分」）
            - distance (str): 距離（例：「1.2 km」）
            - steps (list): 移動手順の詳細
            - error (str): エラーメッセージ（失敗時）
    """
    try:
        # 環境変数からGoogle Maps APIキーを取得
        api_key = os.getenv("GOOGLE_MAPS_API_KEY")

        # APIキーが設定されているか確認
        if not api_key or api_key == "your-google-maps-api-key":
            return {
                "success": False,
                "error": "GOOGLE_MAPS_API_KEYが設定されていません",
                "duration": "",
                "distance": "",
                "steps": []
            }

        # Google Mapsクライアントを初期化
        gmaps = googlemaps.Client(key=api_key)

        # ルート検索を実行
        # language='ja'で日本語で結果を取得
        directions_result = gmaps.directions(
            origin=origin,
            destination=destination,
            mode=mode,
            language='ja'
        )

        # 結果が存在する場合
        if directions_result and len(directions_result) > 0:
            route = directions_result[0]
            leg = route['legs'][0]

            # 所要時間と距離を取得
            duration = leg.get('duration', {}).get('text', '不明')
            distance = leg.get('distance', {}).get('text', '不明')

            # 移動手順を取得
            steps = []
            if 'steps' in leg:
                for step in leg['steps']:
                    # HTMLタグを削除して純粋なテキストのみ取得
                    instruction = step.get('html_instructions', '')
                    # 簡易的なHTMLタグ削除（正規表現を使わないシンプルな方法）
                    instruction = instruction.replace('<b>', '').replace('</b>', '')
                    instruction = instruction.replace('<div>', '').replace('</div>', '')
                    instruction = instruction.replace('<div style="font-size:0.9em">', '')

                    steps.append({
                        'instruction': instruction,
                        'distance': step.get('distance', {}).get('text', ''),
                        'duration': step.get('duration', {}).get('text', '')
                    })

            return {
                "success": True,
                "duration": duration,
                "distance": distance,
                "steps": steps,
                "error": None
            }
        else:
            return {
                "success": False,
                "error": "ルートが見つかりませんでした",
                "duration": "",
                "distance": "",
                "steps": []
            }

    except Exception as e:
        # エラーが発生した場合
        return {
            "success": False,
            "error": f"ルート検索エラー: {str(e)}",
            "duration": "",
            "distance": "",
            "steps": []
        }


def prepare_data():
    """
    CSVファイルを読み込み、テキスト形式に変換する関数

    この関数は以下の処理を行います:
    1. komeda_list_tokyo.csvを読み込む
    2. 各行のデータを項目名付きのテキスト形式に変換
    3. 変換したテキストのリストを返す

    なぜこの形式にするか:
    - 各行に項目名を含めることで、データの一部だけを取り出しても意味が分かる
    - AIが情報を理解しやすくなる
    - 検索精度が向上する

    Returns:
        list: 変換されたテキストのリスト
    """
    # CSVファイルを読み込む
    df = pd.read_csv("komeda_list_tokyo.csv")

    # テキスト形式に変換
    texts = []
    for _, row in df.iterrows():
        # 各行を以下の形式の文章に変換
        text = f"""【エリア】{row['エリア']}
【店名】{row['店名']}
【喫煙】{row['喫煙の有無']}
【電源】{row['電源の有無']}
【WiFi】{row['WiFiの有無']}"""
        texts.append(text)

    return texts


@st.cache_resource
def create_index():
    """
    RAG用のインデックスを作成する関数

    この関数は以下の処理を行います:
    1. HuggingFaceの埋め込みモデルを初期化
    2. データを準備してLlamaIndexのドキュメント形式に変換
    3. 検索用インデックスを作成

    @st.cache_resource デコレータを使用することで、
    アプリ起動時に一度だけ実行され、結果がキャッシュされます。
    これにより、ユーザーが質問するたびにインデックスを
    再作成する必要がなくなり、パフォーマンスが向上します。

    Returns:
        VectorStoreIndex: 作成されたインデックス
    """
    # HuggingFaceの埋め込みモデルを初期化
    # intfloat/multilingual-e5-base は多言語対応で日本語も扱える
    embed_model = HuggingFaceEmbedding(
        model_name="intfloat/multilingual-e5-base"
    )

    # LlamaIndexの設定に埋め込みモデルを設定
    Settings.embed_model = embed_model

    # データ準備: CSVファイルをテキスト形式に変換
    texts = prepare_data()

    # LlamaIndexのDocument形式に変換
    # Documentオブジェクトは検索可能な単位となる
    documents = [Document(text=text) for text in texts]

    # インデックスを作成
    # VectorStoreIndexは、テキストをベクトル化して検索可能にする
    index = VectorStoreIndex.from_documents(documents)

    return index


def create_chat_engine(index):
    """
    会話機能を持つチャットエンジンを作成する関数

    この関数は以下の処理を行います:
    1. Claude Sonnet 4モデルを設定
    2. 会話機能を持つチャットエンジンを作成
    3. 検索結果を活用した回答を生成できるようにする

    Args:
        index: RAG用のインデックス

    Returns:
        ChatEngine: チャットエンジン
    """
    # Claude Sonnet 4モデルを初期化
    # APIキーは環境変数から自動的に読み込まれる
    llm = Anthropic(
        model="claude-sonnet-4-20250514",
        temperature=0.7,  # 回答の創造性を調整(0.0-1.0)
    )

    # LlamaIndexの設定にLLMを設定
    Settings.llm = llm

    # チャットエンジンを作成
    # as_chat_engine()を使用すると、会話履歴を保持できる
    chat_engine = index.as_chat_engine(
        chat_mode="context",  # コンテキストモードで会話履歴を保持
        verbose=False,  # デバッグ情報を表示しない
    )

    return chat_engine


def main():
    """
    アプリケーションのメイン処理

    この関数は以下の処理を行います:
    1. Streamlitの画面設定
    2. インデックスとチャットエンジンの初期化
    3. 会話履歴の管理
    4. ユーザー入力の受け取り
    5. AIの回答生成と表示
    """
    # ページ設定
    st.set_page_config(
        page_title="カフェめぐりAI",
        page_icon="☕",
        layout="wide",
    )

    # タイトルと説明
    st.title("☕ カフェめぐりAIチャット")
    st.markdown("""
    東京都内のカフェについて質問してください。
    - コメダ珈琲店のエリア、店名、設備(喫煙/電源/WiFi)に関する質問に答えます
    - Google Maps検索をONにすると、スタバ、ドトールなど様々なカフェを検索できます
    - 過去の会話を覚えているので、「さっきの店は?」などの質問もできます
    """)

    # APIキーの確認
    if not os.getenv("ANTHROPIC_API_KEY") or os.getenv("ANTHROPIC_API_KEY") == "your_api_key_here":
        st.error("⚠️ ANTHROPIC_API_KEYが設定されていません。.envファイルを確認してください。")
        st.info("https://console.anthropic.com/settings/keys でAPIキーを取得できます。")
        st.stop()

    # インデックスとチャットエンジンの初期化
    # 初回のみ実行され、結果がキャッシュされる
    with st.spinner("📚 店舗情報を読み込んでいます..."):
        try:
            index = create_index()
            chat_engine = create_chat_engine(index)
        except Exception as e:
            st.error(f"❌ エラーが発生しました: {str(e)}")
            st.info("必要なパッケージがインストールされているか確認してください:\npip install -r requirements.txt")
            st.stop()

    # 会話履歴の初期化
    # セッションステートを使用して、会話履歴を保持する
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # 過去の会話を表示
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # ユーザー入力の受け取り
    if prompt := st.chat_input("店舗について質問してください（例: 池袋エリアの店舗を教えて）"):
        # ユーザーメッセージを会話履歴に追加
        st.session_state.messages.append({"role": "user", "content": prompt})

        # ユーザーメッセージを画面に表示
        with st.chat_message("user"):
            st.markdown(prompt)

        # AIの回答を生成
        with st.chat_message("assistant"):
            with st.spinner("考え中..."):
                try:
                    # 追加コンテキストを保持する変数
                    additional_context = ""
                    sources_list = []

                    # Google Maps検索がONの場合
                    if st.session_state.use_google_maps:
                        # Google Mapsで店舗を検索
                        # ユーザーの質問をそのまま検索（カフェ全般に対応）
                        search_query = f"カフェ {prompt}"
                        places_result = search_places(search_query)

                        # 検索が成功した場合
                        if places_result["success"] and places_result["results"]:
                            # Google Maps検索結果をテキスト形式に整形
                            gmaps_context = "\n\n【Google Maps検索結果】\n"

                            for idx, place in enumerate(places_result["results"], 1):
                                gmaps_context += f"\n{idx}. {place['name']}\n"
                                gmaps_context += f"   住所: {place['address']}\n"
                                gmaps_context += f"   電話: {place['phone']}\n"

                                # 営業時間がある場合
                                if place['opening_hours']:
                                    gmaps_context += f"   営業時間:\n"
                                    for hours in place['opening_hours'][:3]:  # 最初の3日分のみ表示
                                        gmaps_context += f"     - {hours}\n"

                                # 評価がある場合
                                if place['rating'] > 0:
                                    gmaps_context += f"   評価: ★{place['rating']} ({place['user_ratings_total']}件のレビュー)\n"

                                # place_idを保存（ルート検索用）
                                place['place_id_for_directions'] = place['place_id']

                            # 追加コンテキストに追加
                            additional_context += gmaps_context

                            # ルート検索が必要かを判定
                            # 質問に「行き方」「アクセス」「ルート」「道順」などが含まれる場合
                            route_keywords = ['行き方', 'アクセス', 'ルート', '道順', '行く', '行ける', '移動']
                            needs_directions = any(keyword in prompt for keyword in route_keywords)

                            # ルート検索が必要で、最初の店舗がある場合
                            if needs_directions and places_result["results"]:
                                first_place = places_result["results"][0]

                                # 質問から出発地を抽出（簡易的な方法）
                                # 「〇〇駅から」「〇〇から」などのパターンを検出
                                origin = None
                                if 'から' in prompt:
                                    parts = prompt.split('から')
                                    if len(parts) > 0:
                                        origin = parts[0].strip()

                                # 出発地が指定されている場合はルート検索
                                if origin:
                                    directions_result = get_directions(
                                        origin=origin,
                                        destination=first_place['place_id'],
                                        mode='walking'
                                    )

                                    # ルート検索が成功した場合
                                    if directions_result["success"]:
                                        route_context = f"\n\n【ルート情報】\n"
                                        route_context += f"{origin}から{first_place['name']}まで:\n"
                                        route_context += f"- 所要時間: {directions_result['duration']}\n"
                                        route_context += f"- 距離: {directions_result['distance']}\n"
                                        route_context += f"- 移動手段: 徒歩\n"

                                        # 追加コンテキストに追加
                                        additional_context += route_context

                        # 検索が失敗した場合
                        elif not places_result["success"]:
                            st.warning(f"⚠️ Google Maps検索に失敗しました: {places_result['error']}")

                    # インターネット検索がONの場合
                    if st.session_state.use_internet:
                        # インターネット検索を実行
                        search_result = search_internet(prompt)

                        # 検索が成功した場合
                        if search_result["success"] and search_result["results"]:
                            # 検索結果をテキスト形式に整形
                            internet_context = "\n\n【インターネット検索結果】\n"

                            for idx, result in enumerate(search_result["results"], 1):
                                internet_context += f"\n{idx}. {result['title']}\n"
                                internet_context += f"   {result['content'][:200]}...\n"
                                sources_list.append(f"- [{result['title']}]({result['url']})")

                            # 追加コンテキストに追加
                            additional_context += internet_context

                        # 検索が失敗した場合
                        elif not search_result["success"]:
                            st.warning(f"⚠️ インターネット検索に失敗しました: {search_result['error']}")

                    # 追加コンテキストがある場合（Google Maps検索またはインターネット検索が有効）
                    if additional_context:
                        # 元の質問に追加コンテキストを含める
                        enhanced_prompt = f"{prompt}\n{additional_context}"

                        # チャットエンジンに質問を送信（追加コンテキストを含む）
                        response = chat_engine.chat(enhanced_prompt)

                        # 回答を表示
                        st.markdown(response.response)

                        # 参考にした情報源がある場合は表示
                        if sources_list:
                            with st.expander("🔍 参考にした情報源"):
                                st.markdown("\n".join(sources_list))

                            # 回答を会話履歴に追加
                            full_response = response.response + "\n\n**参考情報源:**\n" + "\n".join(sources_list)
                            st.session_state.messages.append(
                                {"role": "assistant", "content": full_response}
                            )
                        else:
                            # 情報源がない場合（Google Maps検索のみ）
                            st.session_state.messages.append(
                                {"role": "assistant", "content": response.response}
                            )

                    # 追加コンテキストがない場合（両方の検索がOFF、または検索結果なし）
                    else:
                        # 通常のチャットエンジンで回答
                        response = chat_engine.chat(prompt)

                        # 回答を表示
                        st.markdown(response.response)

                        # 回答を会話履歴に追加
                        st.session_state.messages.append(
                            {"role": "assistant", "content": response.response}
                        )

                except Exception as e:
                    error_message = f"❌ エラーが発生しました: {str(e)}"
                    st.error(error_message)
                    st.session_state.messages.append(
                        {"role": "assistant", "content": error_message}
                    )

    # サイドバーに情報を表示
    with st.sidebar:
        st.header("⚙️ 設定")

        # インターネット検索のON/OFF切り替え
        # セッションステートで状態を保持
        if "use_internet" not in st.session_state:
            st.session_state.use_internet = False

        use_internet = st.checkbox(
            "🌐 インターネット検索を使用する",
            value=st.session_state.use_internet,
            help="ONにすると、店舗情報に加えてインターネットの最新情報も検索します"
        )
        st.session_state.use_internet = use_internet

        # インターネット検索の説明
        if use_internet:
            st.success("✅ インターネット検索: 有効")
            st.caption("最新情報を含めて回答します")
        else:
            st.info("📚 店舗情報のみで回答します")

        st.divider()

        # Google Maps検索のON/OFF切り替え
        # セッションステートで状態を保持
        if "use_google_maps" not in st.session_state:
            st.session_state.use_google_maps = False

        # APIキーが設定されていない場合は無効化
        has_google_maps_key = check_google_maps_key()

        use_google_maps = st.checkbox(
            "🗺️ Google Maps検索を使用する",
            value=st.session_state.use_google_maps,
            disabled=not has_google_maps_key,
            help="ONにすると、Google Mapsから店舗の詳細情報やルート情報を取得します"
        )

        # APIキーが設定されている場合のみ状態を更新
        if has_google_maps_key:
            st.session_state.use_google_maps = use_google_maps
        else:
            st.session_state.use_google_maps = False

        # Google Maps検索の説明
        if not has_google_maps_key:
            st.warning("⚠️ Google Maps APIキーが未設定")
            st.caption(".envファイルにGOOGLE_MAPS_API_KEYを設定してください")
        elif use_google_maps:
            st.success("✅ Google Maps検索: 有効")
            st.caption("詳細な店舗情報とルート案内を提供します")
        else:
            st.info("🗺️ Google Maps検索: 無効")

        st.divider()

        st.header("📊 アプリ情報")
        st.info(f"登録店舗数: {len(prepare_data())}店舗")
        st.info(f"会話履歴: {len(st.session_state.messages)}メッセージ")

        st.header("💡 質問例")

        # インターネット検索・Google Maps検索のON/OFFで質問例を変える
        if use_google_maps:
            st.markdown("""
            **Google Maps検索ON時の質問例:**
            - 渋谷でおすすめのカフェを探して
            - 新宿駅周辺のスタバを教えて
            - 池袋で評価が高いカフェは?
            - 表参道のおしゃれなカフェは?
            - 東京駅から近いドトールは?
            """)
        elif use_internet:
            st.markdown("""
            **インターネット検索ON時の質問例:**
            - 東京で人気のカフェは?
            - スタバの新メニューは?
            - 今話題のカフェを教えて
            - コメダ珈琲のキャンペーン情報は?
            """)
        else:
            st.markdown("""
            **コメダ珈琲店検索の質問例:**
            - 池袋エリアの店舗を教えて
            - 電源が使えるカフェを探しています
            - 喫煙できる店はある?
            - WiFiが使える新宿の店を教えて
            - 渋谷で充電できるところある?
            """)

        # 会話履歴をクリアするボタン
        if st.button("🗑️ 会話履歴をクリア"):
            st.session_state.messages = []
            st.rerun()


# アプリケーションの起動
if __name__ == "__main__":
    main()
