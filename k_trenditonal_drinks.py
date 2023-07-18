import streamlit as st
st.set_page_config(layout="wide")
import numpy as np
import pandas as pd
from tqdm import tqdm
tqdm.pandas()
import os
from pyparsing import empty
import time
import openai
openai.api_key = st.secrets.OPENAI_TOKEN
from supabase import create_client
import pickle
from openai.embeddings_utils import (
    get_embedding,
    distances_from_embeddings,
    indices_of_nearest_neighbors_from_distances,
)
import matplotlib.pyplot as plt
import plotly.express as px
from scipy import spatial
from sklearn.decomposition import PCA


@st.cache_resource
def init_connection():
    url = st.secrets["SUPABASE_URL"]
    key = st.secrets["SUPABASE_KEY"]
    return create_client(url, key)

supabase_client = init_connection()

EMBEDDING_MODEL = "text-embedding-ada-002"

empty1, con1, empty2 = st.columns([0.3, 1.0, 0.3])
empty3, con2, empty4 = st.columns([0.3, 1.0, 0.3])
e_7, image_c, text_c, e_8 = st.columns([0.3, 0.5, 1.0, 0.3])


with empty1:
    st.empty()

with con1:
    st.image("./f_image/title_03.png")

with empty2:
    st.empty()


#STEP 2) 데이터 로드
@st.cache_resource
def load_data():
    feature_df = pd.read_csv("./data/features_with_name.csv", encoding="utf-8-sig")
    main_df = pd.read_csv("./data/main_total_no_features.csv", encoding="utf-8-sig")
    ingredient_df = pd.read_csv("./data/ingredient_total_id.csv", encoding="utf-8-sig")
    embedding_df = pd.read_csv("./data/embedding_w_emotion.csv", encoding="utf-8-sig")
    food_df = pd.read_csv("./data/food_preprocessed.csv", encoding="utf-8-sig")
    emoji_df = pd.read_csv("./data/f_total_emoji.csv", encoding="utf-8-sig")
    return feature_df, main_df, ingredient_df, embedding_df, emoji_df, food_df

feature_df, main_df, ingredient_df, embedding_df, emoji_df, food_df = load_data()

@st.cache_resource
def embedding_c():
    embeddings = [np.array(eval(embedding)).astype(float) for embedding in embedding_df["embeddings"].values]
    stacked_embeddings = np.vstack(embeddings)

    return stacked_embeddings

stacked_embeddings = embedding_c()

#STEP 3) 캐시 불러오고 임베딩 저장하기
embedding_cache_path = "./data/recommendations_embeddings_cache.pkl"

try:
    embedding_cache = pd.read_pickle(embedding_cache_path)
except FileNotFoundError:
    embedding_cache = {}
with open(embedding_cache_path, "wb") as embedding_cache_file:
    pickle.dump(embedding_cache, embedding_cache_file)

def embedding_from_string(
    string: str,
    model: str = EMBEDDING_MODEL,
    embedding_cache=embedding_cache
) -> list:
    """Return embedding of given string, using a cache to avoid recomputing."""
    if (string, model) not in embedding_cache.keys():
        embedding_cache[(string, model)] = get_embedding(string, model)
        with open(embedding_cache_path, "wb") as embedding_cache_file:
            pickle.dump(embedding_cache, embedding_cache_file)
    return embedding_cache[(string, model)]


def get_idx(
        input_query: str,
        alcohol_min: int,
        alcohol_max: int,
):
    # 입력받은 쿼리 임베딩
    input_query_embedding = embedding_from_string(input_query, model=EMBEDDING_MODEL)

    # 임베딩 벡터간 거리 계산 (open ai 라이브러리 사용 - embeddings_utils.py)
    ## 도수 제한

    alcohol_limited_list = main_df.loc[
        (main_df["alcohol"] >= alcohol_min) & (main_df["alcohol"] <= alcohol_max)].index.tolist()
    source_embeddings = stacked_embeddings[alcohol_limited_list]

    distances = distances_from_embeddings(input_query_embedding, source_embeddings, distance_metric="cosine")

    # 가까운 벡터 인덱스 구하기 (open ai 라이브러리 사용 - embeddings_utils.py)
    indices_of_nearest_neighbors = indices_of_nearest_neighbors_from_distances(distances)

    # k개의 가까운 벡터 인덱스 출력
    k_nearest_neighbors = 1
    k_counter = 0

    idx_list = []
    for i in indices_of_nearest_neighbors:
        # stop after printing out k articles
        if k_counter >= k_nearest_neighbors:
            break
        k_counter += 1

        idx_list.append(i)

    return idx_list, alcohol_limited_list


def get_result(
        emotion: str,
        situation: str,
        food: str,
        alcohol_min: int,
        alcohol_max: int
):
    emoji_keywords = {}

    for i in range(len(emoji_df)):
        emoji_keywords[emoji_df["sample"][i]] = emoji_df["k_keywords"][i]

    if emotion[0] in emoji_keywords:
        k_emotion = emoji_keywords[emotion[0]]

    else:
        k_emotion = emotion

    if situation[0] in emoji_keywords:
        k_situation = emoji_keywords[situation[0]]

    else:
        k_situation = situation

    if food[0] in emoji_keywords:
        k_food = emoji_keywords[food[0]]

    else:
        k_food = food


    # 출력 query 수정
    input_query = f"{k_food} {k_food} {k_food} {k_emotion} {k_situation}"

    idx_list, alcohol_limited_list = get_idx(input_query, alcohol_min, alcohol_max)

    if not idx_list or not alcohol_limited_list:
        return [], []

    for i in idx_list:
        name_id_list.append(main_df.loc[alcohol_limited_list].iloc[i]["name_id"])

    return name_id_list, alcohol_limited_list

@st.cache_resource
def get_embedding(text, model="text-embedding-ada-002"):
    text = text.replace("\n", " ")
    return openai.Embedding.create(input=[text], model=model)['data'][0]['embedding']

def image_name(name_id):
    directory = "./f_image/"
    matching_files = [file for file in os.listdir(directory) if name_id in file]
    if len(matching_files) > 0:
        filename = os.path.join(directory, matching_files[0])
        return filename  # 변수 filename을 반환합니다.
    else:
        return None

input_container = None


with st.container():  # 외부 컨테이너
    empty7, pro, empty9 = st.columns([0.3, 1.0, 0.3])
    empty1, image_c, text_c, empty2 = st.columns([0.3, 0.3, 0.5, 0.3])
    empty3, con2, empty4 = st.columns([0.3, 1.0, 0.3])

    with con2:
        container = st.empty()
        form = container.form("my_form", clear_on_submit=True)  # 내부 컨테이너의 폼 생성

        with form:
            col_s, col_e, col_f = st.columns(3)  # 내부 컨테이너의 컬럼 생성

            with col_s:
                emotion = st.text_input('감성 (1개)', value="", placeholder="❤️")

            with col_e:
                situation = st.text_input("상황 (1개)", value="", placeholder="✈️")

            with col_f:
                food = st.text_input('입맛 (1개)', value="", placeholder="🍇 또는 🍰 등")

            alcohol_min, alcohol_max = st.select_slider(
                '도수를 선택해주세요',
                options=[0, 20, 40, 60, 70],
                value=(0, 70)
            )
            submitted = st.form_submit_button("Submit")

    name_id_list = []  # name_id_list 변수를 초기화합니다.
    if submitted:
        if not situation:
            st.error("어떤 상황에서 술을 마시고 싶은지 입력해주세요")
        elif not emotion:
            st.error("어떤 기분일 때 마시고 싶은지 입력해주세요")
        else:
            with pro:
                with st.spinner('당신을 위한 전통주를 찾고 있습니다...🔍'):
                    name_id_list, alcohol_limited_list = get_result(situation = situation, emotion = emotion, food = food, alcohol_min = alcohol_min, alcohol_max = alcohol_max)
                    time.sleep(5)
                    if not name_id_list or not alcohol_limited_list:
                        st.warning("검색 결과가 없습니다.")
                    else:
                        filtered_main_df = main_df.loc[alcohol_limited_list].copy()
                        filtered_main_df['name_id'] = filtered_main_df['name_id'].astype(str)
                        filtered_main_df.set_index('name_id', inplace=True)

                        container.empty()

                        with image_c:
                            name_id = name_id_list[0]
                            if name_id in filtered_main_df.index:
                                input_query = f"{situation} 상황에서 {emotion} 기분으로 {food}와 함께 마시기 좋은 술"
                                result_query = ''.join([input_query[:-1], "전통주"])
                                filtered_df = main_df[main_df["name_id"].str.contains(name_id)]
                                if not filtered_df.empty:
                                    loaded_image = image_name(name_id)
                                    st.image(loaded_image, use_column_width='auto')
                                else:
                                    st.write("해당하는 이미지가 없습니다.")

                        with text_c:
                            st.subheader(f"{emotion} {situation} {food}")
                            name_id = name_id_list[0]
                            if name_id in filtered_main_df.index:
                                input_query = f"{situation} 상황에서 {emotion} 기분으로 {food}와 함께 마시기 좋은 술"
                                result_query = ''.join([input_query[:-1], "전통주"])
                                st.write(f"🔸 전통주 이름 : {filtered_main_df.loc[name_id, 'name']}")
                                st.write(f"🔸 도수 : {filtered_main_df.loc[name_id, 'alcohol']}")
                                st.write("🔸 특징 :")
                                features = feature_df.loc[feature_df['name_id'] == name_id]['features'].tolist()
                                for feature in features:
                                    st.write(f"{feature}")
                                with_food = food_df.loc[food_df['name_id'] == name_id]['food'].values[0]
                                st.write(f"🔸 어울리는 음식 : {with_food}")
                                ingredients = ", ".join(
                                    ingredient_df.loc[ingredient_df['name_id'] == name_id]['ingredients'])
                                st.write(f"🔸 재료 : {ingredients}")
                                if st.button('다시하기'):
                                    st.experimental_rerun()


                            else:
                                st.warning(f"전통주 이름: {name_id} 에 해당하는 정보가 없습니다.")







