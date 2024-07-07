import os
import streamlit as st
import pandas as pd
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
import hashlib
from langchain.prompts import PromptTemplate

# OpenAI APIキーの設定
openai_api_key = os.getenv("OPENAI_API_KEY")

# データの読み込み
def load_data():
    return pd.read_csv('/app/data/data.csv')

# データのハッシュを計算
def compute_hash(data):
    return hashlib.md5(pd.util.hash_pandas_object(data, index=True).values).hexdigest()

# エンベディングとインデックスの作成
def create_embeddings_and_index(data):
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    texts = data['text'].tolist()
    faiss_index = FAISS.from_texts(texts, embeddings)
    return faiss_index, embeddings

# キャッシュされたハッシュとインデックスをロード
model_dir = '/app/model'
os.makedirs(model_dir, exist_ok=True)

index_path = os.path.join(model_dir, 'faiss_index')
hash_path = os.path.join(model_dir, 'hash.txt')

data = load_data()
current_hash = compute_hash(data)

# インデックスの管理
# 既存のインポート文に以下を追加
from langchain.vectorstores.utils import DistanceStrategy

# インデックスの管理部分を以下のように修正

# インデックスの管理
if not os.path.exists(index_path) or not os.path.exists(hash_path):
    # フォルダまたはインデックスデータがない場合
    print("Creating new index...")
    faiss_index, embeddings = create_embeddings_and_index(data)
    faiss_index.save_local(index_path)
    with open(hash_path, 'w') as f:
        f.write(current_hash)
else:
    with open(hash_path, 'r') as f:
        stored_hash = f.read().strip()
    
    if current_hash != stored_hash:
        # データファイルが更新されている場合
        print("Data updated. Recreating index...")
        faiss_index, embeddings = create_embeddings_and_index(data)
        faiss_index.save_local(index_path)
        with open(hash_path, 'w') as f:
            f.write(current_hash)
    else:
        # データファイルに更新がない場合
        print("Loading existing index...")
        embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
        faiss_index = FAISS.load_local(
            index_path,
            embeddings,
            allow_dangerous_deserialization=True,
            distance_strategy=DistanceStrategy.COSINE
        )

# RetrievalQAのセットアップ
llm = ChatOpenAI(api_key=openai_api_key, model_name="gpt-4o")
retriever = faiss_index.as_retriever()

# プロンプトテンプレートの作成
prompt_template = """Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer.

{context}

Question: {question}
Answer:"""
PROMPT = PromptTemplate(
    template=prompt_template, input_variables=["context", "question"]
)

# RetrievalQA チェーンの構築
qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True,
    chain_type_kwargs={"prompt": PROMPT}
)

st.title('RAG App using LangChain, FAISS, and OpenAI API')
st.write('Enter a query to retrieve and generate answers from the data.')

query = st.text_input('Query:', '')

if query:
    result = qa({"query": query})
    st.write('Answer:', result['result'])