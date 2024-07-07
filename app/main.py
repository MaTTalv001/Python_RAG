import os
import streamlit as st
import pandas as pd
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA
import hashlib
import pickle
from langchain.chains.question_answering import load_qa_chain
from langchain import PromptTemplate

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
    faiss_index = FAISS.from_texts(data['text'].tolist(), embeddings)
    return faiss_index

# キャッシュされたハッシュとインデックスをロード
model_dir = '/app/model'
os.makedirs(model_dir, exist_ok=True)

index_path = os.path.join(model_dir, 'index.pkl')
hash_path = os.path.join(model_dir, 'hash.txt')

if os.path.exists(index_path) and os.path.exists(hash_path):
    with open(index_path, 'rb') as f:
        faiss_index = pickle.load(f)
    with open(hash_path, 'r') as f:
        cached_hash = f.read().strip()
else:
    faiss_index = None
    cached_hash = None

data = load_data()
current_hash = compute_hash(data)

# データが変更された場合のみエンベディングとインデックスを作成
if current_hash != cached_hash:
    faiss_index = create_embeddings_and_index(data)
    with open(index_path, 'wb') as f:
        pickle.dump(faiss_index, f)
    with open(hash_path, 'w') as f:
        f.write(current_hash)

# RetrievalQAのセットアップ
llm = OpenAI(api_key=openai_api_key, model_name="gpt-4o")
qa_chain = load_qa_chain(llm, chain_type="stuff")  # "stuff", "map_reduce", etc.
retriever = faiss_index.as_retriever()

qa = RetrievalQA(combine_documents_chain=qa_chain, retriever=retriever)

st.title('RAG App using LangChain, FAISS, and OpenAI API')
st.write('Enter a query to retrieve and generate answers from the data.')

query = st.text_input('Query:', '')

if query:
    answer = qa.run(query)
    st.write('Answer:', answer)
