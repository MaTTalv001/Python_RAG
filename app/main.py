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
    faiss_index = FAISS.from_texts(data['text'].tolist(), embeddings)
    return faiss_index

# データの読み込みとインデックスの作成
data = load_data()
faiss_index = create_embeddings_and_index(data)

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

st.title('シンプルRAG App with LangChain, FAISS, and OpenAI API')
st.write('クエリを入力してください')

query = st.text_input('Query:', '')

if query:
    result = qa({"query": query})
    st.write('Answer:', result['result'])