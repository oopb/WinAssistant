import time
import os

from dotenv import load_dotenv, find_dotenv

_ = load_dotenv(find_dotenv())

from langchain_community.document_loaders import TextLoader
from langchain_community.embeddings.huggingface import HuggingFaceBgeEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import CharacterTextSplitter, RecursiveCharacterTextSplitter

# from langchain.text_splitter import NLTKTextSplitter

time_list = []

t = time.time()
# 导入文本
file_path = os.getenv("DOCUMENT_PATH_TXT")
loader = TextLoader(file_path, encoding="utf-8")
data = loader.load()

# 文本切分
text_splitter = RecursiveCharacterTextSplitter(chunk_size=128, chunk_overlap=16)
# text_splitter = NLTKTextSplitter()
split_docs = text_splitter.split_documents(data)
print(split_docs)
model_name = "BAAI/bge-large-zh"
model_kwargs = {'device': 'cpu'}
encode_kwargs = {'normalize_embeddings': True}
embeddings = HuggingFaceBgeEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs
)

# 初始化加载器 构建本地知识向量库
db = Chroma.from_documents(split_docs, embeddings, persist_directory="./chroma/document_txt")
# 持久化
db.persist()

# 打印时间##
time_list.append(time.time() - t)
print(time.time() - t)
