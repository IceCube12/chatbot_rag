import os
import shutil
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

CHROMA_PATH = "chroma_db"
DATA_PATH = r"data\PIA.pdf"

response = input("clear DB(C) or update DB(U): ")


if response.upper() == "C":

    shutil.rmtree("chroma_db_nccn")

    loaders = [PyPDFLoader(DATA_PATH)]

    docs = []

    for file in loaders:
        docs.extend(file.load())

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs = text_splitter.split_documents(docs)
    embedding_function = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2", model_kwargs={'device': 'cpu'})

    vectorstore = Chroma.from_documents(docs, embedding_function, persist_directory=CHROMA_PATH)

    print("DB cleared!")
    print(f"new data loaded: {vectorstore._collection.count()}")

elif response.upper() == "U":
    loaders = [PyPDFLoader(DATA_PATH)]

    docs = []

    for file in loaders:
        docs.extend(file.load())

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs = text_splitter.split_documents(docs)
    embedding_function = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2", model_kwargs={'device': 'cpu'})

    vectorstore = Chroma.from_documents(docs, embedding_function, persist_directory=CHROMA_PATH)

    print(f"updated DB: {vectorstore._collection.count()}")

else:
    print("Not valid!! Pls. enter C to clear DB or U to update DB")

