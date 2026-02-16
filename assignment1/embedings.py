from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os


def embed(data):
    load_dotenv()
    embedding_model = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
    chroma_path = os.getenv("CHROMA_PATH", "./chroma_db")
    embeddings = HuggingFaceEmbeddings(model_name=embedding_model)

    if os.path.exists(chroma_path):
        return Chroma(persist_directory=chroma_path, embedding_function=embeddings)
    loader = TextLoader(data, encoding="utf-8")
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(docs)
    vector_store = Chroma.from_documents(
        chunks, embeddings, persist_directory=chroma_path
    )

    return vector_store


def retrieve(vector_store, query):
    return vector_store.similarity_search(query)
