from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
base_url = os.getenv("OPENAI_BASE_URL")
model = os.getenv("MODEL_NAME", "llama3.1")
embedding_model = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")

embeddings = HuggingFaceEmbeddings(model_name=embedding_model)

#
# llm = ChatOpenAI(base_url=base_url, api_key=api_key, model=model)
# response = llm.invoke("Hello, how are you?")
# print(response.content)


loader = TextLoader("./data/prompt.txt")
docs = loader.load()

splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_documents(docs)

vector_store = Chroma.from_documents(chunks, embeddings)

results = vector_store.similarity_search("What are the grading criteria?")

# for doc in results:
#     print(doc.page_content)
#     print("---")
#


context = "\n\n".join([doc.page_content for doc in results])
question = "What are the grading criteria?"

prompt = f"""Use the following context to answer the question.

Context:
{context}

Question: {question}

Answer:"""

llm = ChatOpenAI(base_url=base_url, api_key=api_key, model=model)
response = llm.invoke(prompt)
print(response.content)
