import streamlit as st
from embedings import embed, retrieve
from llm import llm
from dotenv import load_dotenv
import os

load_dotenv()
data_path = os.getenv("DATA_PATH", "./data/prompt.txt")

vector_store = embed(data_path)

st.set_page_config(page_title="Chatbot", menu_items={})

if "vector_store" not in st.session_state:
    st.session_state.vector_store = embed(data_path)
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

question = st.chat_input("Ask a question")
if question:
    st.chat_message("user").write(question)
    st.session_state.messages.append({"role": "user", "content": question})
    results = retrieve(st.session_state.vector_store, question)
    response = llm(f"context: {results}\n\nquestion: {question}")
    st.chat_message("assistant").write(response)
    st.session_state.messages.append({"role": "assistant", "content": response})
