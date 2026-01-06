from embedings import embed, retrieve
from llm import llm
from dotenv import load_dotenv
import os

load_dotenv()
data_path = os.getenv("DATA_PATH", "./data/prompt.txt")
vector_store = embed(data_path)

print("Enter /exit to quit\n")

while True:
    question = input("\nYou: ")
    if question.lower() in ["/exit"]:
        break
    results = retrieve(vector_store, question)
    messages = f"context: {results}\n\nquestion: {question}"
    output = llm(messages)
    print(f"\nAI: {output}")
