from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")
base_url = os.getenv("OPENAI_BASE_URL")
model = os.getenv("MODEL_NAME", "llama3.1")

client = OpenAI(
    base_url=base_url,
    api_key=api_key,
)
print("model:", model)
print("base_url:", base_url)
print("model:", model)
print("base_url:", base_url)
response = client.chat.completions.create(
    model=model, messages=[{"role": "user", "content": "Hello, how are you?"}]
)

print(response.choices[0].message.content)
