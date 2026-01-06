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


def llm(user_input):
    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": "You are a helpful chat assistant. Keep responses to 1-2 sentences.",
            },
            {"role": "user", "content": user_input},
        ],
    )
    return response.choices[0].message.content
