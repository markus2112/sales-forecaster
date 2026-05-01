import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
api_key = os.environ.get("OPENAI_API_KEY")

print(f"API Key loaded: {api_key[:10]}..." if api_key else "No API Key")

try:
    client = OpenAI(api_key=api_key)
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": "Say test"}],
        max_tokens=5
    )
    print("Success:", response.choices[0].message.content)
except Exception as e:
    print("Error:", type(e).__name__, str(e))
