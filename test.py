import os
from openai import OpenAI

client = OpenAI(
    api_key=os.environ["TOKENHARBOR_API_KEY"],
    base_url="https://tokenharbor.ai/v1",
)

res = client.chat.completions.create(
    model="deepseek-v4.1-flash:free",
    messages=[{"role": "user", "content": "Why is the sky blue?"}],
)

print(res.choices[0].message.content)