import time
from typing import Optional

import openai
from openai import OpenAI

from phishing_generator import DEFAULT_MODEL, client, PHISHING_SCHEMA


def call_llm(prompt: str, model: Optional[str] = None, max_retries: int = 5) -> str:
    model = model or DEFAULT_MODEL

    for attempt in range(1, max_retries + 1):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_schema", "json_schema": PHISHING_SCHEMA},
                temperature=0.8,
            )
            return response.choices[0].message.content

        except openai.RateLimitError as e:
            # 429: превышен request rate. Ждём столько, сколько просит сервер.
            retry_after = e.response.headers.get("retry-after") if e.response else None
            wait = float(retry_after) if retry_after else min(2 ** attempt, 30)
            print(f"[429] Rate limit. Ждём {wait:.1f}с (попытка {attempt}/{max_retries})...")
            time.sleep(wait)

        except openai.BadRequestError as e:
            # 400: часто означает, что модель не поддерживает json_schema.
            # Пробуем без response_format и полагаемся на parse_generation.
            if "response_format" in str(e).lower() or "json_schema" in str(e).lower():
                print("[400] Модель не поддерживает json_schema, пробуем без него...")
                response = client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": "Return ONLY valid JSON. No markdown."},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0.8,
                )
                return response.choices[0].message.content
            raise

    raise RuntimeError(f"Не удалось получить ответ после {max_retries} попыток")