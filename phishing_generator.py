"""
Модуль генерации тренировочных фишинговых писем на русском языке.
Использует TokenHarbor (OpenAI-совместимый шлюз) + DeepSeek.
Возвращает структурированный объект с текстом и разметкой техник.
"""
import os
import json
import re
import time
from dataclasses import dataclass
from typing import Optional

import openai
from openai import OpenAI
from dotenv import load_dotenv

from validation import validate_generation, sanitize

load_dotenv()

# ---------- Клиент TokenHarbor ----------
client = OpenAI(
    base_url=os.environ.get("TOKENHARBOR_BASE_URL", "https://tokenharbor.ai/v1"),
    api_key=os.environ["TOKENHARBOR_API_KEY"],
)
DEFAULT_MODEL = os.environ.get("TOKENHARBOR_MODEL", "deepseek-v4.1-flash:free")


# ---------- Сценарий ----------
@dataclass
class PhishingScenario:
    pretext: str
    target_role: str
    technique: str
    difficulty: str = "medium"


# ---------- Библиотека техник ----------
TECHNIQUE_SPECS = {
    "at_symbol": {
        "instruction": (
            "Вставь в письмо ссылку, где до символа @ стоит правдоподобный "
            "адрес компании, а после @ — фишинговый домен на .test. "
            "Пример: https://mail.company.ru@verify-login.evil.test/auth"
        ),
    },
    "idn_homograph": {
        "instruction": (
            "Используй домен, визуально неотличимый от легитимного, но с "
            "кириллическими или смешанными символами. Домен — на .test."
        ),
    },
    "urgency": {
        "instruction": (
            "Создай искусственный дедлайн: 'в течение 30 минут', "
            "'до конца дня', 'иначе доступ будет заблокирован'."
        ),
    },
    "quishing": {
        "instruction": (
            "Вместо текстовой ссылки вставь упоминание QR-кода: "
            "'отсканируйте QR-код ниже для подтверждения'. "
            "Текстовая ссылка должна отсутствовать."
        ),
    },
}


# ---------- JSON-схема ответа ----------
PHISHING_SCHEMA = {
    "name": "phishing_training_sample",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "subject": {"type": "string"},
            "body": {"type": "string"},
            "annotations": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "technique": {"type": "string"},
                        "fragment": {"type": "string"},
                        "explanation_ru": {"type": "string"},
                    },
                    "required": ["technique", "fragment", "explanation_ru"],
                    "additionalProperties": False,
                },
            },
        },
        "required": ["subject", "body", "annotations"],
        "additionalProperties": False,
    },
}


def build_prompt(scenario: PhishingScenario) -> str:
    spec = TECHNIQUE_SPECS.get(scenario.technique, {})
    instruction = spec.get("instruction", "")
    return f"""
Ты — генератор тренировочных фишинговых писем для русскоязычной аудитории.
Цель — ОБУЧЕНИЕ сотрудников, а не реальная атака.

Сценарий:
- Легенда: {scenario.pretext}
- Роль получателя: {scenario.target_role}
- Техника фишинга: {scenario.technique}
- Инструкция по технике: {instruction}
- Уровень сложности: {scenario.difficulty}

Верни JSON с полями subject, body, annotations.
В annotations для каждого фишингового элемента укажи:
- technique: код техники (например, "{scenario.technique}")
- fragment: ТОЧНАЯ подстрока из subject или body (без изменения регистра!)
- explanation_ru: объяснение для сотрудника (не короче 20 символов)

Требования:
- Только русский язык в subject и body.
- Все URL — только на доменах .test, .example, .invalid или .localhost.
- Никаких HTML-тегов и markdown-разметки в тексте письма.
""".strip()


def call_llm(prompt: str, model: Optional[str] = None, max_retries: int = 5) -> str:
    model = model or DEFAULT_MODEL

    for attempt in range(1, max_retries + 1):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                response_format={
                    "type": "json_schema",
                    "json_schema": PHISHING_SCHEMA,
                },
                temperature=0.8,
            )
            return response.choices[0].message.content

        except openai.RateLimitError as e:
            retry_after = (
                e.response.headers.get("retry-after") if e.response else None
            )
            wait = float(retry_after) if retry_after else min(2 ** attempt, 30)
            print(
                f"[429] Rate limit. Ждём {wait:.1f}с "
                f"(попытка {attempt}/{max_retries})..."
            )
            time.sleep(wait)

        except openai.BadRequestError as e:
            msg = str(e).lower()
            if "response_format" in msg or "json_schema" in msg:
                print("[400] Модель не поддерживает json_schema, пробуем без него...")
                response = client.chat.completions.create(
                    model=model,
                    messages=[
                        {
                            "role": "system",
                            "content": "Return ONLY valid JSON. No markdown.",
                        },
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0.8,
                )
                return response.choices[0].message.content
            raise

    raise RuntimeError(
        f"Не удалось получить ответ после {max_retries} попыток"
    )


def parse_generation(raw: str) -> Optional[dict]:
    cleaned = re.sub(r"^```json\s*", "", raw.strip(), flags=re.IGNORECASE)
    cleaned = re.sub(r"```$", "", cleaned).strip()
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        return None
    if not {"subject", "body", "annotations"}.issubset(data.keys()):
        return None
    full_text = data["subject"] + " " + data["body"]
    valid = []
    for ann in data["annotations"]:
        frag = ann.get("fragment", "")
        if frag and frag in full_text:
            valid.append(ann)
        else:
            print(f"[WARN] fragment не найден в тексте: {frag!r}")
    data["annotations"] = valid
    return data


def generate_training_sample(
    scenario: PhishingScenario, max_attempts: int = 3
) -> dict:
    last_errors = []
    for attempt in range(1, max_attempts + 1):
        print(f"[{attempt}/{max_attempts}] генерация...")
        raw = call_llm(build_prompt(scenario))
        sample = parse_generation(raw)
        if sample is None:
            last_errors = ["LLM вернул невалидный JSON"]
            print("  ↳ JSON невалиден, повтор")
            continue

        result = validate_generation(sample, scenario.technique)
        for w in result.warnings:
            print(f"  [WARN] {w}")

        if not result.ok:
            last_errors = result.errors
            print("  ↳ валидация провалилась:")
            for e in result.errors:
                print(f"     - {e}")
            continue

        sample = sanitize(result.sanitized)
        sample["difficulty"] = scenario.difficulty
        sample["technique"] = scenario.technique
        return sample

    raise RuntimeError(
        f"Не удалось сгенерировать валидное письмо за {max_attempts} попыток. "
        f"Последние ошибки: {last_errors}"
    )


# ---------- Публичные представления ----------
def present_to_trainee(sample: dict) -> dict:
    """Только то, что видит стажёр во время тренировки."""
    return {"subject": sample["subject"], "body": sample["body"]}


def reveal_explanations(sample: dict) -> list:
    """Показывается после действия стажёра."""
    return sample["annotations"]


def present_to_admin(sample: dict) -> dict:
    """Полный объект для администратора/лог-системы."""
    return sample
