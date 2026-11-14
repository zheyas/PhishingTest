"""
Валидация и санитизация сгенерированных фишинговых писем.
Гарантирует, что стажёр не увидит битых аннотаций, реальных доменов,
refusal-артефактов LLM и прочего «мусора».
"""
import re
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any

# RFC 2606 / RFC 6761 — гарантированно неразрешаемые TLD
RESERVED_TLDS = (".test", ".example", ".invalid", ".localhost")
ALLOWED_DOMAINS = {"example.com", "example.org", "example.net"}

URL_RE = re.compile(r'https?://[^\s<>"\']+')
EMAIL_RE = re.compile(r'[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}')
HTML_TAG_RE = re.compile(r'<[^>]+>')
MARKDOWN_RE = re.compile(r'\*\*.+?\*\*|__.+?__|^#{1,6}\s+', re.MULTILINE)
PHONE_RU_RE = re.compile(
    r'(?:\+7|8)[\s\-\(\)]*\d{3}[\s\-\(\)]*\d{3}[\s\-]?\d{2}[\s\-]?\d{2}'
)
REFUSAL_RE = re.compile(
    r'\b(as an ai|i cannot|i can\'t|я не могу|я не могу выполнить|'
    r'извините, но я)\b',
    re.IGNORECASE,
)

ALLOWED_TOP_KEYS = {"subject", "body", "annotations"}


@dataclass
class ValidationResult:
    ok: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    sanitized: Optional[Dict[str, Any]] = None


# ---------- Утилиты ----------
def _host_from_url(url: str) -> str:
    rest = re.sub(r'^https?://', '', url)
    host_part = re.split(r'[/?#]', rest, maxsplit=1)[0]
    if '@' in host_part:
        host_part = host_part.rsplit('@', 1)[1]
    return host_part.split(':', 1)[0].lower()


def _userinfo_from_url(url: str) -> Optional[str]:
    rest = re.sub(r'^https?://', '', url)
    host_part = re.split(r'[/?#]', rest, maxsplit=1)[0]
    if '@' in host_part:
        return host_part.rsplit('@', 1)[0]
    return None


def _is_reserved(host: str) -> bool:
    host = host.lower().rstrip('.')
    if host in ALLOWED_DOMAINS:
        return True
    return any(host.endswith(tld) for tld in RESERVED_TLDS)


# ---------- Проверки ----------
def _check_structure(data: dict, errors: list):
    for key in ("subject", "body"):
        if not isinstance(data.get(key), str):
            errors.append(f"Поле '{key}' отсутствует или не строка")
    if not isinstance(data.get("annotations"), list):
        errors.append("Поле 'annotations' отсутствует или не список")
    if errors:
        return

    if not data["subject"].strip():
        errors.append("subject пустой")
    if len(data["subject"]) > 200:
        errors.append("subject длиннее 200 символов")
    if len(data["body"].strip()) < 20:
        errors.append("body короче 20 символов")
    if len(data["body"]) > 8000:
        errors.append("body длиннее 8000 символов")


def _check_urls(text: str, errors: list):
    for url in URL_RE.findall(text):
        host = _host_from_url(url)
        if not host:
            errors.append(f"Не удалось извлечь хост из URL: {url}")
            continue
        if not _is_reserved(host):
            errors.append(f"URL ведёт на реальный домен: {url} (host={host})")


def _check_emails(text: str, warnings: list):
    spans = [(m.start(), m.end()) for m in URL_RE.finditer(text)]
    for m in EMAIL_RE.finditer(text):
        if any(s <= m.start() and m.end() <= e for s, e in spans):
            continue  # часть URL (например, userinfo в at_symbol)
        domain = m.group(0).rsplit('@', 1)[1].lower()
        if not _is_reserved(domain):
            warnings.append(f"Email на нерезервированном домене: {m.group(0)}")


def _check_artifacts(text: str, errors: list, warnings: list):
    if HTML_TAG_RE.search(text):
        errors.append("Обнаружены HTML-теги в тексте")
    if MARKDOWN_RE.search(text):
        warnings.append("Обнаружена markdown-разметка (будет удалена при санитизации)")
    if REFUSAL_RE.search(text):
        errors.append("Обнаружены refusal-артефакты LLM")
    phone = PHONE_RU_RE.search(text)
    if phone:
        warnings.append(f"Найден телефонный номер: {phone.group(0)}")


def _check_annotations(data: dict, errors: list, warnings: list) -> list:
    full_text = data["subject"] + " " + data["body"]
    valid = []
    for i, ann in enumerate(data["annotations"]):
        if not isinstance(ann, dict):
            errors.append(f"annotations[{i}]: не объект")
            continue
        missing = {"technique", "fragment", "explanation_ru"} - ann.keys()
        if missing:
            errors.append(f"annotations[{i}]: нет полей {missing}")
            continue
        frag = ann["fragment"]
        if not frag or frag not in full_text:
            errors.append(f"annotations[{i}]: fragment не найден в тексте: {frag!r}")
            continue
        if len(ann["explanation_ru"].strip()) < 20:
            errors.append(f"annotations[{i}]: explanation_ru короче 20 символов")
            continue
        extra = set(ann.keys()) - {"technique", "fragment", "explanation_ru"}
        if extra:
            warnings.append(f"annotations[{i}]: лишние поля {extra} будут удалены")
        valid.append(
            {k: ann[k] for k in ("technique", "fragment", "explanation_ru")}
        )
    if not valid:
        errors.append("Ни одной валидной аннотации")
    return valid


def _check_technique(data: dict, technique: str, errors: list):
    text = data["subject"] + " " + data["body"]
    urls = URL_RE.findall(text)

    if technique == "at_symbol":
        if not any(_userinfo_from_url(u) for u in urls):
            errors.append("at_symbol: нет URL с userinfo (часть до @)")
        for u in urls:
            info = _userinfo_from_url(u)
            if info and "." not in info:
                errors.append(f"at_symbol: userinfo не похож на домен: {info!r}")

    elif technique == "idn_homograph":
        has_unicode_host = any(
            any(ord(c) > 127 for c in _host_from_url(u)) for u in urls
        )
        if not has_unicode_host:
            errors.append("idn_homograph: нет URL с non-ASCII символами в хосте")

    elif technique == "urgency":
        markers = (
            "немедленно", "срочно", "в течение", "до конца",
            "блокир", "минут", "час",
        )
        if not any(m in text.lower() for m in markers):
            errors.append("urgency: нет маркеров срочности")

    elif technique == "quishing":
        if "qr" not in text.lower() and "куар" not in text.lower():
            errors.append("quishing: нет упоминания QR-кода")
        if urls:
            errors.append("quishing: найден текстовый URL — должен быть только QR")


# ---------- Публичный API ----------
def validate_generation(data: dict, technique: str) -> ValidationResult:
    if not isinstance(data, dict):
        return ValidationResult(ok=False, errors=["Ответ не объект"])

    errors, warnings = [], []

    _check_structure(data, errors)
    if errors:
        return ValidationResult(ok=False, errors=errors, warnings=warnings)

    text = data["subject"] + " " + data["body"]
    _check_urls(text, errors)
    _check_emails(text, warnings)
    _check_artifacts(text, errors, warnings)
    annotations = _check_annotations(data, errors, warnings)
    _check_technique(data, technique, errors)

    if errors:
        return ValidationResult(ok=False, errors=errors, warnings=warnings)

    extra = set(data.keys()) - ALLOWED_TOP_KEYS
    if extra:
        warnings.append(f"Лишние top-level поля {extra} будут удалены")

    sanitized = {
        "subject": data["subject"].strip(),
        "body": data["body"].strip(),
        "annotations": annotations,
    }
    return ValidationResult(ok=True, errors=[], warnings=warnings, sanitized=sanitized)


def sanitize(sample: dict) -> dict:
    """Убирает markdown и нормализует пробелы; fragment чинит при необходимости."""
    def clean(s: str) -> str:
        s = re.sub(r'\*\*(.+?)\*\*', r'\1', s)
        s = re.sub(r'__(.+?)__', r'\1', s)
        s = re.sub(r'^#{1,6}\s+', '', s, flags=re.MULTILINE)
        return re.sub(r'[ \t]+', ' ', s).strip()

    sample["subject"] = clean(sample["subject"])
    sample["body"] = clean(sample["body"])

    full = sample["subject"] + " " + sample["body"]
    fixed = []
    for ann in sample["annotations"]:
        if ann["fragment"] in full:
            fixed.append(ann)
        else:
            cleaned_frag = clean(ann["fragment"])
            if cleaned_frag in full:
                ann["fragment"] = cleaned_frag
                fixed.append(ann)
            # если и это не сработало — аннотацию выбрасываем
    sample["annotations"] = fixed
    return sample
