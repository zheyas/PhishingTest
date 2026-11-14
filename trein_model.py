import pandas as pd
import numpy as np
import re
import json
import io
import requests
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments
)
from datasets import load_dataset

# ============================================================
# 1. ЗАГРУЗКА И ОБРАБОТКА ПЕРВОГО ДАТАСЕТА (Phishing)
# ============================================================
print("Загрузка первого датасета (Uzbek_and_Russian_Phishing)...")
dataset1 = load_dataset("Bektemir96az96/Uzbek_and_Russian_Phishing_dataset", split="train")
df1 = dataset1.to_pandas()

df1 = df1[["text", "label"]].copy()
df1 = df1.dropna(subset=["text"])
print(f"Первый датасет: {len(df1)} строк")

# ============================================================
# 2. ЗАГРУЗКА И ОБРАБОТКА ВТОРОГО ДАТАСЕТА (Scam Detection)
#    РЕШЕНИЕ 2: прямое скачивание JSONL через requests
# ============================================================
print("Загрузка второго датасета (Scam Detection SFT) — прямое скачивание...")

# Прямая ссылка на файл внутри репозитория датасета
# При необходимости можно поменять train_0.jsonl на другой файл из /data
url = "https://huggingface.co/datasets/dokster/adaption-scam-dataset-ru-en-v1/resolve/main/data/train_0.jsonl"

response = requests.get(url, timeout=60)
response.raise_for_status()  # выбросит исключение при HTTP-ошибке

# Читаем JSONL (по одному JSON-объекту на строку)
df2 = pd.read_json(io.StringIO(response.text), lines=True)
print(f"Второй датасет загружен: {len(df2)} строк")
print(f"Колонки: {list(df2.columns)}")

# 2.1. Фильтруем только русскоязычные строки
df2_ru = df2[df2["language"] == "ru"].copy()
print(f"Второй датасет (только ru): {len(df2_ru)} строк")


# 2.2. Извлекаем текст сообщения из колонки 'user'
def extract_message_from_user_prompt(user_text):
    if not isinstance(user_text, str):
        return ""
    match = re.search(
        r"/no_think\s*(.*?)\s*Return only the JSON object",
        user_text, re.DOTALL | re.IGNORECASE
    )
    if match:
        return match.group(1).strip()
    return user_text.strip()


df2_ru["text"] = df2_ru["user"].apply(extract_message_from_user_prompt)


# 2.3. Преобразуем risk_level в бинарную метку (0/1)
def map_risk_to_label(json_str):
    if not isinstance(json_str, str):
        return np.nan
    try:
        json_str_clean = re.sub(r"^```json\s*", "", json_str, flags=re.IGNORECASE)
        json_str_clean = re.sub(r"```$", "", json_str_clean).strip()
        data = json.loads(json_str_clean)
        risk = data.get("risk_level", "").lower()

        if risk in ["dangerous", "suspicious"]:
            return 1
        elif risk == "safe":
            return 0
        elif risk == "needs_check":
            return 1
        else:
            return np.nan
    except (json.JSONDecodeError, AttributeError):
        return np.nan


df2_ru["label"] = df2_ru["assistant"].apply(map_risk_to_label)

df2_ru = df2_ru.dropna(subset=["text", "label"])
df2_ru = df2_ru[df2_ru["text"].str.strip() != ""]
df2_ru["label"] = df2_ru["label"].astype(int)
df2_ru = df2_ru[["text", "label"]].copy()
print(f"Второй датасет после обработки: {len(df2_ru)} строк")

# ============================================================
# 3. ОБЪЕДИНЕНИЕ ДАТАСЕТОВ
# ============================================================
df_combined = pd.concat([df1, df2_ru], ignore_index=True)
print(f"Объединённый датасет: {len(df_combined)} строк")
print("Распределение классов в объединённом датасете:")
print(df_combined["label"].value_counts())

df_combined = df_combined.sample(frac=1, random_state=42).reset_index(drop=True)

# ============================================================
# 4. ПРЕДОБРАБОТКА ТЕКСТА
# ============================================================
def clean_text_russian(text):
    text = str(text).lower()
    text = re.sub(r"http\S+", " url ", text)
    text = re.sub(r"\S+@\S+", " email ", text)
    text = re.sub(r"[^а-яёa-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


df_combined["text"] = df_combined["text"].apply(clean_text_russian)

X = np.asarray(df_combined["text"].astype(str).tolist(), dtype=object)
y = np.asarray(df_combined["label"].astype(int).tolist(), dtype=int)

# ============================================================
# 5. TRAIN/TEST SPLIT, ТОКЕНИЗАЦИЯ И ОБУЧЕНИЕ
# ============================================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

MODEL_NAME = "DeepPavlov/rubert-base-cased"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

train_encodings = tokenizer(
    list(X_train), truncation=True, padding=True, max_length=256
)
test_encodings = tokenizer(
    list(X_test), truncation=True, padding=True, max_length=256
)


class EmailDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


train_dataset = EmailDataset(train_encodings, y_train)
test_dataset = EmailDataset(test_encodings, y_test)

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME, num_labels=2
)


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return {"accuracy": accuracy_score(labels, predictions)}


training_args = TrainingArguments(
    output_dir="./results_combined_ru",
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=64,
    warmup_steps=500,
    weight_decay=0.01,
    logging_steps=100,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    compute_metrics=compute_metrics,
)

trainer.train()

# ============================================================
# 6. ОЦЕНКА
# ============================================================
results = trainer.evaluate()
print(f"Evaluation Accuracy: {results['eval_accuracy']:.4f}")

predictions = trainer.predict(test_dataset)
y_pred = np.argmax(predictions.predictions, axis=-1)
print(classification_report(y_test, y_pred, target_names=["Benign", "Phishing/Scam"]))