"""
Точка входа для проверки генератора.
Запуск: python generate_test.py
"""
import json
from phishing_generator import (
    PhishingScenario,
    generate_training_sample,
    present_to_trainee,
    reveal_explanations,
    present_to_admin,
)


def main():
    scenario = PhishingScenario(
        pretext="Служба информационной безопасности",
        target_role="бухгалтер",
        technique="at_symbol",
        difficulty="medium",
    )

    print("=" * 60)
    print("Генерация письма...")
    print("=" * 60)

    sample = generate_training_sample(scenario)

    trainee_view = present_to_trainee(sample)
    print("\n=== ЧТО ВИДИТ СТАЖЁР ===")
    print(f"Тема: {trainee_view['subject']}")
    print()
    print(trainee_view["body"])

    print("\n=== РАЗБОР (после действия стажёра) ===")
    for ann in reveal_explanations(sample):
        print(f"[{ann['technique']}]")
        print(f"  fragment:     {ann['fragment']!r}")
        print(f"  explanation:  {ann['explanation_ru']}")

    print("\n=== ПОЛНЫЙ ОБЪЕКТ (для логов) ===")
    print(json.dumps(present_to_admin(sample), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()