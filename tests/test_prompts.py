import pytest

from llm_product_cards.models import Product
from llm_product_cards.prompts import (
    SYSTEM_PROMPT,
    build_messages,
    build_retry_messages,
    build_user_message,
)
from llm_product_cards.validation import Violation


@pytest.fixture
def product() -> Product:
    return {
        "product_id": "P-0001",
        "title": "Пылесос Klin Eco",
        "category": "Бытовая техника / Пылесосы",
        "attributes": {"бренд": "Klin", "мощность всасывания": "220 Вт", "вес": "1.5 кг"},
        "reviews": ["Щётка забивается волосами", "Под мебель достаёт легко", "Шумный"],
    }


def test_user_message_contains_all_product_data(product):
    text = build_user_message(product, product["reviews"])
    assert product["product_id"] in text
    assert product["title"] in text
    assert product["category"] in text
    for key, value in product["attributes"].items():
        assert f"{key}: {value}" in text
    for review in product["reviews"]:
        assert review in text


def test_user_message_uses_given_reviews_not_product_reviews(product):
    text = build_user_message(product, ["Только этот отзыв"])
    assert "Только этот отзыв" in text
    assert "Шумный" not in text
    assert "Щётка" not in text


def test_user_message_defaults_to_product_reviews(product):
    assert build_user_message(product) == build_user_message(product, product["reviews"])


def test_user_message_ends_with_format_reminder(product):
    # the format reminder must be the last thing the model reads (lost in the middle)
    text = build_user_message(product, product["reviews"]).rstrip()
    assert text.endswith("без пояснений.")


def test_user_message_has_no_indentation_artifacts(product):
    text = build_user_message(product, product["reviews"])
    assert "\n    " not in text
    assert "\n\n\n" not in text


def test_build_messages_roles_and_content(product):
    messages = build_messages(product, product["reviews"])
    assert [m["role"] for m in messages] == ["system", "user"]
    assert messages[0]["content"] == SYSTEM_PROMPT
    assert messages[1]["content"] == build_user_message(product, product["reviews"])


def test_system_prompt_is_product_independent(product):
    other = {**product, "product_id": "P-0002", "title": "Другой товар"}
    assert build_messages(product)[0] == build_messages(other)[0]


def test_retry_messages_extend_without_mutating(product):
    messages = build_messages(product, product["reviews"])
    snapshot = [dict(m) for m in messages]
    violations = [
        Violation("pros_count", "pros length 1 is beyond the interval [2, 5]"),
        Violation("cons_count", "cons length 0 is beyond the interval [1, 3]"),
    ]
    raw = '{"product_id": "P-0001", "pros": ["x"], "cons": []}'

    retry = build_retry_messages(messages, raw, violations)

    assert messages == snapshot  # original history untouched
    assert len(retry) == 4
    assert [m["role"] for m in retry] == ["system", "user", "assistant", "user"]
    assert retry[2]["content"] == raw
    for v in violations:
        assert v.detail in retry[3]["content"]
