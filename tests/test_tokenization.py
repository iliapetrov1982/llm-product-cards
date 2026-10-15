import pytest

from llm_product_cards.config import MAX_INPUT_TOKENS
from llm_product_cards.models import Product
from llm_product_cards.prompts import build_messages
from llm_product_cards.tokenization import (
    count_prompt_tokens,
    count_tokens,
    fit_reviews,
    get_tokenizer,
)


@pytest.fixture
def short_product() -> Product:
    return {
        "product_id": "P-0001",
        "title": "Пылесос Klin Eco",
        "category": "Бытовая техника / Пылесосы",
        "attributes": {"бренд": "Klin", "мощность всасывания": "220 Вт"},
        "reviews": ["Щётка забивается волосами", "Под мебель достаёт легко", "Шумный"],
    }


@pytest.fixture
def long_product(short_product) -> Product:
    # 200 distinct reviews of ~25 tokens each: far beyond any sane budget
    reviews = [
        f"Отзыв номер {i}: пылесос работает неплохо, но контейнер маловат и шнур короткий."
        for i in range(200)
    ]
    return {**short_product, "reviews": reviews}


def test_tokenizer_is_cached():
    assert get_tokenizer() is get_tokenizer()


def test_count_tokens_empty_and_growing():
    assert count_tokens("") == 0
    assert 0 < count_tokens("Пылесос") < count_tokens("Пылесос Klin Eco с контейнером")


def test_prompt_tokens_include_chat_markup(short_product):
    messages = build_messages(short_product)
    plain = sum(count_tokens(m["content"]) for m in messages)
    assert count_prompt_tokens(messages) > plain


def test_fit_reviews_keeps_short_product_untouched(short_product):
    assert fit_reviews(short_product) == short_product["reviews"]


def test_fit_reviews_respects_budget_and_order(long_product):
    kept = fit_reviews(long_product)
    assert 0 < len(kept) < len(long_product["reviews"])
    assert kept == long_product["reviews"][: len(kept)]  # prefix, original order
    assert count_prompt_tokens(build_messages(long_product, kept)) <= MAX_INPUT_TOKENS


def test_fit_reviews_fills_the_budget(long_product):
    # adding one more review must break the budget, otherwise the fit is too conservative
    kept = fit_reviews(long_product)
    one_more = long_product["reviews"][: len(kept) + 1]
    assert count_prompt_tokens(build_messages(long_product, one_more)) > MAX_INPUT_TOKENS


def test_fit_reviews_with_custom_budget(long_product):
    small = fit_reviews(long_product, budget=900)
    default = fit_reviews(long_product)
    assert 0 < len(small) < len(default)
    assert count_prompt_tokens(build_messages(long_product, small)) <= 900


def test_fit_reviews_budget_below_skeleton_returns_nothing(short_product):
    assert fit_reviews(short_product, budget=100) == []
