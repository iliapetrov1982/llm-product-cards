from collections import Counter

import pytest

from llm_product_cards.metrics import (
    cost_api_per_1000,
    cost_own_server_per_1000,
    count_validity,
    speed,
)

VALID_CARD = {
    "product_id": "P-0",
    "description": (
        "Хороший товар для дома и работы, подходит всем. "
        "Служит долго и не ломается при аккуратном обращении. "
        "Рекомендуем к покупке без сомнений."
    ),
    "pros": ["Надёжный", "Удобный"],
    "cons": ["Дорогой"],
    "tags": ["товар", "дом", "работа"],
}


def product(product_id: str) -> dict:
    return {"product_id": product_id, "title": "t", "category": "c", "attributes": {}, "reviews": []}


def test_count_validity_matches_cards_to_products():
    products = [product(f"P-{i}") for i in range(4)]
    cards = [
        VALID_CARD,                                     # P-0: valid
        {**VALID_CARD, "product_id": "P-1", "tags": []},  # P-1: one rule broken
        {"product_id": "P-2"},                          # P-2: failed generation, stub only
        # P-3: no card at all
        {**VALID_CARD, "product_id": "P-9"},            # stray card, not in the input
    ]
    valid, reasons = count_validity(products, cards)
    assert valid == 1
    assert reasons == Counter({"tags_count": 1, "missing_fields": 1, "missing_card": 1})


def test_count_validity_counts_every_violation_of_a_card():
    products = [product("P-0")]
    cards = [{**VALID_CARD, "description": "Коротко.", "cons": []}]
    valid, reasons = count_validity(products, cards)
    assert valid == 0
    assert reasons == Counter({"description_length": 1, "description_sentences": 1, "cons_count": 1})


def test_count_validity_all_valid():
    products = [product("P-0"), product("P-1")]
    cards = [VALID_CARD, {**VALID_CARD, "product_id": "P-1"}]
    assert count_validity(products, cards) == (2, Counter())


def test_speed_full_run_projects_to_itself():
    throughput, time_for_300 = speed(300, 330.5)
    assert throughput == pytest.approx(0.9077, abs=1e-4)
    assert time_for_300 == pytest.approx(330.5)


def test_speed_extrapolates_from_a_subset():
    throughput, time_for_300 = speed(30, 33.0)
    assert throughput == pytest.approx(30 / 33)
    assert time_for_300 == pytest.approx(330.0)


def test_cost_own_server_matches_worked_example():
    # 300 cards in 180 s at 65 rub/h -> about 10.8 rub per 1000
    assert cost_own_server_per_1000(180, 300, gpu_hour_rub=65) == pytest.approx(10.83, abs=0.01)


def test_cost_api_matches_worked_example():
    # 25 cards, 34k input and 6k output tokens at 20/60 rub per 1M -> 41.6 rub per 1000
    cost = cost_api_per_1000(34_000, 6_000, 25, input_rub_per_1m=20, output_rub_per_1m=60)
    assert cost == pytest.approx(41.6)


def test_cost_api_scales_linearly_with_tokens():
    base = cost_api_per_1000(10_000, 1_000, 10, input_rub_per_1m=20, output_rub_per_1m=60)
    doubled = cost_api_per_1000(20_000, 2_000, 10, input_rub_per_1m=20, output_rub_per_1m=60)
    assert doubled == pytest.approx(2 * base)
