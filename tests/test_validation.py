import json
import pytest
from llm_product_cards.validation import Violation, count_sentences, validate_card

PRODUCT_ID = "P-1001"


@pytest.fixture
def valid_card() -> dict:
    return {
        "product_id": PRODUCT_ID,
        "description": (
            "Компактные внутриканальные наушники с Bluetooth 5.3 и защитой IPX4. "
            "Держат заряд до 24 часов и надёжно сидят при активном движении. "
            "Подойдут для спорта и повседневного использования."
        ),
        "pros": ["Чистый звук", "Удобная посадка", "Долгое время работы"],
        "cons": ["Быстро разряжаются на морозе"],
        "tags": ["наушники", "беспроводные", "спорт", "IPX4", "Bluetooth 5.3"],
    }


def as_json(obj) -> str:
    return json.dumps(obj, ensure_ascii=False)


def rules_of(violations: list[Violation]) -> list[str]:
    return [v.rule for v in violations]


@pytest.mark.parametrize(
    "text, expected",
    [
        ('Экран 6.1" и Bluetooth 5.3 радуют.', 1),
        ("Первое. Второе! Третье?", 3),
        ("Объём 0.5 л. Хватает надолго...", 2),
        ("", 0),
    ],
)
def test_count_sentences(text, expected):
    assert count_sentences(text) == expected


def test_valid_card_has_no_violations(valid_card):
    assert validate_card(as_json(valid_card), PRODUCT_ID) == []


def test_violation_carries_rule_and_detail(valid_card):
    card = {**valid_card, "cons": []}
    (violation,) = validate_card(as_json(card), PRODUCT_ID)
    assert violation.rule == "cons_count"
    assert "0" in violation.detail and "1" in violation.detail  # actual and expected counts


def test_markdown_fence_is_invalid_json(valid_card):
    raw = "```json\n" + as_json(valid_card) + "\n```"
    assert rules_of(validate_card(raw, PRODUCT_ID)) == ["invalid_json"]


@pytest.mark.parametrize("top_level", [[{"product_id": PRODUCT_ID}], 42, "text", None])
def test_non_object_top_level(top_level):
    violations = validate_card(as_json(top_level), PRODUCT_ID)
    assert rules_of(violations) == ["not_an_object"]
    assert type(top_level).__name__ in violations[0].detail


def test_missing_field_is_reported(valid_card):
    card = {k: v for k, v in valid_card.items() if k != "tags"}
    violations = validate_card(as_json(card), PRODUCT_ID)
    assert rules_of(violations) == ["missing_fields"]
    assert "tags" in violations[0].detail


def test_missing_description_is_reported_once(valid_card):
    card = {k: v for k, v in valid_card.items() if k != "description"}
    # only "missing_fields", not an extra "description_type" for the same problem
    assert rules_of(validate_card(as_json(card), PRODUCT_ID)) == ["missing_fields"]


def test_product_id_mismatch(valid_card):
    card = {**valid_card, "product_id": "P-9999"}
    assert rules_of(validate_card(as_json(card), PRODUCT_ID)) == ["product_id_mismatch"]


def test_description_too_short_and_one_sentence(valid_card):
    card = {**valid_card, "description": "Коротко."}
    rules = rules_of(validate_card(as_json(card), PRODUCT_ID))
    assert sorted(rules) == ["description_length", "description_sentences"]


def test_description_wrong_type(valid_card):
    card = {**valid_card, "description": ["a", "b"]}
    assert rules_of(validate_card(as_json(card), PRODUCT_ID)) == ["description_type"]


def test_list_field_given_as_string_is_a_type_violation(valid_card):
    # "good, cheap" must not be walked character by character (length 11, empty item 6)
    card = {**valid_card, "pros": "good, cheap"}
    assert rules_of(validate_card(as_json(card), PRODUCT_ID)) == ["pros_type"]


def test_short_string_instead_of_list_is_not_silently_valid(valid_card):
    # "ok" has 2 characters, which fits the [2, 5] range if the string is mistaken for a list
    card = {**valid_card, "pros": "ok"}
    assert rules_of(validate_card(as_json(card), PRODUCT_ID)) == ["pros_type"]


def test_list_items_must_be_strings(valid_card):
    card = {**valid_card, "pros": ["ok", 5, None]}
    rules = rules_of(validate_card(as_json(card), PRODUCT_ID))
    assert rules == ["pros_item_type", "pros_item_type"]


@pytest.mark.parametrize(
    "field, value, expected_rule",
    [
        ("pros", ["only one"], "pros_count"),
        ("pros", ["a", "b", "c", "d", "e", "f"], "pros_count"),
        ("cons", [], "cons_count"),
        ("cons", ["a", "b", "c", "d"], "cons_count"),
        ("tags", ["one", "two"], "tags_count"),
        ("tags", ["", "   ", "x", "y"], "tags_item_empty"),
    ],
)
def test_list_length_and_empty_items(valid_card, field, value, expected_rule):
    card = {**valid_card, field: value}
    assert expected_rule in rules_of(validate_card(as_json(card), PRODUCT_ID))


def test_violations_accumulate(valid_card):
    card = {**valid_card, "description": "Коротко.", "cons": []}
    rules = rules_of(validate_card(as_json(card), PRODUCT_ID))
    assert sorted(rules) == ["cons_count", "description_length", "description_sentences"]
