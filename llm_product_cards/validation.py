import re
import json
from typing import Any
from dataclasses import dataclass

PRODUCT_ID = "product_id"
DESCRIPTION = "description"
PROS = "pros"
CONS = "cons"
TAGS = "tags"

EXPECTED_FIELDS: set[str] = {
    PRODUCT_ID,
    DESCRIPTION,
    PROS,
    CONS,
    TAGS
}


@dataclass(frozen=True)
class Violation:
    """One broken card rule: a stable key for metrics and a message for the model."""
    rule: str
    detail: str


def count_sentences(text: str) -> int:
    """Number of sentences; a dot inside a number does not end a sentence"""
    return len([s for s in re.split(r"[.!?]+(?!\d)", text) if s.strip()])


def check_list(key_name: str, list_to_check: list[str], interval: tuple[int, int]) -> list[Violation]:
    err: list[Violation] = []

    if not isinstance(list_to_check, list):
        return [
            Violation(f"{key_name}_type", f"{key_name} must be of type list[str], got {type(list_to_check).__name__}")]

    list_len = len(list_to_check)

    if list_len < interval[0] or list_len > interval[1]:
        err.append(
            Violation(f"{key_name}_count", f"{key_name} length {list_len} is beyond the interval {list(interval)}")
        )

    for i, elem in enumerate(list_to_check, start=1):
        if isinstance(elem, str):
            if elem.strip() == "":
                err.append(
                    Violation(f"{key_name}_item_empty", f"{key_name} string number {i} is empty")
                )
        else:
            err.append(
                Violation(f"{key_name}_item_type", f"{key_name} element number {i} is not of type str")
            )

    return err


def validate_card(raw: str, product_id: str) -> list[Violation]:
    errors: list[Violation] = []

    try:
        data: dict[str, Any] = json.loads(raw)
    except json.JSONDecodeError as e:
        return [
            Violation("invalid_json", f"a response isn't a valid JSON: {e.msg}")
        ]

    if not isinstance(data, dict):
        return [
            Violation("not_an_object", f"top-level JSON value must be an object, got {type(data).__name__}")
        ]

    received_fields: set[str] = set(data)
    missing_fields: set[str] = EXPECTED_FIELDS - received_fields

    if missing_fields:
        errors.append(
            Violation("missing_fields", f"lack of mandatory fields: {sorted(missing_fields)}")
        )

    received_prod_id: str | None = data.get(PRODUCT_ID)
    if (received_prod_id is not None) and (product_id != received_prod_id):
        errors.append(
            Violation(
                "product_id_mismatch",
                f"{PRODUCT_ID} mismatch: in prompt - {product_id} vs received - {received_prod_id}"
            )
        )

    description: str | None = data.get(DESCRIPTION)

    if description is None:
        pass  # already reported as a missing field
    elif not isinstance(description, str):
        errors.append(Violation("description_type", f"{DESCRIPTION} is not of type str"))
    else:
        description_len = len(description)
        if 120 > description_len or description_len > 700:
            errors.append(
                Violation(
                    "description_length",
                    f"{DESCRIPTION} length {description_len} is beyond the interval [120, 700]"
                )
            )

        sentence_cnt = count_sentences(description)
        if 2 > sentence_cnt or sentence_cnt > 5:
            errors.append(
                Violation(
                    "description_sentences",
                    f"number of sentences {sentence_cnt} in the description is beyond the interval [2, 5]"
                )
            )

    pros: list[str] | None = data.get(PROS)
    if pros is not None:
        errors += check_list(PROS, pros, (2, 5))

    cons: list[str] | None = data.get(CONS)
    if cons is not None:
        errors += check_list(CONS, cons, (1, 3))

    tags: list[str] | None = data.get(TAGS)
    if tags is not None:
        errors += check_list(TAGS, tags, (3, 8))

    return errors
