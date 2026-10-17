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
                f"{PRODUCT_ID} must be exactly \"{product_id}\" as given in the input (got \"{received_prod_id}\")"
            )
        )

    description: str | None = data.get(DESCRIPTION)

    if description is None:
        pass  # already reported as a missing field
    elif not isinstance(description, str):
        errors.append(Violation("description_type", f"{DESCRIPTION} is not of type str"))
    else:
        description_len = len(description)
        # Details are read by a small model on retry: say what to do, not only what is wrong
        if description_len < 120:
            errors.append(
                Violation(
                    "description_length",
                    f"{DESCRIPTION} is too short ({description_len} chars): write 3-4 full sentences, "
                    f"about 300 chars, mention the key features and what buyers say"
                )
            )
        elif description_len > 700:
            errors.append(
                Violation(
                    "description_length",
                    f"{DESCRIPTION} is too long ({description_len} chars): keep it within 4-5 sentences, under 700 chars"
                )
            )

        sentence_cnt = count_sentences(description)
        if sentence_cnt < 2:
            errors.append(
                Violation(
                    "description_sentences",
                    f"{DESCRIPTION} has {sentence_cnt} sentence(s): write 3-4 separate sentences ending with a period"
                )
            )
        elif sentence_cnt > 5:
            errors.append(
                Violation(
                    "description_sentences",
                    f"{DESCRIPTION} has {sentence_cnt} sentences: merge them into at most 5"
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
