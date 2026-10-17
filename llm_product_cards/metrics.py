import json
from collections import Counter

from llm_product_cards.config import GPU_HOUR_RUB, API_INPUT_RUB_PER_1M, API_OUTPUT_RUB_PER_1M
from llm_product_cards.models import Product
from llm_product_cards.validation import validate_card


def count_validity(products: list[Product], cards: list[dict]) -> tuple[int, Counter[str]]:
    """Valid card and violation counts by rule, matching cards to products by product_id."""
    cards_by_id: dict[str, dict] = {c["product_id"]: c for c in cards}
    valid = 0
    reasons: Counter[str] = Counter()

    for product in products:
        card = cards_by_id.get(product["product_id"])
        if card is None:
            reasons["missing_card"] += 1  # no card at all counts as a failure
            continue
        violations = validate_card(json.dumps(card, ensure_ascii=False), product["product_id"])
        if violations:
            for v in violations:
                reasons[v.rule] += 1
        else:
            valid += 1

    return valid, reasons


def speed(n: int, elapsed_sec: float, benchmark_size: int = 300) -> tuple[float, float]:
    """Throughput in cards per second and the projected time for the full benchmark."""
    throughput = n / elapsed_sec
    return throughput, benchmark_size / throughput


def cost_own_server_per_1000(elapsed_sec: float, n: int, gpu_hour_rub: float = GPU_HOUR_RUB) -> float:
    """Rubles per 1000 cards when paying for GPU time."""
    return elapsed_sec / 3600 * gpu_hour_rub / n * 1000


def cost_api_per_1000(input_tokens: int, output_tokens: int, n: int,
                      input_rub_per_1m: float = API_INPUT_RUB_PER_1M,
                      output_rub_per_1m: float = API_OUTPUT_RUB_PER_1M) -> float:
    """Rubles per 1000 cards when paying per token."""
    return (input_tokens * input_rub_per_1m + output_tokens * output_rub_per_1m) / 1_000_000 / n * 1000