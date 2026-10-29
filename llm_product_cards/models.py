from dataclasses import dataclass
from typing import TypedDict


class Product(TypedDict):
    product_id: str
    title: str
    category: str
    attributes: dict[str, str]
    reviews: list[str]


class Card(TypedDict):
    product_id: str
    description: str
    pros: list[str]
    cons: list[str]
    tags: list[str]


@dataclass(frozen=True)
class CardResult:
    """Result of one card generation: card or None, attempts, tokens over all attempts, latency, error."""
    product_id: str
    card: Card | None
    attempts: int
    violations: list[str]
    prompt_tokens: int
    completion_tokens: int
    latency_sec: float
    reviews_used: int
    reviews_total: int
    error: str | None