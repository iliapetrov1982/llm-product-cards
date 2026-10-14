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
