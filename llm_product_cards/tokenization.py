from functools import cache

from transformers import AutoTokenizer, PreTrainedTokenizerBase

from llm_product_cards.models import Product
from llm_product_cards.prompts import build_messages
from llm_product_cards.config import MODEL, MAX_INPUT_TOKENS


@cache
def get_tokenizer() -> PreTrainedTokenizerBase:
    return AutoTokenizer.from_pretrained(MODEL)


def count_tokens(text: str) -> int:
    """Number of tokens based on chosen Model tokenizer."""
    return len(get_tokenizer()(text)["input_ids"])


def count_prompt_tokens(messages: list[dict[str, str]]) -> int:
    """Length of the rendered prompt: messages plus chat markup and the answer prompt."""
    text = get_tokenizer().apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return count_tokens(str(text))


def fit_reviews(product: Product, budget: int = MAX_INPUT_TOKENS) -> list[str]:
    """Longest prefix of the product's reviews whose rendered prompt fits into the token budget."""
    base = count_prompt_tokens(build_messages(product, []))
    remaining = budget - base

    reviews: list[str] = []

    for review in product["reviews"]:
        additional_review_tokens = count_tokens(review) + 2
        if additional_review_tokens <= remaining:  # "-" + "\n" = 2
            reviews.append(review)
            remaining -= additional_review_tokens
        else:
            break

    # The per-review estimate ignores BPE merges across line breaks, so correct it both ways
    # against the rendered prompt: drop while over budget, then top up while there is room.
    while reviews and count_prompt_tokens(build_messages(product, reviews)) > budget:
        reviews.pop()

    all_reviews = product["reviews"]
    while len(reviews) < len(all_reviews):
        candidate = reviews + [all_reviews[len(reviews)]]
        if count_prompt_tokens(build_messages(product, candidate)) > budget:
            break
        reviews = candidate

    return reviews
