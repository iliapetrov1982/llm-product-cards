import asyncio
import json
import logging
import random
import time
from collections.abc import Callable, Awaitable
from typing import Any
from dataclasses import dataclass

from openai import (
    AsyncOpenAI,
    APIConnectionError,
    APITimeoutError,
    APIStatusError,
)

from llm_product_cards.models import CardResult, Product
from llm_product_cards.prompts import build_messages, build_retry_messages
from llm_product_cards.tokenization import fit_reviews
from llm_product_cards.validation import Violation, validate_card

logger = logging.getLogger(__name__)

RETRYABLE_STATUS = {408, 409, 429, 500, 502, 503, 504}


@dataclass(frozen=True)
class GenerationParams:
    """Sampling settings for one experiment; vLLM-only knobs travel in extra_body."""

    temperature: float = 0.3
    top_p: float = 0.9
    top_k: int = 50
    repetition_penalty: float = 1.1
    max_tokens: int = 350

    def as_kwargs(self) -> dict[str, Any]:
        return {
            "temperature": self.temperature,
            "top_p": self.top_p,
            "max_tokens": self.max_tokens,
            "extra_body": {"top_k": self.top_k, "repetition_penalty": self.repetition_penalty},
        }


@dataclass(frozen=True)
class Completion:
    """One model answer with the accounting needed by metrics."""

    text: str
    prompt_tokens: int
    completion_tokens: int
    finish_reason: str


def calculate_retry_delay(
    attempt: int,
    base_delay: float = 1.0,
    max_delay: float = 30.0,
) -> float:
    """Calculates exponential backoff with random bias."""

    exponential_delay = base_delay * 2 ** (attempt - 1)
    jitter = random.uniform(0, base_delay)

    return min(exponential_delay + jitter, max_delay)


def make_client(base_url: str, api_key: str, timeout: float = 120.0) -> AsyncOpenAI:
    """Async OpenAI-compatible client; SDK retries are off so that attempts and tokens are counted here."""

    return AsyncOpenAI(
        base_url=base_url,
        api_key=api_key,
        timeout=timeout,
        max_retries=0,
    )


def _is_retryable(error: Exception) -> bool:
    if isinstance(error, (APITimeoutError, APIConnectionError)):
        return True            # no response at all: network, timeout
    if isinstance(error, APIStatusError):
        return error.status_code in RETRYABLE_STATUS
    return False


async def _complete(
    client: AsyncOpenAI,
    model: str,
    messages: list[dict[str, str]],
    params: GenerationParams,
    max_attempts: int = 3
) -> Completion:
    """Sends a request with retries for transient errors."""

    last_error: Exception | None = None
    for attempt in range(1, max_attempts + 1):
        try:
            kwargs = params.as_kwargs()
            response = await client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=kwargs["temperature"],
                top_p=kwargs["top_p"],
                max_tokens=kwargs["max_tokens"],
                extra_body=kwargs["extra_body"],
            )

            choice = response.choices[0]
            usage = response.usage

            return Completion(
                text=choice.message.content or "",
                prompt_tokens=usage.prompt_tokens if usage else 0,
                completion_tokens=usage.completion_tokens if usage else 0,
                finish_reason=choice.finish_reason or "",
            )

        except (APIStatusError, APITimeoutError, APIConnectionError) as error:
            if not _is_retryable(error):
                raise
            last_error = error
            if attempt == max_attempts:
                break
            delay = calculate_retry_delay(attempt)
            logger.warning("%s on attempt %d/%d, retrying in %.1fs", type(error).__name__, attempt, max_attempts, delay)
            await asyncio.sleep(delay)

    raise RuntimeError(f"request failed after {max_attempts} attempts") from last_error


RequestFn = Callable[[list[dict[str, str]]], Awaitable[Completion]]


def make_request_fn(client: AsyncOpenAI, model: str, params: GenerationParams) -> RequestFn:
    async def request(messages: list[dict[str, str]]) -> Completion:
        return await _complete(client, model, messages, params)
    return request


async def generate_card(product: Product, request_fn: RequestFn, max_attempts: int = 3) -> CardResult:
    """Get a valid card for one product, retrying on format violations.

        The whole function is timed: this is the per-card latency reported in metrics.
        Tokens are summed over all attempts, so correction retries are paid for honestly.
        Transport failures do not raise: the product ends up as a failed card with `error`
        set, and the batch run goes on.
        """

    started = time.perf_counter()

    # Long-input strategy: keep the longest prefix of reviews that fits the budget.
    # Both counts go into the result so the report can show how much was cut.
    reviews = fit_reviews(product)
    messages = build_messages(product, reviews)

    card = None
    violations: list[Violation] = []
    prompt_tokens = completion_tokens = 0
    error: str | None = None
    attempt = 0

    for attempt in range(1, max_attempts + 1):
        # Transport layer (429, 5xx, timeouts) is handled inside request_fn -> _complete.
        # Whatever still escapes - 400, retries exhausted, anything unexpected - must not
        # kill the batch: record it and stop trying this product.
        try:
            completion = await request_fn(messages)
        except Exception as exc:  # noqa: BLE001 - one bad product must not abort 300 others
            error = f"{type(exc).__name__}: {exc}"
            logger.error("%s: request failed on attempt %d: %s", product["product_id"], attempt, error)
            break

        prompt_tokens += completion.prompt_tokens
        completion_tokens += completion.completion_tokens

        violations = validate_card(completion.text, product["product_id"])

        # A cut-off answer shows up as invalid JSON, but the model needs a different hint
        # ("write less"), and the metrics need a different reason than a Markdown wrapper.
        if completion.finish_reason == "length":
            violations.append(
                Violation("truncated", "the answer was cut off at max_tokens, make the description shorter")
            )

        if not violations:
            # validate_card has already parsed this text successfully; parsing a ~300-char
            # string twice is cheaper than widening the validator's contract
            card = json.loads(completion.text)
            break

        logger.info(
            "%s: attempt %d/%d rejected: %s",
            product["product_id"], attempt, max_attempts, ", ".join(v.rule for v in violations),
        )

        if attempt < max_attempts:
            # Correction retry: the model sees its own answer and every violation at once
            messages = build_retry_messages(messages, completion.text, violations)

    return CardResult(
        product_id=product["product_id"],
        card=card,
        attempts=attempt,
        violations=[v.rule for v in violations],  # stable keys for errors_by_reason
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        latency_sec=time.perf_counter() - started,
        reviews_used=len(reviews),
        reviews_total=len(product["reviews"]),
        error=error,
    )
