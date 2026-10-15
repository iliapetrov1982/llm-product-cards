import asyncio
import logging
import random
from typing import Any
from dataclasses import dataclass

from openai import (
    AsyncOpenAI,
    APIConnectionError,
    APITimeoutError,
    APIStatusError,
)


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
