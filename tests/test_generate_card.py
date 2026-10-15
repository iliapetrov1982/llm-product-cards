import json

import pytest

from llm_product_cards.clients import Completion, generate_card, make_request_fn, GenerationParams
from llm_product_cards.models import Product

VALID_CARD = {
    "product_id": "P-0001",
    "description": (
        "Компактный пылесос с контейнером на 2 литра и мощностью всасывания 220 Вт. "
        "Работает до 60 минут от батареи и весит всего 1.5 кг. "
        "Подходит для ежедневной уборки небольшой квартиры."
    ),
    "pros": ["Лёгкий", "Долго работает", "Удобные насадки"],
    "cons": ["Щётка забивается волосами"],
    "tags": ["пылесос", "беспроводной", "контейнер", "Klin"],
}
VALID_JSON = json.dumps(VALID_CARD, ensure_ascii=False)
WRAPPED_JSON = "```json\n" + VALID_JSON + "\n```"


@pytest.fixture
def product() -> Product:
    return {
        "product_id": "P-0001",
        "title": "Пылесос Klin Eco",
        "category": "Бытовая техника / Пылесосы",
        "attributes": {"бренд": "Klin", "мощность всасывания": "220 Вт", "вес": "1.5 кг"},
        "reviews": ["Щётка забивается волосами", "Под мебель достаёт легко", "Шумный"],
    }


def completion(text: str, finish_reason: str = "stop") -> Completion:
    return Completion(text=text, prompt_tokens=100, completion_tokens=50, finish_reason=finish_reason)


class ScriptedRequests:
    """request_fn stub: hands out scripted completions (or raises) and records every messages list."""

    def __init__(self, *outcomes):
        self.outcomes = list(outcomes)
        self.calls: list[list[dict[str, str]]] = []

    async def __call__(self, messages):
        self.calls.append(messages)
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


async def test_valid_on_first_attempt(product):
    requests = ScriptedRequests(completion(VALID_JSON))
    result = await generate_card(product, requests)

    assert result.card == VALID_CARD
    assert result.attempts == 1
    assert result.violations == []
    assert result.error is None
    assert (result.prompt_tokens, result.completion_tokens) == (100, 50)
    assert result.latency_sec > 0
    assert result.reviews_used == result.reviews_total == 3


async def test_correction_retry_fixes_markdown_wrapper(product):
    requests = ScriptedRequests(completion(WRAPPED_JSON), completion(VALID_JSON))
    result = await generate_card(product, requests)

    assert result.card == VALID_CARD
    assert result.attempts == 2
    assert result.violations == []
    # tokens are summed over both attempts: retries are not free
    assert (result.prompt_tokens, result.completion_tokens) == (200, 100)

    # the second request carries the failed answer and the violation details
    second = requests.calls[1]
    assert [m["role"] for m in second] == ["system", "user", "assistant", "user"]
    assert second[2]["content"] == WRAPPED_JSON
    assert "JSON" in second[3]["content"]


async def test_gives_up_after_max_attempts(product):
    requests = ScriptedRequests(*[completion("not json")] * 3)
    result = await generate_card(product, requests, max_attempts=3)

    assert result.card is None
    assert result.attempts == 3
    assert result.violations == ["invalid_json"]
    assert result.error is None
    assert len(requests.calls) == 3
    assert result.prompt_tokens == 300


async def test_truncated_answer_is_a_separate_reason(product):
    cut = VALID_JSON[:-20]  # cut mid-JSON, as max_tokens would
    requests = ScriptedRequests(completion(cut, finish_reason="length"), completion(VALID_JSON))
    result = await generate_card(product, requests)

    assert result.card == VALID_CARD
    assert result.attempts == 2
    retry_text = requests.calls[1][3]["content"]
    assert "max_tokens" in retry_text


async def test_truncated_but_parsable_answer_is_still_retried(product):
    # finish_reason=length with text that happens to parse must not be accepted
    requests = ScriptedRequests(completion(VALID_JSON, finish_reason="length"), completion(VALID_JSON))
    result = await generate_card(product, requests)

    assert result.attempts == 2
    assert len(requests.calls) == 2


async def test_transport_failure_does_not_raise(product):
    requests = ScriptedRequests(RuntimeError("request failed after 3 attempts"))
    result = await generate_card(product, requests)

    assert result.card is None
    assert result.error is not None and "request failed" in result.error
    assert result.attempts == 1
    assert result.violations == []
    assert (result.prompt_tokens, result.completion_tokens) == (0, 0)


async def test_transport_failure_on_second_attempt_keeps_first_attempt_tokens(product):
    requests = ScriptedRequests(completion("not json"), RuntimeError("boom"))
    result = await generate_card(product, requests)

    assert result.card is None
    assert result.attempts == 2
    assert result.prompt_tokens == 100  # the failed first attempt was still billed
    assert result.error is not None


async def test_long_product_reviews_are_cut(product):
    long_product = {
        **product,
        "reviews": [f"Отзыв {i}: пылесос работает неплохо, но контейнер маловат и шнур короткий." for i in range(200)],
    }
    requests = ScriptedRequests(completion(VALID_JSON))
    result = await generate_card(long_product, requests)

    assert result.reviews_total == 200
    assert 0 < result.reviews_used < 200


async def test_make_request_fn_delegates_to_client():
    from types import SimpleNamespace
    from unittest.mock import AsyncMock

    response = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content="x"), finish_reason="stop")],
        usage=SimpleNamespace(prompt_tokens=1, completion_tokens=2),
    )
    create = AsyncMock(return_value=response)
    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))

    request_fn = make_request_fn(client, "model-x", GenerationParams(temperature=0.1))
    result = await request_fn([{"role": "user", "content": "hi"}])

    assert result == Completion("x", 1, 2, "stop")
    assert create.await_args.kwargs["model"] == "model-x"
    assert create.await_args.kwargs["temperature"] == 0.1
