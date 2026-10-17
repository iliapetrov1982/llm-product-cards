from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest
from openai import APIConnectionError, APIStatusError, BadRequestError, RateLimitError

from llm_product_cards import clients
from llm_product_cards.clients import Completion, GenerationParams, _complete, _is_retryable

MESSAGES = [{"role": "user", "content": "hi"}]


def status_error(status: int) -> APIStatusError:
    request = httpx.Request("POST", "http://test/v1/chat/completions")
    response = httpx.Response(status, request=request)
    cls = {429: RateLimitError, 400: BadRequestError}.get(status, APIStatusError)
    return cls(f"status {status}", response=response, body=None)


def fake_response(text: str, finish_reason: str = "stop", usage: bool = True):
    """Shape of openai.types.chat.ChatCompletion as far as _complete reads it."""
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=text), finish_reason=finish_reason)],
        usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5) if usage else None,
    )


def fake_client(*outcomes):
    """Client whose chat.completions.create returns/raises the given outcomes in order."""
    create = AsyncMock(side_effect=list(outcomes))
    return SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create))), create


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch):
    monkeypatch.setattr(clients, "calculate_retry_delay", lambda attempt: 0.0)


def test_generation_params_kwargs_route_vllm_knobs_to_extra_body():
    kwargs = GenerationParams(temperature=0.7, top_k=20, repetition_penalty=1.2).as_kwargs()
    assert kwargs["temperature"] == 0.7
    assert kwargs["extra_body"] == {"top_k": 20, "repetition_penalty": 1.2}
    assert "top_k" not in kwargs  # not a standard OpenAI argument


def test_generation_params_extra_fields_are_merged_into_extra_body():
    params = GenerationParams(top_k=None, repetition_penalty=None, extra={"reasoning": {"exclude": True}})
    assert params.as_kwargs()["extra_body"] == {"reasoning": {"exclude": True}}
    params = GenerationParams(extra={"reasoning": {"exclude": True}})
    assert params.as_kwargs()["extra_body"] == {"reasoning": {"exclude": True}, "top_k": 50, "repetition_penalty": 1.1}


def test_generation_params_none_knobs_are_not_sent():
    kwargs = GenerationParams(top_k=None, repetition_penalty=None).as_kwargs()
    assert kwargs["extra_body"] == {}
    assert GenerationParams(top_k=None).as_kwargs()["extra_body"] == {"repetition_penalty": 1.1}


@pytest.mark.parametrize("status, expected", [(429, True), (500, True), (503, True), (400, False), (401, False), (404, False)])
def test_is_retryable_by_status(status, expected):
    assert _is_retryable(status_error(status)) is expected


def test_is_retryable_network_error():
    error = APIConnectionError(request=httpx.Request("POST", "http://test"))
    assert _is_retryable(error) is True


async def test_complete_returns_completion_with_usage():
    client, create = fake_client(fake_response('{"a": 1}'))
    result = await _complete(client, "m", MESSAGES, GenerationParams())
    assert result == Completion(text='{"a": 1}', prompt_tokens=10, completion_tokens=5, finish_reason="stop")
    assert create.await_count == 1


async def test_complete_passes_generation_params():
    client, create = fake_client(fake_response("x"))
    await _complete(client, "m", MESSAGES, GenerationParams(temperature=0.0, max_tokens=99, top_k=7))
    kwargs = create.await_args.kwargs
    assert kwargs["model"] == "m" and kwargs["messages"] == MESSAGES
    assert kwargs["temperature"] == 0.0 and kwargs["max_tokens"] == 99
    assert kwargs["extra_body"]["top_k"] == 7


async def test_complete_retries_on_429_then_succeeds():
    client, create = fake_client(status_error(429), status_error(503), fake_response("ok"))
    result = await _complete(client, "m", MESSAGES, GenerationParams(), max_attempts=3)
    assert result.text == "ok"
    assert create.await_count == 3


async def test_complete_does_not_retry_on_400():
    client, create = fake_client(status_error(400), fake_response("never"))
    with pytest.raises(BadRequestError):
        await _complete(client, "m", MESSAGES, GenerationParams())
    assert create.await_count == 1


async def test_complete_gives_up_after_max_attempts():
    client, create = fake_client(status_error(500), status_error(500))
    with pytest.raises(RuntimeError) as info:
        await _complete(client, "m", MESSAGES, GenerationParams(), max_attempts=2)
    assert create.await_count == 2
    assert isinstance(info.value.__cause__, APIStatusError)  # original cause is kept


async def test_complete_handles_empty_content_and_missing_usage():
    client, _ = fake_client(fake_response(None, finish_reason="length", usage=False))
    result = await _complete(client, "m", MESSAGES, GenerationParams())
    assert result.text == ""
    assert result.prompt_tokens == 0 and result.completion_tokens == 0
    assert result.finish_reason == "length"
