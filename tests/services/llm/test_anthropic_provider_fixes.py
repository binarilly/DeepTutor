"""Regression tests for Anthropic provider support of newer Claude models.

Covers three fixes:
1. A ``base_url`` ending in ``/v1`` must not be doubled by the SDK.
2. ``temperature`` is omitted for effort-based models that reject it.
3. ``cache_control`` breakpoints never exceed the Anthropic limit of 4.
"""

from __future__ import annotations

from typing import Any

from deeptutor.services.llm.provider_core.anthropic_provider import AnthropicProvider


def _provider(model: str = "claude-opus-4-8", api_base: str | None = None) -> AnthropicProvider:
    return AnthropicProvider(api_key="test-key", api_base=api_base, default_model=model)


def test_base_url_trailing_v1_is_not_doubled() -> None:
    # The SDK appends its own `/v1/...`; a base_url of `.../v1` would 404.
    provider = _provider(api_base="https://api.anthropic.com/v1")
    assert str(provider._client.base_url).rstrip("/") == "https://api.anthropic.com"


def test_base_url_without_v1_is_preserved() -> None:
    provider = _provider(api_base="https://proxy.example.com/anthropic")
    assert str(provider._client.base_url).rstrip("/") == "https://proxy.example.com/anthropic"


def _kwargs(provider: AnthropicProvider, model: str) -> dict[str, Any]:
    return provider._build_kwargs(
        messages=[{"role": "user", "content": "hi"}],
        tools=None,
        model=model,
        max_tokens=1024,
        temperature=0.7,
        reasoning_effort=None,
        tool_choice=None,
    )


def test_temperature_omitted_for_effort_based_models() -> None:
    provider = _provider()
    for model in ("claude-opus-4-8", "claude-sonnet-5", "claude-opus-4-7"):
        assert "temperature" not in _kwargs(provider, model), model


def test_temperature_kept_for_older_models() -> None:
    provider = _provider()
    for model in ("claude-sonnet-4-5-20250929", "claude-haiku-4-5-20251001", "claude-opus-4-1"):
        assert "temperature" in _kwargs(provider, model), model


def _count_cache_control(system: Any, messages: list[dict[str, Any]], tools: list) -> int:
    s, msgs, tls = AnthropicProvider._apply_cache_control(system, messages, tools)
    total = 0
    if isinstance(s, list):
        total += sum("cache_control" in b for b in s)
    for m in msgs:
        c = m.get("content")
        if isinstance(c, list):
            total += sum("cache_control" in b for b in c)
    if tls:
        total += sum("cache_control" in t for t in tls)
    return total


def test_cache_control_never_exceeds_four() -> None:
    messages = [
        {"role": "user", "content": "a"},
        {"role": "assistant", "content": "b"},
        {"role": "user", "content": "c"},
    ]
    for n_tools in (0, 1, 5, 12, 15, 25, 40):
        tools = [{"name": f"t{i}", "description": "d", "input_schema": {}} for i in range(n_tools)]
        assert _count_cache_control("system prompt", messages, tools) <= 4, n_tools
