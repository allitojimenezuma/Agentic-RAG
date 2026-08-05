"""Unit tests for OpenCode Go prompt-cache instrumentation in agents/llm.py.

Verifies that when the base URL is the opencode.ai/zen/go gateway, requests
carry the cache fields (prompt_cache_key, prompt_cache_retention) and
cache_control breakpoints, and that non-gateway / unsupported-model paths are
left untouched.
"""

from __future__ import annotations

from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_openai import ChatOpenAI

from agentic_rag.agents.llm import (
    _is_opencode_go,
    _OpencodeGoCachedChat,
    _unsupported_for_cache,
    get_model,
)

GATEWAY_URL = "https://opencode.ai/zen/go/v1"


class _Settings:
    openai_api_key = "sk-test"
    openai_base_url = GATEWAY_URL
    openai_model = "deepseek-v4-flash"


def _model(**overrides) -> _OpencodeGoCachedChat:
    params = dict(
        model="deepseek-v4-flash",
        api_key="sk-test",
        base_url=GATEWAY_URL,
        temperature=0,
        prompt_cache_key="wiki-query",
    )
    params.update(overrides)
    return _OpencodeGoCachedChat(**params)


def test_opencode_gateway_detected() -> None:
    assert _is_opencode_go(GATEWAY_URL)
    assert _is_opencode_go("https://opencode.ai/zen/go/v1/chat/completions")
    assert not _is_opencode_go("https://api.openai.com/v1")


def test_unsupported_model_patterns() -> None:
    assert _unsupported_for_cache("opencode-go/glm-5.2")
    assert _unsupported_for_cache("opencode-go/zhipu-glm")
    assert not _unsupported_for_cache("opencode-go/deepseek-v4-flash")


def test_get_model_returns_cached_subclass_for_gateway() -> None:
    model = get_model(_Settings(), cache_key="wiki-query")
    assert isinstance(model, _OpencodeGoCachedChat)
    # The gateway reads the session from this header: it drives the session
    # column in the usage dashboard and sticky upstream affinity.
    assert model.default_headers["x-opencode-session"] == "wiki-query"
    assert model.default_headers["x-opencode-client"] == "agentic-rag"


def test_session_header_matches_cache_key_when_default() -> None:
    model = get_model(_Settings())
    assert model.default_headers["x-opencode-session"] == "default"
    assert model.prompt_cache_key == "default"


def test_get_model_returns_plain_chatopenai_for_openai() -> None:
    class NonGatewaySettings(_Settings):
        openai_base_url = "https://api.openai.com/v1"

    model = get_model(NonGatewaySettings())
    assert type(model) is ChatOpenAI


def test_payload_has_cache_fields_and_breakpoints() -> None:
    model = _model()
    msgs = [
        SystemMessage(content="you are a wiki agent"),
        HumanMessage(content="hello"),
        AIMessage(content="hi there"),
        HumanMessage(content="tell me about pages"),
    ]
    payload = model._get_request_payload(msgs)

    assert payload["prompt_cache_key"] == "wiki-query"
    assert payload["prompt_cache_retention"] == "24h"

    # System message: string content converted to a marked content array.
    system = payload["messages"][0]
    assert isinstance(system["content"], list)
    assert system["content"][0]["type"] == "text"
    assert system["content"][0]["text"] == "you are a wiki agent"
    assert system["content"][0]["cache_control"] == {"type": "ephemeral", "ttl": "1h"}

    # Last 2 user/assistant messages carry breakpoints.
    marked_roles = [
        m["role"] for m in payload["messages"]
        if isinstance(m["content"], list) and m["content"][-1].get("cache_control")
    ]
    assert marked_roles == ["system", "assistant", "user"]

    # Unmarked messages keep plain string content.
    assert payload["messages"][1]["content"] == "hello"


def test_last_tool_gets_breakpoint() -> None:
    model = _model()
    msgs = [SystemMessage(content="sys"), HumanMessage(content="hi")]
    payload = model._get_request_payload(msgs, tools=[{"type": "function", "function": {"name": "f", "description": "d", "parameters": {"type": "object", "properties": {}}}}])
    assert payload["tools"][-1]["cache_control"] == {"type": "ephemeral", "ttl": "1h"}


def test_glm_model_gets_no_instrumentation() -> None:
    model = _model(model="opencode-go/glm-5.2")
    payload = model._get_request_payload([SystemMessage(content="sys"), HumanMessage(content="hi")])
    assert "prompt_cache_key" not in payload
    assert "prompt_cache_retention" not in payload
    assert payload["messages"][0]["content"] == "sys"


def test_stale_cache_control_stripped_before_re_stamp() -> None:
    model = _model()
    msgs = [
        SystemMessage(
            content=[
                {"type": "text", "text": "sys", "cache_control": {"type": "ephemeral"}}
            ]
        ),
        HumanMessage(content="hi"),
    ]
    payload = model._get_request_payload(msgs)
    # The stale marker is replaced with the fresh one (same shape is fine, but
    # the strip+restamp must not leave duplicates).
    system = payload["messages"][0]
    assert len(system["content"]) == 1
    assert system["content"][0]["cache_control"] == {"type": "ephemeral", "ttl": "1h"}


def test_tool_call_round_trip_payload_builds() -> None:
    """A tool-calling conversation must still build and stamp cleanly."""
    model = _model()
    msgs = [
        SystemMessage(content="sys"),
        HumanMessage(content="search for X"),
        AIMessage(
            content="",
            tool_calls=[{"name": "wiki_search", "args": {"query": "X"}, "id": "call_1", "type": "tool_call"}],
        ),
        ToolMessage(content="found pages", tool_call_id="call_1"),
        HumanMessage(content="summarize"),
    ]
    payload = model._get_request_payload(msgs)
    assert payload["prompt_cache_key"] == "wiki-query"
    # Tool message untouched, last user message stamped.
    assert payload["messages"][3]["role"] == "tool"
    assert payload["messages"][-1]["content"][-1]["cache_control"]["type"] == "ephemeral"
