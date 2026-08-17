"""Unit tests for the reasoning-mode passthrough in agents/llm.py.

Covers the one behavior the model factory guarantees beyond a plain
ChatOpenAI: ``reasoning_content`` (thinking-mode text) is captured from
responses and re-injected into the outgoing request payload, so thinking-mode
models served over the OpenAI-compatible API work across turns. All other
provider-specific behavior (prompt-cache instrumention for one third-party
proxy) was removed in the dead-weight cleanup; these tests pin what remains.
No network calls are made — the payload/result builders are exercised with
crafted responses, mirroring how the old cache tests ran.
"""

from __future__ import annotations

from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    HumanMessage,
    SystemMessage,
)
from langchain_core.outputs import ChatGenerationChunk
from langchain_openai import ChatOpenAI

from agentic_rag.agents.llm import ReasoningPassthroughChat, get_model


class _Settings:
    openai_api_key = "sk-test"
    openai_base_url = "https://api.openai.com/v1"
    openai_model = "deepseek-v4-flash"


def _model() -> ReasoningPassthroughChat:
    return ReasoningPassthroughChat(
        model="deepseek-v4-flash",
        api_key="sk-test",
        base_url="https://api.openai.com/v1",
        temperature=0,
    )


def test_get_model_returns_chatopenai_subclass() -> None:
    model = get_model(_Settings())
    assert isinstance(model, ChatOpenAI)
    assert model.temperature == 0
    # Explicit base_url/api_key from settings, no auto-detection surprises.
    assert model.openai_api_base == "https://api.openai.com/v1"
    assert model.openai_api_key.get_secret_value() == "sk-test"


def test_non_streaming_response_captures_reasoning_content() -> None:
    model = _model()
    response = {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": "final answer",
                    "reasoning_content": "let me think",
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
    }
    result = model._create_chat_result(response, generation_info={})
    msg = result.generations[0].message
    assert isinstance(msg, AIMessage)
    assert msg.additional_kwargs["reasoning_content"] == "let me think"


def test_reasoning_content_reinjected_into_outgoing_payload() -> None:
    model = _model()
    msgs = [
        SystemMessage(content="sys"),
        HumanMessage(content="hi"),
        AIMessage(
            content="final answer",
            additional_kwargs={"reasoning_content": "let me think"},
        ),
    ]
    payload = model._get_request_payload(msgs)
    assert payload["messages"][2]["reasoning_content"] == "let me think"


def test_no_reasoning_content_when_absent() -> None:
    model = _model()
    msgs = [
        SystemMessage(content="sys"),
        HumanMessage(content="hi"),
        AIMessage(content="plain answer"),
    ]
    payload = model._get_request_payload(msgs)
    assert "reasoning_content" not in payload["messages"][2]


def test_streaming_chunk_extracts_reasoning_delta() -> None:
    model = _model()
    gen = model._convert_chunk_to_generation_chunk(
        {
            "choices": [
                {
                    "delta": {"content": "", "reasoning_content": "Think"},
                    "index": 0,
                    "finish_reason": None,
                }
            ]
        },
        AIMessageChunk,
        {},
    )
    assert gen is not None
    assert gen.message.additional_kwargs["reasoning_content"] == "Think"


async def test_astream_accumulates_reasoning_across_chunks() -> None:
    class _Chunked(ReasoningPassthroughChat):
        async def _astream(self, messages, stop=None, run_manager=None, **kwargs):
            for part in ("Think", "ing", " hard"):
                yield ChatGenerationChunk(
                    message=AIMessageChunk(
                        content="",
                        additional_kwargs={"reasoning_content": part},
                    )
                )

    model = _Chunked(
        model="deepseek-v4-flash", api_key="sk-test", base_url="https://api.openai.com/v1", temperature=0
    )
    chunks = [c async for c in model.astream([HumanMessage(content="hi")])]
    # Each yielded chunk carries the reasoning accumulated SO FAR (deltas come
    # split across chunks) and the streamed result appends one final empty
    # usage/marker chunk stamped with the full text, which is what langgraph's
    # stream merge stores in the conversation history.
    assert [c.additional_kwargs.get("reasoning_content") for c in chunks] == [
        "Think",
        "Thinking",
        "Thinking hard",
        "Thinking hard",
    ]