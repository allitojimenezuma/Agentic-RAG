"""LLM factory — builds a ChatOpenAI instance from Settings.

Includes a slim, vendor-agnostic passthrough for *reasoning models* (DeepSeek-
style thinking modes served over the OpenAI-compatible API): such models stream
the thinking text as ``delta.reasoning_content`` and require it to be sent back
unchanged on every subsequent turn. ``langchain-openai`` drops the field in
both directions, so the wrapper below captures it into
``message.additional_kwargs`` when responses arrive and re-injects it into the
outgoing request payload when the history contains it. For ordinary models the
field is never present and every hook is a no-op.

This replaces an earlier class that also injected vendor-specific prompt-cache
fields (``prompt_cache_key`` / ``cache_control`` breakpoints) for a single
third-party proxy. That instrumentation was removed: it monkeypatched private
``langchain-openai`` methods, could not be exercised without the proxy, and its
supposed savings were speculative.
"""

from __future__ import annotations

import logging
from typing import Any

from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage
from langchain_openai import ChatOpenAI

logger = logging.getLogger(__name__)


def _reasoning_delta_from_chunk(chunk: dict) -> str:
    """Extract the ``reasoning_content`` delta from a raw streaming chunk.

    Reasoning providers stream the thinking text as ``delta.reasoning_content``
    alongside the answer content. Handles both the plain SSE chunk dict and the
    ``beta.chat.completions.stream`` wrapper (``{"chunk": {...}}``).
    langchain-openai drops this field when converting deltas to message chunks,
    so we pull it out here and stash it in ``additional_kwargs`` so it survives
    the turn.
    """
    if not isinstance(chunk, dict):
        return ""
    raw = chunk.get("chunk") if "chunk" in chunk else chunk
    if not isinstance(raw, dict):
        return ""
    choices = raw.get("choices") or []
    if not choices:
        return ""
    delta = (choices[0] or {}).get("delta") or {}
    rc = delta.get("reasoning_content")
    return rc if isinstance(rc, str) else ""


def _stamp_reasoning(payload: dict, messages: list[BaseMessage]) -> None:
    """Re-inject assistant ``reasoning_content`` into the outgoing payload.

    Reasoning models require the thinking text of every previous assistant turn
    to be passed back unchanged; langchain-openai's ``_convert_message_to_dict``
    only forwards tool_calls/function_call/audio, so without this the history
    silently loses it and the upstream rejects the request with
    ``reasoning_content in the thinking mode must be passed back``. The
    chat/completions ``messages`` list is built 1:1 from ``messages``, so a zip
    is safe; other payload shapes (responses API) are left untouched.
    """
    payload_messages = payload.get("messages")
    if not isinstance(payload_messages, list) or len(payload_messages) != len(messages):
        return
    for out, msg in zip(payload_messages, messages):
        if isinstance(msg, AIMessage):
            reasoning = msg.additional_kwargs.get("reasoning_content")
            if reasoning:
                out["reasoning_content"] = reasoning


class ReasoningPassthroughChat(ChatOpenAI):
    """ChatOpenAI that preserves ``reasoning_content`` across turns.

    Adds no network behavior: every hook is a no-op for models that never emit
    a ``reasoning_content`` field. Kept deliberately small — this is the
    minimal subclass needed to make thinking-mode models work through
    langchain-openai, not a vehicle for provider-specific tweaks.
    """

    def _get_request_payload(
        self,
        input_: Any,
        *,
        stop: list[str] | None = None,
        **kwargs: Any,
    ) -> dict:
        payload = super()._get_request_payload(input_, stop=stop, **kwargs)
        try:
            _stamp_reasoning(payload, self._convert_input(input_).to_messages())
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("reasoning_content passthrough skipped: %s", exc)
        return payload

    def _create_chat_result(
        self,
        response: Any,
        generation_info: dict | None = None,
    ) -> Any:
        """Capture ``reasoning_content`` from non-streamed responses."""
        result = super()._create_chat_result(response, generation_info=generation_info)
        response_dict = (
            response if isinstance(response, dict) else response.model_dump(warnings=False)
        )
        choices = response_dict.get("choices") or []
        for gen, choice in zip(result.generations, choices):
            if not isinstance(gen.message, AIMessage):
                continue
            reasoning = (choice.get("message") or {}).get("reasoning_content")
            if reasoning:
                gen.message.additional_kwargs["reasoning_content"] = reasoning
        return result

    def _convert_chunk_to_generation_chunk(
        self,
        chunk: dict,
        default_chunk_class: Any,
        base_generation_info: dict | None,
    ) -> Any:
        """Stash per-chunk ``reasoning_content`` deltas before langchain drops them."""
        gen = super()._convert_chunk_to_generation_chunk(
            chunk, default_chunk_class, base_generation_info
        )
        if gen is not None and isinstance(gen.message, AIMessageChunk):
            reasoning = _reasoning_delta_from_chunk(chunk)
            if reasoning:
                gen.message.additional_kwargs["reasoning_content"] = reasoning
        return gen

    async def astream(self, input_: Any, config: Any = None, **kwargs: Any) -> Any:
        """Accumulate reasoning deltas so every yielded chunk carries the FULL
        thinking text of the turn (a single chunk only holds one delta)."""
        reasoning: list[str] = []
        async for chunk in super().astream(input_, config=config, **kwargs):
            rc = (chunk.additional_kwargs or {}).get("reasoning_content")
            if rc:
                reasoning.append(rc)
            if reasoning:
                chunk.additional_kwargs["reasoning_content"] = "".join(reasoning)
            yield chunk

    def stream(self, input_: Any, config: Any = None, **kwargs: Any) -> Any:
        """Sync twin of :meth:`astream` — same reasoning accumulation."""
        reasoning: list[str] = []
        for chunk in super().stream(input_, config=config, **kwargs):
            rc = (chunk.additional_kwargs or {}).get("reasoning_content")
            if rc:
                reasoning.append(rc)
            if reasoning:
                chunk.additional_kwargs["reasoning_content"] = "".join(reasoning)
            yield chunk


def get_model(settings: Any) -> ChatOpenAI:
    """Create a ChatOpenAI instance from settings.

    Uses explicit base_url and api_key to avoid provider auto-detection issues.
    The returned model preserves ``reasoning_content`` across turns when a
    thinking-mode provider is configured (see :class:`ReasoningPassthroughChat`).

    Args:
        settings: Settings instance (openai_model, openai_api_key, openai_base_url).
    """
    return ReasoningPassthroughChat(
        model=settings.openai_model,
        api_key=settings.openai_api_key,
        base_url=settings.openai_base_url,
        temperature=0,
    )