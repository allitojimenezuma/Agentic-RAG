"""LLM helper — creates ChatOpenAI with explicit base_url/api_key.

When the base URL is the OpenCode Go gateway (``opencode.ai/zen/go``), the
returned model adds prompt-cache instrumentation. The gateway auto-caches the
request prefix, but only for ~5 minutes and with no session-scoped key, so
nearly every turn pays full input price. Sending the same fields OpenCode CLI
sends — ``prompt_cache_key`` (session-scoped), ``prompt_cache_retention``
(``"24h"``) and Anthropic-style ``cache_control`` breakpoints — makes the
cache survive long sessions and keeps the stable prefix (system prompt + tool
schemas) reusable as the conversation grows. Cache reads are 5–120× cheaper
than input tokens on opencode-go models (50× for deepseek-v4-flash).
"""

from __future__ import annotations

import logging
import os
from typing import Any

from langchain_openai import ChatOpenAI

logger = logging.getLogger(__name__)

# Anthropic-style cache breakpoint marker. "1h" is the documented ceiling for
# cache_control.ttl on the gateway; "24h" is the max prompt_cache_retention
# value in its schema (enum["in_memory", "24h"]).
_CACHE_CONTROL_MARKER = {"type": "ephemeral", "ttl": "1h"}
_CACHE_RETENTION = "24h"

# Models whose downstream API rejects Anthropic-style cache markers (the
# gateway does NOT strip them and the API errors with "Extra inputs are not
# permitted"). Mirror of the opencode-go-cache extension's known list.
_UNSUPPORTED_CACHE_MODEL_PATTERNS = ("glm", "zhipu")

# Opt-out for the whole instrumentation (env: OPENCODE_GO_CACHE=0).
_CACHE_ENABLED = os.environ.get("OPENCODE_GO_CACHE", "1") != "0"


def _is_opencode_go(base_url: str) -> bool:
    """True if the base URL points at the OpenCode Go gateway."""
    return "opencode.ai/zen/go" in base_url


def _unsupported_for_cache(model: str) -> bool:
    """Models that reject cache instrumentation outright."""
    model = model.lower()
    return any(p in model for p in _UNSUPPORTED_CACHE_MODEL_PATTERNS)


def _strip_stale_cache_control(payload: dict) -> None:
    """Remove leftover ephemeral cache_control markers so breakpoints land
    exactly where we want them this turn (tool results can round-trip content
    that carried a marker from a previous turn)."""

    def visit(node: Any) -> None:
        if isinstance(node, list):
            for item in node:
                visit(item)
            return
        if not isinstance(node, dict):
            return
        cc = node.get("cache_control")
        if isinstance(cc, dict) and cc.get("type") == "ephemeral":
            del node["cache_control"]
        for key, value in list(node.items()):
            if key == "cache_control":
                continue
            visit(value)

    for key in ("messages", "system", "tools"):
        if key in payload:
            visit(payload[key])


def _stamp_message(message: dict, marker: dict) -> bool:
    """Add a cache breakpoint to one message dict. Handles both the
    string-content form (common in openai-completions) and the array form.
    Returns True iff a marker was placed."""
    content = message.get("content")
    if isinstance(content, str):
        if not content:
            return False
        message["content"] = [{"type": "text", "text": content, "cache_control": marker}]
        return True
    if isinstance(content, list) and content:
        for part in reversed(content):
            if not isinstance(part, dict):
                continue
            if part.get("cache_control"):
                return True
            if part.get("type") in ("text", "image", "image_url", "tool_use", "tool_result"):
                part["cache_control"] = marker
                return True
    return False


def _apply_conversation_breakpoints(messages: list[dict], marker: dict) -> None:
    """OpenCode CLI's "2 system + 2 final" strategy: breakpoints on up to 2
    leading system/developer messages and the last 2 user/assistant messages,
    so the stable prefix stays cached as the tail changes every turn."""
    stamped = 0
    for msg in messages:
        if msg.get("role") not in ("system", "developer"):
            break
        if _stamp_message(msg, marker):
            stamped += 1
            if stamped >= 2:
                break
    stamped = 0
    for msg in reversed(messages):
        if msg.get("role") not in ("user", "assistant"):
            continue
        if _stamp_message(msg, marker):
            stamped += 1
            if stamped >= 2:
                break


class _OpencodeGoCachedChat(ChatOpenAI):
    """ChatOpenAI that instruments prompt caching for the OpenCode Go gateway.

    Adds three things to every request payload (same strategy OpenCode CLI
    uses, ported from the pi-opencode-go-cache extension):

    1. ``prompt_cache_key`` — scope the cache to a stable per-agent key so it
       survives across many turns/runs instead of only the prefix hash.
    2. ``prompt_cache_retention: "24h"`` — keep the cache alive for a day
       instead of the gateway's ~5-minute default.
    3. ``cache_control`` breakpoints — on up to 2 system messages, the last 2
       user/assistant messages, and the last tool, so the cache stays useful
       as the conversation grows.

    Instrumentation is best-effort: any failure just sends the request
    unchanged (never breaks the LLM call).
    """

    prompt_cache_key: str = "default"

    def _get_request_payload(
        self,
        input_: Any,
        *,
        stop: list[str] | None = None,
        **kwargs: Any,
    ) -> dict:
        payload = super()._get_request_payload(input_, stop=stop, **kwargs)
        if not _CACHE_ENABLED or _unsupported_for_cache(self.model_name):
            return payload
        try:
            payload["prompt_cache_key"] = self.prompt_cache_key[:64]
            payload["prompt_cache_retention"] = _CACHE_RETENTION
            _strip_stale_cache_control(payload)
            messages = payload.get("messages")
            if isinstance(messages, list) and messages:
                _apply_conversation_breakpoints(messages, _CACHE_CONTROL_MARKER)
            tools = payload.get("tools")
            if isinstance(tools, list) and tools and isinstance(tools[-1], dict):
                tools[-1]["cache_control"] = _CACHE_CONTROL_MARKER
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("opencode-go prompt-cache instrumentation skipped: %s", exc)
        return payload


def get_model(settings: Any, cache_key: str | None = None) -> ChatOpenAI:
    """Create a ChatOpenAI instance from settings.

    Uses explicit base_url and api_key to avoid provider auto-detection issues.
    When the base URL is the OpenCode Go gateway, returns a subclass that adds
    prompt-cache instrumentation.

    Args:
        settings: Settings instance (openai_model, openai_api_key, openai_base_url).
        cache_key: Stable per-session cache key (<=64 chars). Used both as the
            body ``prompt_cache_key`` and as the ``x-opencode-session`` header
            that the gateway shows as a session in its usage dashboard and
            uses for sticky provider affinity. Use a key that is stable for
            the agent type / conversation so the system prompt and tool
            schemas are reused from cache across turns and runs. Defaults to
            "default" when the gateway is detected.
    """
    if _is_opencode_go(settings.openai_base_url) and _CACHE_ENABLED:
        session_id = (cache_key or "default")[:64]
        return _OpencodeGoCachedChat(
            model=settings.openai_model,
            api_key=settings.openai_api_key,
            base_url=settings.openai_base_url,
            temperature=0,
            prompt_cache_key=session_id,
            default_headers={
                # The Zen gateway reads the session from this header: it
                # populates the session column in the usage dashboard and pins
                # requests to the same upstream provider (x-session-affinity),
                # which is what keeps the upstream prompt cache warm.
                "x-opencode-session": session_id,
                "x-opencode-client": "agentic-rag",
            },
        )
    return ChatOpenAI(
        model=settings.openai_model,
        api_key=settings.openai_api_key,
        base_url=settings.openai_base_url,
        temperature=0,
    )
