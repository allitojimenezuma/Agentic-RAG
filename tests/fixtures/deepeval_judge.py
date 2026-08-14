"""DeepEval judge for the L3 answer-quality tier (T1 — deterministic harness, real model).

WHAT IT IS:
- ``JsonObjectLiteLLM`` — a ``LiteLLMModel`` subclass that translates
  DeepEval's ``json_schema`` response_format into ``json_object``. The project's
  OpenAI-compatible proxy (``OPENAI_BASE_URL``, e.g. Console Go / OpenRouter)
  REJECTS ``json_schema`` but accepts ``json_object`` — without this shim every
  DeepEval metric call fails at the provider. Parsing is inherited from
  ``LiteLLMModel._parse_response`` (JSON -> pydantic schema).
- ``deepeval_judge()`` — builds the judge from ``Settings()`` exactly like
  the retired hand-rolled harness did (reads ``OPENAI_API_KEY`` /
  ``OPENAI_BASE_URL`` / ``OPENAI_MODEL``, temperature 0, optional ``.env``).
  Returns None without a key so tests can skip cleanly.

WHY IT MATTERS:
- This is the LLM-as-judge behind Faithfulness / Answer Relevancy / Contextual
  Recall in ``tests/levels/level3/test_answer_quality_real_llm.py``. It makes
  the L3 real-model tier work against ANY OpenAI-compatible endpoint, not just
  api.openai.com.
"""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel

from deepeval.models import LiteLLMModel


class JsonObjectLiteLLM(LiteLLMModel):
    """LiteLLMModel that always requests ``json_object`` (json_schema unsupported)."""

    def _generate(self, prompt: str, schema: Optional[BaseModel] = None):
        from litellm import completion

        params = self._completion_params(self._build_content(prompt))
        params["response_format"] = {"type": "json_object"}
        response = completion(**params)
        return self._parse_response(response, schema)

    async def _a_generate(self, prompt: str, schema: Optional[BaseModel] = None):
        from litellm import acompletion

        params = self._completion_params(self._build_content(prompt))
        params["response_format"] = {"type": "json_object"}
        response = await acompletion(**params)
        return self._parse_response(response, schema)


def deepeval_judge():
    """Build the DeepEval judge from ``Settings()``, or None without a key.

    Mirrors the retired hand-rolled judge harness: reads ``OPENAI_API_KEY`` /
    ``OPENAI_BASE_URL`` / ``OPENAI_MODEL`` (optionally from ``.env``).
    Temperature 0 for deterministic scoring; max_tokens generous enough for the
    thinking-style models the proxy serves. Returns None when no key is
    configured so callers/tests can skip cleanly.
    """
    try:
        from agentic_rag.config import Settings

        settings = Settings()
    except Exception:
        return None
    if not settings.openai_api_key:
        return None
    return JsonObjectLiteLLM(
        model=f"openai/{settings.openai_model}",
        api_key=settings.openai_api_key,
        base_url=settings.openai_base_url or None,
        temperature=0,
        generation_kwargs={"max_tokens": 8192},
    )
