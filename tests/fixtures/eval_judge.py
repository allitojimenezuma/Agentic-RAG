"""Calibrated real-LLM judge harness (T1 — deterministic harness, real model).

The judges are hand-rolled prompts over ``langchain-openai`` (no ragas, no
extra dependencies). Every prompt embeds fixed few-shot anchors (module string
constants, reviewable) so scores are calibrated: a fully grounded answer must
score >= 0.7, a fabricated/off-topic answer <= 0.4.

``judge_model()`` reads ``Settings()`` exactly as the CLI does and returns None
when no API key is configured — callers/tests must skip in that case.
Deterministic tests never call the real model (stub-model round-trip only).
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from agentic_rag.config import Settings

# ---------------------------------------------------------------------------
# Few-shot anchors (fixed, reviewable strings — calibration must not drift).
# ---------------------------------------------------------------------------

FAITHFULNESS_ANCHORS = """\
Scoring rules (few-shot anchors):

Example 1 — directly supported:
Question: What is MLX?
Context: "MLX is a machine learning framework developed by Apple."
Answer: "MLX is Apple's machine learning framework."
Score: 1.0 — the answer claim is directly supported by the context.

Example 2 — partially supported:
Question: Where does Álvaro work?
Context: "He interned at BHS Corrugated Spain."
Answer: "Álvaro works as a backend engineer at BHS Corrugated Spain."
Score: 0.5 — the company is supported, but the role is not present in context.

Example 3 — unsupported / contradicted:
Question: Who developed MLX?
Context: "MLX is a machine learning framework developed by Apple."
Answer: "MLX was developed by Google."
Score: 0.0 — the answer claim is contradicted by the context.

An answer claim must be explicitly supported by the context to score above
0.5. Absent or contradicted claims score 0.0. Do not reward general
plausibility; score the answer strictly against the provided context.
"""

RELEVANCY_ANCHORS = """\
Scoring rules (few-shot anchors):

Example 1 — directly on-topic:
Question: What is MLX?
Answer: "MLX is a machine learning framework developed by Apple."
Score: 1.0 — the answer directly addresses the question.

Example 2 — partially relevant:
Question: What is MLX?
Answer: "MLX is used in the mlx-modernbert project."
Score: 0.5 — related to the topic but does not directly answer the question.

Example 3 — off-topic / dodge:
Question: What is MLX?
Answer: "I don't have information about that topic."
Score: 0.0 — the answer does not address the question at all.

Score the extent to which the answer directly addresses the question asked.
Refusal-dodges and off-topic text score 0.0.
"""

FAITHFULNESS_SYSTEM = (
    "You are a strict faithfulness judge. Given a question, an answer, and "
    "retrieved context, score how well the answer is supported by the context "
    "on a 0.0–1.0 scale.\n\n" + FAITHFULNESS_ANCHORS +
    "\n\nRespond ONLY with a JSON object of the form "
    '{"score": <0.0-1.0>, "rationale": "<short reason>"}.'
)

RELEVANCY_SYSTEM = (
    "You are a strict answer-relevancy judge. Given a question and an answer, "
    "score how directly the answer addresses the question on a 0.0–1.0 scale.\n\n"
    + RELEVANCY_ANCHORS +
    "\n\nRespond ONLY with a JSON object of the form "
    '{"score": <0.0-1.0>, "rationale": "<short reason>"}.'
)

# Score bounds used by tests to assert anchor separation.
GROUNDED_MIN: float = 0.7
FABRICATED_MAX: float = 0.4


class FaithfulnessScore(BaseModel):
    """Pydantic-enforced bounds — no score drift outside [0.0, 1.0]."""

    score: float = Field(ge=0.0, le=1.0)
    rationale: str


class RelevancyScore(BaseModel):
    """Pydantic-enforced bounds — no score drift outside [0.0, 1.0]."""

    score: float = Field(ge=0.0, le=1.0)
    rationale: str


def judge_model():
    """Build the real judge ChatOpenAI from ``Settings()``, or None without a key.

    Mirrors the CLI: ``Settings()`` reads ``OPENAI_API_KEY`` /
    ``OPENAI_BASE_URL`` / ``OPENAI_MODEL`` (optionally from a user-supplied
    ``.env``). Temperature 0 for deterministic scoring. Returns None when no
    key is configured (including when ``Settings()`` cannot construct) so
    callers/tests can skip cleanly.
    """
    from langchain_openai import ChatOpenAI

    try:
        settings = Settings()
    except Exception:
        return None
    if not settings.openai_api_key:
        return None
    return ChatOpenAI(
        model=settings.openai_model,
        api_key=settings.openai_api_key,
        base_url=settings.openai_base_url or None,
        temperature=0,
    )


def _judge(model, system: str, user_payload: str, score_model: type) -> BaseModel:
    """Run one judge call with strict-JSON + corrective retry semantics.

    1. Ask for strict JSON with ``response_format={"type": "json_object"}``.
    2. If the endpoint rejects it, retry once WITHOUT the param (local proxies).
    3. Pydantic-parse the raw text; on parse failure, ONE corrective retry that
       re-sends the parse error + raw output.
    4. On a second failure, raise ``RuntimeError`` including the raw output —
       never silently default to a passing score.
    """
    try:
        raw = _invoke(model, system, user_payload, response_format={"type": "json_object"})
    except Exception:
        # Some local proxies reject response_format — retry without it.
        raw = _invoke(model, system, user_payload, response_format=None)

    try:
        return score_model.model_validate_json(raw)
    except Exception as parse_err:
        corrective = (
            f"The previous output was not valid JSON for {score_model.__name__}. "
            f"Parse error: {parse_err}\nRaw output was:\n{raw}\n\n"
            "Return ONLY a valid JSON object of the expected form."
        )
        try:
            raw2 = _invoke(model, system, user_payload, response_format=None)
        except Exception as exc:
            raise RuntimeError(f"Judge model call failed on corrective retry: {exc}") from exc
        try:
            return score_model.model_validate_json(raw2)
        except Exception as parse_err2:
            raise RuntimeError(
                f"Judge returned unparseable output after corrective retry: {parse_err2}\n"
                f"Raw output:\n{raw2}"
            ) from parse_err2


def _invoke(model, system: str, user_payload: str, *, response_format) -> str:
    """Single chat invocation returning the raw response text."""
    from langchain_core.messages import HumanMessage, SystemMessage

    kwargs = {"response_format": response_format} if response_format is not None else {}
    response = model.invoke(
        [SystemMessage(content=system), HumanMessage(content=user_payload)], **kwargs
    )
    return response.content


def judge_faithfulness(question: str, answer: str, contexts: list[str]) -> FaithfulnessScore:
    """Score how well ``answer`` is supported by ``contexts`` (real model)."""
    model = judge_model()
    if model is None:
        raise RuntimeError("judge_faithfulness requires OPENAI_API_KEY (skip test instead)")
    context_block = "\n\n".join(f"[{i + 1}] {c}" for i, c in enumerate(contexts))
    payload = (
        f"Question:\n{question}\n\n"
        f"Answer:\n{answer}\n\n"
        f"Context:\n{context_block}"
    )
    return _judge(model, FAITHFULNESS_SYSTEM, payload, FaithfulnessScore)


def judge_relevancy(question: str, answer: str) -> RelevancyScore:
    """Score how directly ``answer`` addresses ``question`` (real model)."""
    model = judge_model()
    if model is None:
        raise RuntimeError("judge_relevancy requires OPENAI_API_KEY (skip test instead)")
    payload = f"Question:\n{question}\n\nAnswer:\n{answer}"
    return _judge(model, RELEVANCY_SYSTEM, payload, RelevancyScore)
