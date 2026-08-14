"""L3 answer-quality eval dataset (T1) — curated queries with gold answers.

WHAT IT IS:
- A small, reviewable set of (question, gold answer, expected-slug) triples used
  by the DeepEval real-LLM answer-quality tier
  (``tests/levels/level3/test_answer_quality_real_llm.py``).
- Questions reuse the retrieval ground truth from ``CURATED_QUERIES``; each gold
  answer is hand-written from the committed eval corpus pages (see ``sources``
  per row). Hand-written answers are LOW-CONFIDENCE until human-reviewed — the
  DeepEval tier uses them as ``expected_output`` for Contextual Recall and as a
  relevancy reference, and reports scores report-only until baselines are
  pinned (D4 of the test-suite plan).

WHY IT MATTERS:
- This is the "exam" for L3 answer quality: Faithfulness needs no gold answer,
  but Contextual Recall and answer-relevancy references do. Keep it small
  (~10 rows) — judge cost scales with rows x metrics.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class EvalQuestion:
    """One curated L3 question with its hand-written gold answer."""

    question: str
    gold_answer: str
    expected_slug: str  # ground-truth page the answer must be grounded in
    sources: tuple[str, ...]  # eval_wiki pages the gold answer derives from


EVAL_QUESTIONS: tuple[EvalQuestion, ...] = (
    EvalQuestion(
        question="Apple's matrix math framework for its chips",
        gold_answer=(
            "MLX is a machine learning framework developed by Apple for efficient "
            "training and inference on Apple Silicon. It uses the Apple Metal API "
            "for GPU acceleration and unified memory for faster data access."
        ),
        expected_slug="entities/mlx",
        sources=("entities/mlx.md",),
    ),
    EvalQuestion(
        question="high-performance ARM processors by Apple",
        gold_answer=(
            "Apple Silicon is Apple's family of ARM-based system-on-a-chip "
            "processors used in Macs, iPads, and iPhones, featuring unified "
            "memory, high-performance GPUs, and the Apple Neural Engine."
        ),
        expected_slug="entities/apple-silicon",
        sources=("entities/apple-silicon.md",),
    ),
    EvalQuestion(
        question="Microsoft cloud platform",
        gold_answer=(
            "Azure is Microsoft's cloud computing platform offering compute, "
            "storage, and AI services. It hosted the LLMs used for tool calling "
            "in the BHS Corrugated Spain project."
        ),
        expected_slug="entities/azure",
        sources=("entities/azure.md",),
    ),
    EvalQuestion(
        question="how do neural networks learn from examples",
        gold_answer=(
            "Machine learning is a subset of AI that enables systems to learn "
            "and improve from experience without explicit programming, with "
            "supervised, unsupervised, and reinforcement learning as the main types."
        ),
        expected_slug="concepts/machine-learning",
        sources=("concepts/machine-learning.md",),
    ),
    EvalQuestion(
        question="calling functions and tools from an LLM",
        gold_answer=(
            "Tool calling lets an LLM invoke functions and external APIs, e.g. "
            "fetching logs and restarting services, as done with Azure-hosted "
            "LLMs in the BHS Corrugated Spain project."
        ),
        expected_slug="concepts/tool-calling",
        sources=("concepts/tool-calling.md",),
    ),
    EvalQuestion(
        question="quantized fine tuning of language models",
        gold_answer=(
            "QLoRA is an efficient fine-tuning method combining quantization "
            "with Low-Rank Adaptation to cut memory and compute; it enables "
            "fine-tuning large models on a single consumer GPU."
        ),
        expected_slug="concepts/llm-fine-tuning-with-qlora",
        sources=("concepts/llm-fine-tuning-with-qlora.md",),
    ),
    EvalQuestion(
        question="real estate ownership on a blockchain",
        gold_answer=(
            "Real estate tokenization converts property into blockchain tokens "
            "representing tradable ownership shares, enabling fractional "
            "ownership and 24/7 trading — implemented on Polygon with over "
            "EUR 2.2 million in listed assets."
        ),
        expected_slug="concepts/real-estate-tokenization",
        sources=("concepts/real-estate-tokenization.md",),
    ),
    EvalQuestion(
        question="design philosophy integrating safety guardrails into AI systems from the outset",
        gold_answer=(
            "Safe-by-design AI prevents irreversible autonomous actions, enforces "
            "deterministic checks for critical operations, and keeps a "
            "human-in-the-loop for high-impact decisions."
        ),
        expected_slug="concepts/safe-by-design-ai",
        sources=("concepts/safe-by-design-ai.md",),
    ),
    EvalQuestion(
        question="general purpose interpreted programming language",
        gold_answer=(
            "Python is a high-level, general-purpose programming language known "
            "for readability and versatility, widely used in machine learning, "
            "web development, and workflow automation."
        ),
        expected_slug="entities/python",
        sources=("entities/python.md",),
    ),
    EvalQuestion(
        question="automatic workflows driven by AI",
        gold_answer=(
            "AI workflow automation uses artificial intelligence to automate "
            "complex workflows; at BHS Corrugated Spain it automated Jira "
            "incident diagnosis, cutting manual ticket handling by 40%."
        ),
        expected_slug="concepts/ai-workflow-automation",
        sources=("concepts/ai-workflow-automation.md",),
    ),
)
