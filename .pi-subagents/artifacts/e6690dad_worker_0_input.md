# Task for worker

You are a delegated subagent running from a fork of the parent session. Treat the inherited conversation as reference-only context, not a live thread to continue. Do not continue or answer prior messages as if they are waiting for a reply. Your sole job is to execute the task below and return a focused result for that task using your tools.

Task:
Implement token usage and cost tracking for logging in the agentic RAG project at /Users/alvarojimenezmartinez/Proyectos/LangChain-RAG.

## Requirements
- Track input tokens and output tokens per LLM call
- Calculate cost based on model pricing (input price per token, output price per token)
- Track total tokens, total cost
- Include timing information
- Log all of this per tool call and per agent invocation

## Implementation

### 1. Create `src/agentic_rag/token_tracker.py`
Token usage tracker class:
```python
import time
import logging
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger("agentic_rag.tokens")

# Pricing per model (USD per 1M tokens) — update as needed
MODEL_PRICING = {
    "gpt-4.1-mini": {"input": 0.40, "output": 1.60},
    "gpt-4.1": {"input": 2.00, "output": 8.00},
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "gpt-4o": {"input": 2.50, "output": 10.00},
    "deepseek-v4-flash": {"input": 0.14, "output": 0.28},
    # Add more models as needed
}

@dataclass
class TokenUsage:
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    input_cost: float = 0.0
    output_cost: float = 0.0
    total_cost: float = 0.0
    duration_seconds: float = 0.0

class TokenTracker:
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.pricing = MODEL_PRICING.get(model_name, {"input": 0.0, "output": 0.0})
        self._cumulative = TokenUsage()
        self._call_count = 0
    
    def record_call(self, input_tokens: int, output_tokens: int, duration: float) -> TokenUsage:
        """Record a single LLM call and return its usage."""
        input_cost = (input_tokens / 1_000_000) * self.pricing["input"]
        output_cost = (output_tokens / 1_000_000) * self.pricing["output"]
        total_cost = input_cost + output_cost
        
        usage = TokenUsage(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=input_tokens + output_tokens,
            input_cost=input_cost,
            output_cost=output_cost,
            total_cost=total_cost,
            duration_seconds=duration,
        )
        
        # Update cumulative
        self._cumulative.input_tokens += input_tokens
        self._cumulative.output_tokens += output_tokens
        self._cumulative.total_tokens += input_tokens + output_tokens
        self._cumulative.input_cost += input_cost
        self._cumulative.output_cost += output_cost
        self._cumulative.total_cost += total_cost
        self._cumulative.duration_seconds += duration
        self._call_count += 1
        
        logger.info(
            f"TOKENS: in={input_tokens} out={output_tokens} "
            f"cost=${total_cost:.6f} (${input_cost:.6f}+${output_cost:.6f}) "
            f"duration={duration:.3f}s"
        )
        logger.debug(
            f"CUMULATIVE: {self._call_count} calls, "
            f"in={self._cumulative.input_tokens} out={self._cumulative.output_tokens}, "
            f"total=${self._cumulative.total_cost:.6f}"
        )
        
        return usage
    
    def get_summary(self) -> TokenUsage:
        """Get cumulative usage summary."""
        return self._cumulative
    
    def log_summary(self):
        """Log the final summary."""
        s = self._cumulative
        logger.info(
            f"SESSION SUMMARY: {self._call_count} LLM calls, "
            f"tokens: in={s.input_tokens} out={s.output_tokens} total={s.total_tokens}, "
            f"cost: ${s.total_cost:.6f} (${s.input_cost:.6f}+${s.output_cost:.6f}), "
            f"time: {s.duration_seconds:.3f}s"
        )
```

### 2. Update `src/agentic_rag/middleware/logging.py`
Add token tracking to the tool call middleware. Extract token usage from LLM response metadata:

```python
import logging
import time
from langchain.agents.middleware import wrap_tool_call
from agentic_rag.token_tracker import TokenTracker

logger = logging.getLogger("agentic_rag.tools")

# Global tracker per agent (set when agent is created)
_current_tracker: TokenTracker | None = None

def set_tracker(tracker: TokenTracker):
    global _current_tracker
    _current_tracker = tracker

@wrap_tool_call
def audit_logging_middleware(request, handler):
    """Log every tool call with args, result, duration, and token usage."""
    tool_name = request.tool_call["name"]
    tool_args = request.tool_call.get("args", {})
    
    logger.info(f"TOOL CALL: {tool_name}({tool_args})")
    start_time = time.time()
    
    try:
        result = handler(request)
        duration = time.time() - start_time
        
        # Extract token usage from result if present
        if _current_tracker and hasattr(result, 'response_metadata'):
            usage = result.response_metadata.get('token_usage', {})
            if usage:
                _current_tracker.record_call(
                    input_tokens=usage.get('prompt_tokens', 0),
                    output_tokens=usage.get('completion_tokens', 0),
                    duration=duration,
                )
        
        logger.info(f"TOOL RESULT: {tool_name} completed in {duration:.3f}s")
        logger.debug(f"TOOL OUTPUT: {tool_name} -> {str(result)[:500]}")
        return result
    except Exception as e:
        duration = time.time() - start_time
        logger.error(f"TOOL ERROR: {tool_name} failed after {duration:.3f}s: {e}")
        raise
```

### 3. Update `src/agentic_rag/agents/factory.py`
Create and attach token tracker when building agents:

```python
from agentic_rag.token_tracker import TokenTracker, set_tracker

def build_agent(model, tools, system_prompt, middleware=None, model_name=None):
    tracker = TokenTracker(model_name or "unknown")
    set_tracker(tracker)
    # ... create agent ...
    # Store tracker on agent for later access
    agent._token_tracker = tracker
    return agent
```

### 4. Update `src/agentic_rag/cli.py`
After agent invocation, log the token summary:

```python
from agentic_rag.token_tracker import TokenTracker

# After agent.invoke() completes:
if hasattr(agent, '_token_tracker'):
    agent._token_tracker.log_summary()
```

### 5. Update tests
Add tests for token tracking in `tests/unit/test_logging.py`:
- Test TokenTracker records calls correctly
- Test cumulative tracking
- Test summary calculation
- Test pricing calculation

### 6. Run all tests
After implementing, run `pytest tests/ -v` and ensure all tests pass.

Work from /Users/alvarojimenezmartinez/Proyectos/LangChain-RAG

## Acceptance Contract
Acceptance level: checked
Completion is not accepted from prose alone. End with a structured acceptance report.

Criteria:
- criterion-1: Implement the requested change without widening scope

Required evidence: changed-files, tests-added, commands-run, residual-risks, no-staged-files

Finish with a fenced JSON block tagged `acceptance-report` in this shape:
Use empty arrays when no items apply; array fields contain strings unless object entries are shown.
`criteriaSatisfied[].status` must be exactly one of: satisfied, not-satisfied, not-applicable.
`commandsRun[].result` must be exactly one of: passed, failed, not-run.
`manualNotes` and `notes` are optional strings; an empty string means no note and does not satisfy `manual-notes` evidence.
```acceptance-report
{
  "criteriaSatisfied": [
    {
      "id": "criterion-1",
      "status": "satisfied",
      "evidence": "specific proof"
    }
  ],
  "changedFiles": [
    "src/file.ts"
  ],
  "testsAddedOrUpdated": [
    "test/file.test.ts"
  ],
  "commandsRun": [
    {
      "command": "command",
      "result": "passed",
      "summary": "short result"
    }
  ],
  "validationOutput": [
    "validation output or concise summary"
  ],
  "residualRisks": [
    "none"
  ],
  "noStagedFiles": true,
  "diffSummary": "short description of the diff",
  "reviewFindings": [
    "blocker: file.ts:12 - issue found, or no blockers"
  ],
  "manualNotes": "anything else the parent should know"
}
```