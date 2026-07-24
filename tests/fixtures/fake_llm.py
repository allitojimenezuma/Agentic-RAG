"""FakeChatModel — returns scripted AIMessages with tool_calls for deterministic agent tests."""

from __future__ import annotations

from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import Runnable


class ResponseState:
    """Shared mutable state for tracking which response to return.

    Because create_agent calls bind_tools (model_copy) on each agent loop iteration,
    a simple _idx field on the model gets reset. This class survives deep-copy.
    """

    def __init__(self, responses: list[AIMessage]):
        self.responses = responses
        self.idx = 0


class ScriptedChatModel(BaseChatModel):
    """A chat model that returns pre-scripted AIMessages in order.

    Uses a shared ResponseState so the response index persists across
    bind_tools / model_copy calls made by create_agent's internal loop.
    """

    _state: ResponseState | None = None

    def __init__(self, *, responses: list[AIMessage] | None = None, **kwargs):
        super().__init__(**kwargs)
        if responses is not None:
            self._state = ResponseState(responses)

    @property
    def _llm_type(self) -> str:
        return "scripted-chat-model"

    def bind_tools(
        self,
        tools,
        *,
        tool_choice: str | None = None,
        **kwargs: Any,
    ) -> Runnable:
        """Bind tools — returns a copy preserving the shared response state."""
        model = self.model_copy(deep=True)
        # Preserve shared state reference (survives deep copy of ResponseState)
        model._state = self._state
        return model

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        if self._state is None:
            response = AIMessage(content="[No responses configured]")
        elif self._state.idx < len(self._state.responses):
            response = self._state.responses[self._state.idx]
            self._state.idx += 1
        else:
            response = AIMessage(content="[No more scripted responses]")

        generation = ChatGeneration(message=response)
        return ChatResult(generations=[generation])

    @property
    def _identifying_params(self) -> dict[str, Any]:
        n = len(self._state.responses) if self._state else 0
        return {"num_responses": n}
