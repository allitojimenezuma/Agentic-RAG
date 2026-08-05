# Fix Agent Plan: General Agent with Command Execution

## Overview

Create a LangChain agent with bash command execution using `create_agent()`, `ShellTool` from `langchain_community`, and `HumanInTheLoopMiddleware` for safety.

## Verified Components

| Component | Source | Status |
|-----------|--------|--------|
| `create_agent()` | langchain-fundamentals skill | ✅ Supported |
| `ShellTool` | langchain_community.tools.shell.tool | ✅ Verified |
| `HumanInTheLoopMiddleware` | langchain-middleware skill | ✅ Supported |
| `@wrap_tool_call` | langchain-middleware skill | ✅ Supported |
| `checkpointer` + `thread_id` | Both skills | ✅ Required for HITL |

## Implementation

### Step 1: Install Dependencies

```bash
pip install langchain langchain-community langchain-openai
```

### Step 2: Define Custom Shell Tool

Use `@tool` decorator from skill to wrap `ShellTool` with validation:

```python
from langchain_core.tools import tool
from langchain_community.tools import ShellTool

shell = ShellTool()

@tool
def execute_command(command: str) -> str:
    """Execute a shell command safely. Use for file operations, system queries, automation.

    Args:
        command: Shell command to execute (e.g., 'ls -la', 'cat file.txt')
    """
    return shell.invoke({"commands": command})
```

### Step 3: Create Agent with HITL

Use `create_agent()` + `HumanInTheLoopMiddleware` from skills:

```python
from langchain.agents import create_agent
from langchain.agents.middleware import HumanInTheLoopMiddleware
from langgraph.checkpoint.memory import MemorySaver

# Dangerous patterns requiring human approval
DANGEROUS_TOOLS = {
    "execute_command": {
        "allowed_decisions": ["approve", "edit", "reject"]
    }
}

agent = create_agent(
    model="anthropic:claude-sonnet-4-5",
    tools=[execute_command],
    checkpointer=MemorySaver(),  # Required for HITL
    system_prompt="""You are a helpful assistant with shell access.

    Capabilities:
    - Execute bash commands
    - Read/write files
    - System queries

    Guidelines:
    - Confirm before destructive operations
    - Prefer read-only commands when possible
    - Report errors clearly
    """,
    middleware=[
        HumanInTheLoopMiddleware(interrupt_on=DANGEROUS_TOOLS)
    ],
)
```

### Step 4: Run Agent with Interrupt Handling

From langchain-middleware skill:

```python
from langgraph.types import Command

config = {"configurable": {"thread_id": "user-123"}}

# Step 1: Agent runs until tool call
result1 = agent.invoke(
    {"messages": [{"role": "user", "content": "List files in /tmp"}]},
    config=config
)

# Check for interrupt
if "__interrupt__" in result1:
    print(f"Waiting for approval: {result1['__interrupt__']}")

# Step 2: Human approves
result2 = agent.invoke(
    Command(resume={"decisions": [{"type": "approve"}]}),
    config=config
)

# Step 3: Human rejects with feedback
result3 = agent.invoke(
    Command(resume={"decisions": [{"type": "reject", "feedback": "Command too dangerous"}]}),
    config=config
)

# Step 4: Human edits command before approval
result4 = agent.invoke(
    Command(resume={"decisions": [{
        "type": "edit",
        "edited_action": {
            "name": "execute_command",
            "args": {"command": "ls -la /tmp"}  # Safer version
        }
    }]}),
    config=config
)
```

### Step 5: Add Custom Middleware for Logging

From langchain-middleware skill:

```python
from langchain.agents.middleware import wrap_tool_call, before_model, after_model

@wrap_tool_call
def log_commands(request, handler):
    """Log all shell commands before execution."""
    if request.tool_call["name"] == "execute_command":
        print(f"[AUDIT] Command: {request.tool_call['args']['command']}")
    return handler(request)

@before_model
def log_reasoning(state, runtime):
    """Log agent reasoning steps."""
    msg_count = len(state['messages'])
    print(f"[TRACE] Step {msg_count}")

agent = create_agent(
    model="anthropic:claude-sonnet-4-5",
    tools=[execute_command],
    checkpointer=MemorySaver(),
    system_prompt="...",
    middleware=[
        HumanInTheLoopMiddleware(interrupt_on=DANGEROUS_TOOLS),
        log_commands,
        log_reasoning,
    ],
)
```

### Step 6: Restrict Commands with Custom Middleware

Use `@wrap_tool_call` to filter dangerous commands without HITL:

```python
@wrap_tool_call
def block_rm_rf(request, handler):
    """Block rm -rf / without human approval."""
    if request.tool_call["name"] == "execute_command":
        cmd = request.tool_call["args"]["command"]
        if "rm -rf /" in cmd or "rm -rf ~" in cmd:
            return "BLOCKED: Dangerous command rejected"
    return handler(request)

agent = create_agent(
    model="anthropic:claude-sonnet-4-5",
    tools=[execute_command],
    middleware=[
        block_rm_rf,  # Filter first
        HumanInTheLoopMiddleware(interrupt_on=DANGEROUS_TOOLS),  # Then HITL
    ],
)
```

### Step 7: Full Example

```python
from langchain.agents import create_agent
from langchain.agents.middleware import HumanInTheLoopMiddleware, wrap_tool_call
from langchain_core.tools import tool
from langchain_community.tools import ShellTool
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command

# Shell tool
shell = ShellTool()

@tool
def execute_command(command: str) -> str:
    """Execute a shell command. Use for file ops, system queries, automation.

    Args:
        command: Shell command to execute
    """
    return shell.invoke({"commands": command})

# Middleware
@wrap_tool_call
def audit_log(request, handler):
    if request.tool_call["name"] == "execute_command":
        print(f"[AUDIT] {request.tool_call['args']['command']}")
    return handler(request)

# Agent
agent = create_agent(
    model="anthropic:claude-sonnet-4-5",
    tools=[execute_command],
    checkpointer=MemorySaver(),
    system_prompt="You are a helpful assistant with shell access. Be careful with destructive commands.",
    middleware=[
        audit_log,
        HumanInTheLoopMiddleware(
            interrupt_on={
                "execute_command": {"allowed_decisions": ["approve", "edit", "reject"]}
            }
        ),
    ],
)

# Run
config = {"configurable": {"thread_id": "session-1"}}

result = agent.invoke(
    {"messages": [{"role": "user", "content": "Delete all files in /tmp/test"}]},
    config=config
)

# Handle interrupt
if "__interrupt__" in result:
    # Approve
    result = agent.invoke(
        Command(resume={"decisions": [{"type": "approve"}]}),
        config=config
    )

print(result["messages"][-1].content)
```

## Guardrails Summary

| Guardrail | Implementation | Skill Reference |
|-----------|---------------|-----------------|
| Human approval | `HumanInTheLoopMiddleware` | langchain-middleware |
| Command filtering | `@wrap_tool_call` | langchain-middleware |
| Audit logging | `@wrap_tool_call` | langchain-middleware |
| Edit before approve | `Command(resume={"type": "edit"})` | langchain-middleware |
| Reject with feedback | `Command(resume={"type": "reject"})` | langchain-middleware |
| Conversation memory | `checkpointer=MemorySaver()` | langchain-fundamentals |
| Thread isolation | `config={"configurable": {"thread_id": "..."}}` | langchain-fundamentals |

## References

- [langchain-fundamentals skill](.agents/skills/langchain-fundamentals/SKILL.md)
- [langchain-middleware skill](.agents/skills/langchain-middleware/SKILL.md)
- [ShellTool source](https://github.com/langchain-ai/langchain-community/blob/main/libs/community/langchain_community/tools/shell/tool.py)
