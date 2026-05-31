# EverOS Integration

This project includes a small Python adapter for EverOS, a persistent memory
layer for AI-agent workflows.

## Setup

Install the SDK:

```bash
pip install -r requirements.txt
```

Set the API key in your shell or deployment secret store:

```bash
export EVEROS_API_KEY="your_api_key"
```

Do not commit the real API key. EverOS authenticates with a bearer token and the
Python SDK reads `EVEROS_API_KEY` automatically.

## Usage

```python
from integrations.everos_memory import EverOSMemory, build_prompt

memory = EverOSMemory()

context = memory.retrieve_context(
    user_id="user_001",
    query="What should I remember before replying?",
)

prompt = build_prompt("Can we schedule the next review?", context.text)
response = call_your_llm(prompt)

memory.remember_exchange(
    user_id="user_001",
    session_id="session_001",
    user_message="Can we schedule the next review?",
    assistant_message=response,
)
```

For command-line checks:

```bash
python -m integrations.everos_memory --user-id user_001 "meeting preferences"
```

## Agent Loop

1. Search memory before generating a response with `method="hybrid"` and
   `top_k=5`.
2. Add the user and assistant turns after generation.
3. Flush at natural task or conversation boundaries so EverOS can run boundary
   detection and extraction.

```python
memory.flush(user_id="user_001", session_id="session_001")
```

## Agent Memories

Use agent memories when storing agent trajectories, including tool calls and
tool results.

```python
from integrations.everos_memory import EverOSMemory

memory = EverOSMemory()

response = memory.remember_agent_messages(
    user_id="agent_001",
    session_id="task_session_001",
    messages=[
        {
            "role": "user",
            "timestamp": 1711900000000,
            "content": "Find all Python files with TODO comments",
        },
        {
            "role": "assistant",
            "timestamp": 1711900001000,
            "content": "I will scan the repository for TODO comments.",
        },
    ],
)
print(response)

memory.flush_agent_memory(
    user_id="agent_001",
    session_id="task_session_001",
)
```

The direct SDK equivalent is:

```python
from everos import EverOS

client = EverOS()
agent = client.v1.memories.agent

response = agent.add(
    user_id="agent_001",
    session_id="task_session_001",
    messages=[
        {
            "role": "user",
            "timestamp": 1711900000000,
            "content": "Find all Python files with TODO comments",
        }
    ],
)
print(response)
```

For complex multi-step context gathering, EverOS supports `method="agentic"`,
but the API reference recommends trying `hybrid` first and falling back to it
when agentic retrieval times out.
