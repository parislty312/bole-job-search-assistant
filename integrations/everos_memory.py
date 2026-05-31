"""EverOS memory helpers for AI-agent style workflows.

The wrapper keeps EverOS optional for the Jekyll site itself while making it
easy for agent scripts to retrieve context, store turns, and flush sessions.
Set EVEROS_API_KEY in the environment before constructing the default client.
"""

from __future__ import annotations

import argparse
import os
import time
from dataclasses import dataclass
from typing import Any, Iterable, Sequence


DEFAULT_MEMORY_TYPES = ("episodic_memory", "profile")


@dataclass(frozen=True)
class MemoryContext:
    """Text prompt context plus the raw EverOS SDK response."""

    text: str
    raw_response: Any


def now_ms() -> int:
    """Return the current Unix timestamp in milliseconds for EverOS messages."""

    return int(time.time() * 1000)


class EverOSMemory:
    """Thin adapter around the EverOS Python SDK.

    The EverOS docs recommend hybrid search as the default retrieval method and
    top_k=5 for chat context, so those are the defaults here.
    """

    def __init__(self, client: Any | None = None) -> None:
        if client is None:
            if not os.getenv("EVEROS_API_KEY"):
                raise RuntimeError(
                    "EVEROS_API_KEY is not set. Export it before using EverOS memory."
                )

            try:
                from everos import EverOS
            except ImportError as exc:
                raise RuntimeError(
                    "The everos package is not installed. Run `pip install -r requirements.txt`."
                ) from exc

            client = EverOS()

        self.client = client
        self.memories = client.v1.memories
        self.agent_memories = client.v1.memories.agent

    def retrieve_context(
        self,
        user_id: str,
        query: str,
        *,
        method: str = "hybrid",
        top_k: int = 5,
        memory_types: Sequence[str] = DEFAULT_MEMORY_TYPES,
    ) -> MemoryContext:
        """Search EverOS and return prompt-ready memory context."""

        response = self.memories.search(
            filters={"user_id": user_id},
            query=query,
            method=method,
            memory_types=list(memory_types),
            top_k=top_k,
        )
        return MemoryContext(text=format_memory_context(response), raw_response=response)

    def remember_exchange(
        self,
        user_id: str,
        user_message: str,
        assistant_message: str,
        *,
        session_id: str | None = None,
        async_mode: bool = True,
    ) -> Any:
        """Store one user/assistant exchange in EverOS."""

        timestamp = now_ms()
        return self.memories.add(
            user_id=user_id,
            session_id=session_id,
            async_mode=async_mode,
            messages=[
                {"role": "user", "timestamp": timestamp, "content": user_message},
                {
                    "role": "assistant",
                    "timestamp": timestamp + 1,
                    "content": assistant_message,
                },
            ],
        )

    def flush(self, user_id: str, *, session_id: str | None = None) -> Any:
        """Trigger EverOS boundary detection and extraction for accumulated turns."""

        return self.memories.flush(user_id=user_id, session_id=session_id)

    def remember_agent_messages(
        self,
        user_id: str,
        messages: Sequence[dict[str, Any]],
        *,
        session_id: str | None = None,
        async_mode: bool = True,
    ) -> Any:
        """Store an agent trajectory using EverOS agent memory.

        Messages may include EverOS agent roles: "user", "assistant", or "tool".
        Tool-call messages can include OpenAI-format tool_calls/tool_call_id fields.
        """

        return self.agent_memories.add(
            user_id=user_id,
            session_id=session_id,
            async_mode=async_mode,
            messages=list(messages),
        )

    def remember_agent_exchange(
        self,
        user_id: str,
        user_message: str,
        assistant_message: str,
        *,
        session_id: str | None = None,
        async_mode: bool = True,
    ) -> Any:
        """Store one user/assistant exchange as an EverOS agent trajectory."""

        timestamp = now_ms()
        return self.remember_agent_messages(
            user_id=user_id,
            session_id=session_id,
            async_mode=async_mode,
            messages=[
                {"role": "user", "timestamp": timestamp, "content": user_message},
                {
                    "role": "assistant",
                    "timestamp": timestamp + 1,
                    "content": assistant_message,
                },
            ],
        )

    def flush_agent_memory(
        self,
        user_id: str,
        *,
        session_id: str | None = None,
    ) -> Any:
        """Trigger extraction for accumulated EverOS agent memory."""

        return self.agent_memories.flush(user_id=user_id, session_id=session_id)


def format_memory_context(response: Any) -> str:
    """Format common EverOS search result fields as prompt context."""

    lines: list[str] = []
    for item in _iter_memory_items(response):
        text = _first_present(
            item,
            ("episode", "summary", "content", "text", "profile_data", "task_intent"),
        )
        if text is None:
            continue
        lines.append(_stringify_memory_item(text))

    return "\n".join(f"- {line}" for line in lines)


def _iter_memory_items(response: Any) -> Iterable[Any]:
    data = _get(response, "data", response)
    for field in (
        "episodes",
        "profiles",
        "results",
        "memories",
        "agent_cases",
        "agent_skills",
    ):
        value = _get(data, field)
        if value:
            yield from value


def _first_present(item: Any, names: Sequence[str]) -> Any | None:
    for name in names:
        value = _get(item, name)
        if value not in (None, "", []):
            return value
    return None


def _get(obj: Any, name: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _stringify_memory_item(value: Any) -> str:
    if isinstance(value, dict):
        parts: list[str] = []
        for key, item in value.items():
            if item not in (None, "", []):
                parts.append(f"{key}: {item}")
        return "; ".join(parts)
    return str(value)


def build_prompt(user_message: str, memory_context: str) -> str:
    """Compose a simple prompt with retrieved memory context."""

    if not memory_context:
        return user_message
    return f"Relevant user memories:\n{memory_context}\n\nUser: {user_message}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Search EverOS memory.")
    parser.add_argument("query", help="Memory search query")
    parser.add_argument("--user-id", required=True, help="EverOS user_id filter")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--method", default="hybrid")
    args = parser.parse_args()

    memory = EverOSMemory()
    context = memory.retrieve_context(
        args.user_id,
        args.query,
        top_k=args.top_k,
        method=args.method,
    )
    print(context.text)


if __name__ == "__main__":
    main()
