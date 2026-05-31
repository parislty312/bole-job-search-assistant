from types import SimpleNamespace
import unittest

from integrations.everos_memory import (
    EverOSMemory,
    build_prompt,
    format_memory_context,
)


class FakeMemories:
    def __init__(self):
        self.search_call = None
        self.add_call = None
        self.flush_call = None

    def search(self, **kwargs):
        self.search_call = kwargs
        return SimpleNamespace(
            data=SimpleNamespace(
                episodes=[SimpleNamespace(episode="User prefers morning meetings.")],
                profiles=[{"profile_data": {"timezone": "America/Los_Angeles"}}],
            )
        )

    def add(self, **kwargs):
        self.add_call = kwargs
        return SimpleNamespace(data=SimpleNamespace(status="queued"))

    def flush(self, **kwargs):
        self.flush_call = kwargs
        return SimpleNamespace(data=SimpleNamespace(status="extracted"))


class FakeClient:
    def __init__(self):
        self.memories = FakeMemories()
        self.memories.agent = FakeMemories()
        self.v1 = SimpleNamespace(memories=self.memories)


class EverOSMemoryTests(unittest.TestCase):
    def test_retrieve_context_uses_hybrid_defaults(self):
        client = FakeClient()
        memory = EverOSMemory(client=client)

        context = memory.retrieve_context("user_123", "meeting preferences")

        self.assertEqual(
            client.memories.search_call,
            {
                "filters": {"user_id": "user_123"},
                "query": "meeting preferences",
                "method": "hybrid",
                "memory_types": ["episodic_memory", "profile"],
                "top_k": 5,
            },
        )
        self.assertIn("User prefers morning meetings.", context.text)
        self.assertIn("timezone", context.text)

    def test_remember_exchange_stores_user_and_assistant_turns(self):
        client = FakeClient()
        memory = EverOSMemory(client=client)

        memory.remember_exchange(
            "user_123",
            "Hello",
            "Hi there",
            session_id="session_abc",
            async_mode=False,
        )

        call = client.memories.add_call
        self.assertEqual(call["user_id"], "user_123")
        self.assertEqual(call["session_id"], "session_abc")
        self.assertFalse(call["async_mode"])
        self.assertEqual(call["messages"][0]["role"], "user")
        self.assertEqual(call["messages"][1]["role"], "assistant")

    def test_remember_agent_messages_uses_agent_endpoint(self):
        client = FakeClient()
        memory = EverOSMemory(client=client)
        messages = [
            {
                "role": "user",
                "timestamp": 1711900000000,
                "content": "Find all Python files with TODO comments",
            }
        ]

        memory.remember_agent_messages(
            "agent_123",
            messages,
            session_id="task_session_001",
        )

        self.assertIsNone(client.memories.add_call)
        self.assertEqual(
            client.memories.agent.add_call,
            {
                "user_id": "agent_123",
                "session_id": "task_session_001",
                "async_mode": True,
                "messages": messages,
            },
        )

    def test_flush_agent_memory_uses_agent_endpoint(self):
        client = FakeClient()
        memory = EverOSMemory(client=client)

        memory.flush_agent_memory("agent_123", session_id="task_session_001")

        self.assertEqual(
            client.memories.agent.flush_call,
            {"user_id": "agent_123", "session_id": "task_session_001"},
        )

    def test_format_memory_context_accepts_dict_responses(self):
        response = {
            "data": {
                "episodes": [{"summary": "Chose PostgreSQL for the service."}],
                "agent_cases": [{"task_intent": "Deploy staging"}],
            }
        }

        context = format_memory_context(response)

        self.assertIn("Chose PostgreSQL", context)
        self.assertIn("Deploy staging", context)

    def test_build_prompt_omits_empty_memory_section(self):
        self.assertEqual(build_prompt("Hello", ""), "Hello")
        self.assertIn("Relevant user memories", build_prompt("Hello", "- Likes tea"))


if __name__ == "__main__":
    unittest.main()
