import asyncio
import unittest
from unittest.mock import patch

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.memory import InMemorySaver

import agents.supervisor as supervisor
import agents.experts.complaint_graph as complaint_graph


class StubModel:
    def with_structured_output(self, schema, **kwargs):
        return self

    async def ainvoke(self, messages, **kwargs):
        return AIMessage(content="测试闲聊回复")


class StubRag:
    async def ainvoke(self, state):
        return {
            "agent_result": {
                "status": "completed",
                "response": "七天内可退货",
                "citations": [{"doc_id": "RETURNS"}],
            }
        }


class SupervisorRegression(unittest.IsolatedAsyncioTestCase):
    def make_graph(self):
        with patch.object(
            supervisor,
            "create_rag_subgraph",
            lambda **kwargs: StubRag(),
        ):
            return supervisor.create_supervisor_graph(
                InMemorySaver(),
                llm=StubModel(),
                retriever=object(),
            )

    async def send(self, graph, thread_id, message):
        return await graph.ainvoke(
            {
                "messages": [HumanMessage(content=message)],
                "user_id": "user_001",
                "session_id": thread_id,
            },
            {"configurable": {"thread_id": thread_id}},
        )

    async def test_confirm_complaint_after_rag_question(self):
        graph = self.make_graph()
        with patch.object(
            complaint_graph,
            "create_complaint",
            return_value={"complaint_id": "CMP-TEST", "status": "open"},
        ) as save:
            first = await self.send(
                graph, "confirm-test", "我要投诉，耳机包装破损"
            )
            middle = await self.send(
                graph, "confirm-test", "签收后几天内可以申请退货？"
            )
            final = await self.send(
                graph, "confirm-test", "确认提交"
            )

        self.assertEqual(first["task_status"], "awaiting_user")
        self.assertEqual(middle["intent"], "rag")
        self.assertEqual(final["intent"], "complaint")
        self.assertEqual(final["task_status"], "completed")
        self.assertIn("CMP-TEST", final["final_response"])
        save.assert_called_once()

    async def test_cancel_complaint_after_rag_question(self):
        graph = self.make_graph()
        with patch.object(complaint_graph, "create_complaint") as save:
            await self.send(graph, "cancel-test", "我要投诉，耳机包装破损")
            middle = await self.send(
                graph, "cancel-test", "签收后几天内可以申请退货？"
            )
            final = await self.send(graph, "cancel-test", "取消。")

        self.assertEqual(middle["intent"], "rag")
        self.assertEqual(final["intent"], "complaint")
        self.assertEqual(final["task_status"], "completed")
        self.assertIn("已取消投诉登记", final["final_response"])
        save.assert_not_called()

    async def test_expert_failure_fallback(self):
        class BrokenExpert:
            def __init__(self):
                self.calls = 0

            async def run(self, state):
                self.calls += 1
                raise RuntimeError("模拟专家异常")

        for name in ("rag", "order", "complaint", "chat"):
            with self.subTest(expert=name):
                handler = BrokenExpert()
                state = {
                    "complaint_draft": {"description": "包装破损"},
                    "complaint_idempotency_key": "test-key",
                }

                with self.assertLogs(supervisor.__name__, level="ERROR"):
                    result = await supervisor.guard_expert(
                        name, handler.run
                    )(state)

                self.assertEqual(result["task_status"], "failed")
                self.assertEqual(result["agent_result"]["citations"], [])
                self.assertEqual(handler.calls, 1)
                self.assertEqual(
                    state["complaint_idempotency_key"],
                    "test-key",
                )

                if name == "complaint":
                    self.assertIn(
                        "结果暂时无法确认",
                        result["final_response"],
                    )


if __name__ == "__main__":
    unittest.main(verbosity=2)