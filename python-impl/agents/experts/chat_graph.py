from typing import Any

from langchain_core.messages import SystemMessage
from langgraph.graph import END, StateGraph

from core.llm import create_chat_model
from core.state import AgentState


CHAT_SYSTEM_PROMPT = """你是电商客服系统的闲聊助手。

你负责问候、介绍客服能做什么，以及自然的日常交流。
回答简洁、友好，使用中文。

不要编造或声称已经执行订单查询、退款、投诉登记等业务操作。
如果用户提出订单查询、商品政策咨询或投诉办理，请说明可以帮助处理，
但不要在闲聊子图中执行这些业务。
"""


def create_chat_subgraph(llm: Any | None = None):
    """构建只负责闲聊的专家子图。"""

    model = llm or create_chat_model()

    async def reply(state: AgentState) -> dict[str, Any]:
        messages = state.get("messages", [])

        if not messages:
            response_text = "你好，请问有什么可以帮你？"
            return {
                "current_agent": "chat",
                "task_status": "completed",
                "agent_result": {
                    "response": response_text,
                    "status": "completed",
                },
                "final_response": response_text,
            }

        prompt = [
            SystemMessage(content=CHAT_SYSTEM_PROMPT),
            *messages[-8:],
        ]

        try:
            result = await model.ainvoke(prompt)
            response_text = str(result.content).strip()

            if not response_text:
                raise ValueError("模型返回了空回复")

            status = "completed"
        except Exception:
            response_text = (
                "抱歉，我现在暂时无法回复。"
                "你可以告诉我需要查询订单、了解商品政策或登记投诉。"
            )
            status = "failed"

        return {
            "current_agent": "chat",
            "task_status": status,
            "agent_result": {
                "response": response_text,
                "status": status,
            },
            "final_response": response_text,
        }

    graph = StateGraph(AgentState)
    graph.add_node("chat_reply", reply)
    graph.set_entry_point("chat_reply")
    graph.add_edge("chat_reply", END)

    return graph.compile()