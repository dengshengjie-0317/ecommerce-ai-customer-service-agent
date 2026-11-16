from __future__ import annotations

from typing import Any

from langchain_core.messages import HumanMessage
from langgraph.graph import END, StateGraph

from agents.experts.chat_graph import create_chat_subgraph
from agents.experts.complaint_graph import (
    CANCEL_WORDS,
    CONFIRM_WORDS,
    create_complaint_subgraph,
    normalize_confirmation,
)
from agents.experts.order_graph import create_order_subgraph
from agents.experts.rag_graph import create_rag_subgraph
from agents.intent_router import IntentDecision, IntentRouterAgent
from core.llm import create_chat_model
from core.state import AgentState
from retrieval.bm25 import BM25Retriever
from retrieval.documents import load_knowledge_documents, split_knowledge_documents


def latest_user_message(state: AgentState) -> str:
    for message in reversed(state.get("messages", [])):
        if isinstance(message, HumanMessage):
            return str(message.content).strip()
    return ""


def create_supervisor_graph(
    checkpointer: Any,
    llm: Any | None = None,
    retriever: Any | None = None,
):
    model = llm or create_chat_model()
    intent_router = IntentRouterAgent(model)

    if retriever is None:
        documents = load_knowledge_documents()
        chunks = split_knowledge_documents(documents)
        retriever = BM25Retriever(chunks)

    rag_graph = create_rag_subgraph(retriever=retriever, llm=model)
    order_graph = create_order_subgraph()
    complaint_graph = create_complaint_subgraph()
    chat_graph = create_chat_subgraph(llm=model)

    async def route(state: AgentState) -> dict[str, Any]:
        message = latest_user_message(state)

        if not message:
            decision = IntentDecision(
                intent="chat",
                route_source="fallback",
                confidence=0.0,
                reason="没有收到用户消息",
                needs_clarification=True,
            )
        else:
            pending_confirmation = (
                    bool(state.get("complaint_draft"))
                    and bool(state.get("complaint_idempotency_key"))
                    and state.get("complaint_action") == "awaiting_confirmation"
            )
            normalized = normalize_confirmation(message)

            if pending_confirmation and normalized in CONFIRM_WORDS:
                decision = IntentDecision(
                    intent="complaint",
                    route_source="continuation",
                    confidence=1.0,
                    reason="用户确认提交投诉",
                )
            elif pending_confirmation and normalized in CANCEL_WORDS:
                decision = IntentDecision(
                    intent="complaint",
                    route_source="continuation",
                    confidence=1.0,
                    reason="用户取消投诉登记",
                )
            else:
                decision = await intent_router.classify(message)

        return {
            "intent": decision.intent,
            "route_source": decision.route_source,
            "route_confidence": decision.confidence,
            "route_reason": decision.reason,
            "route_needs_clarification": decision.needs_clarification,
            "active_task": decision.intent,
            "current_agent": "supervisor",
            "iteration_count": state.get("iteration_count", 0) + 1,
        }

    async def run_rag(state: AgentState) -> dict[str, Any]:
        result = await rag_graph.ainvoke({"query": latest_user_message(state)})
        agent_result = result.get("agent_result", {})
        return {
            "current_agent": "rag",
            "task_status": agent_result.get("status", "failed"),
            "agent_result": agent_result,
            "final_response": agent_result.get(
                "response", "暂时无法回答这个问题，请稍后再试。"
            ),
        }

    async def clarify(_: AgentState) -> dict[str, Any]:
        response = "我还不确定你需要哪类帮助。请说明是查询订单、咨询商品或配送规则，还是登记投诉。"
        return {
            "task_status": "awaiting_user",
            "agent_result": {
                "response": response,
                "status": "awaiting_user",
            },
            "final_response": response,
        }

    def next_node(state: AgentState) -> str:
        if state.get("route_needs_clarification"):
            return "clarify"
        return state.get("intent", "chat")

    graph = StateGraph(AgentState)
    graph.add_node("route", route)
    graph.add_node("clarify", clarify)
    graph.add_node("rag", run_rag)
    graph.add_node("order", order_graph)
    graph.add_node("complaint", complaint_graph)
    graph.add_node("chat", chat_graph)

    graph.set_entry_point("route")
    graph.add_conditional_edges(
        "route",
        next_node,
        {
            "clarify": "clarify",
            "rag": "rag",
            "order": "order",
            "complaint": "complaint",
            "chat": "chat",
        },
    )

    for node in ("clarify", "rag", "order", "complaint", "chat"):
        graph.add_edge(node, END)

    return graph.compile(checkpointer=checkpointer)