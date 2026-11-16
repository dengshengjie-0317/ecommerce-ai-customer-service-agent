from typing import Annotated, Any, Literal, TypedDict
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


Intent = Literal["rag", "order", "complaint", "chat"]
RouteSource = Literal["rule", "llm", "fallback", "continuation"]
TaskStatus = Literal["completed", "awaiting_user", "handoff", "failed"]

class ComplaintDraft(TypedDict):
    order_id: str | None
    category: str
    description: str

class AgentResult(TypedDict, total=False):
    response: str
    status: TaskStatus
    data: dict[str, Any]
    citations: list[dict[str, Any]]
    pending_fields: list[str]


class AgentState(TypedDict, total=False):
    messages: Annotated[list[BaseMessage], add_messages]
    user_id: str
    session_id: str

    intent: Intent
    route_source: RouteSource
    current_agent: str

    task_status: TaskStatus
    pending_fields: list[str]
    iteration_count: int
    tool_call_count: int

    agent_result: AgentResult
    final_response: str

    complaint_draft: ComplaintDraft | None
    complaint_idempotency_key: str
    complaint_action: Literal[
        "create",
        "cancel",
        "awaiting_confirmation",
        "idle",
    ]
    route_confidence: float
    route_reason: str
    route_needs_clarification: bool
    active_task: Intent | None