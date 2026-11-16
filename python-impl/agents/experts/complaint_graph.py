import re
from typing import Any
from uuid import uuid4

from langgraph.graph import END, StateGraph

from core.state import AgentState, ComplaintDraft
from storage.complaints import create_complaint


ORDER_ID_PATTERN = re.compile(r"\bORD-[A-Z0-9-]+\b", re.IGNORECASE)

CONFIRM_WORDS = {
    "确认",
    "确认提交",
    "确认创建",
    "同意",
    "提交",
    "yes",
    "y",
}

CANCEL_WORDS = {
    "取消",
    "取消投诉",
    "不提交",
    "不同意",
    "不用了",
    "no",
    "n",
}


def normalize_confirmation(text: str) -> str:
    text = text.strip().lower()
    return re.sub(r"[\s，。！？,.!?；;：:]", "", text)


def detect_category(text: str) -> str:
    if any(word in text for word in ("物流", "配送", "快递", "发货")):
        return "delivery"
    if any(word in text for word in ("退款", "退钱", "退款不到账")):
        return "refund"
    if any(word in text for word in ("破损", "损坏", "质量", "坏了")):
        return "product_quality"
    return "general"


async def collect_or_confirm(state: AgentState) -> dict[str, Any]:
    user_id = state.get("user_id", "")
    messages = state.get("messages", [])

    if not user_id:
        response = "无法确认当前用户身份，暂时不能登记投诉。"
        return {
            "current_agent": "complaint",
            "task_status": "failed",
            "agent_result": {
                "response": response,
                "status": "failed",
            },
            "final_response": response,
            "complaint_action": "idle",
        }

    if not messages:
        response = "请简单描述你遇到的问题。"
        return {
            "current_agent": "complaint",
            "task_status": "awaiting_user",
            "pending_fields": ["description"],
            "agent_result": {
                "response": response,
                "status": "awaiting_user",
                "pending_fields": ["description"],
            },
            "final_response": response,
            "complaint_action": "idle",
        }

    message = str(messages[-1].content).strip()
    draft = state.get("complaint_draft")

    if draft:
        confirmation = normalize_confirmation(message)

        if confirmation in CONFIRM_WORDS:
            return {
                "current_agent": "complaint",
                "complaint_action": "create",
            }

        if confirmation in CANCEL_WORDS:
            response = "已取消投诉登记，没有创建投诉记录。"
            return {
                "current_agent": "complaint",
                "task_status": "completed",
                "pending_fields": [],
                "complaint_draft": None,
                "complaint_idempotency_key": "",
                "complaint_action": "cancel",
                "agent_result": {
                    "response": response,
                    "status": "completed",
                },
                "final_response": response,
            }

        response = "请回复“确认提交”或“取消”，我再继续处理。"
        return {
            "current_agent": "complaint",
            "task_status": "awaiting_user",
            "pending_fields": ["confirmation"],
            "complaint_action": "awaiting_confirmation",
            "agent_result": {
                "response": response,
                "status": "awaiting_user",
                "pending_fields": ["confirmation"],
            },
            "final_response": response,
        }

    if len(message) < 4:
        response = "请再描述一下问题经过，方便我准确登记。"
        return {
            "current_agent": "complaint",
            "task_status": "awaiting_user",
            "pending_fields": ["description"],
            "complaint_action": "idle",
            "agent_result": {
                "response": response,
                "status": "awaiting_user",
                "pending_fields": ["description"],
            },
            "final_response": response,
        }

    order_match = ORDER_ID_PATTERN.search(message)
    order_id = order_match.group(0).upper() if order_match else None

    complaint_draft: ComplaintDraft = {
        "order_id": order_id,
        "category": detect_category(message),
        "description": message,
    }

    idempotency_key = f"{state.get('session_id', 'session')}:{uuid4().hex}"

    response = (
        "我准备登记以下投诉：\n"
        f"问题描述：{message}\n"
        f"关联订单：{order_id or '未提供'}\n"
        "请回复“确认提交”或“取消”。"
    )

    return {
        "current_agent": "complaint",
        "task_status": "awaiting_user",
        "pending_fields": ["confirmation"],
        "complaint_draft": complaint_draft,
        "complaint_idempotency_key": idempotency_key,
        "complaint_action": "awaiting_confirmation",
        "agent_result": {
            "response": response,
            "status": "awaiting_user",
            "pending_fields": ["confirmation"],
        },
        "final_response": response,
    }


async def save_confirmed_complaint(state: AgentState) -> dict[str, Any]:
    draft = state.get("complaint_draft")
    user_id = state.get("user_id", "")
    idempotency_key = state.get("complaint_idempotency_key", "")

    if not draft or not user_id or not idempotency_key:
        response = "投诉信息已失效，请重新描述问题。"
        return {
            "current_agent": "complaint",
            "task_status": "failed",
            "complaint_draft": None,
            "complaint_idempotency_key": "",
            "complaint_action": "idle",
            "agent_result": {
                "response": response,
                "status": "failed",
            },
            "final_response": response,
        }

    try:
        complaint = create_complaint(
            user_id=user_id,
            category=draft["category"],
            description=draft["description"],
            idempotency_key=idempotency_key,
            order_id=draft["order_id"],
        )
    except (PermissionError, ValueError) as error:
        response = f"无法登记这条投诉：{error}"
        return {
            "current_agent": "complaint",
            "task_status": "failed",
            "complaint_draft": None,
            "complaint_idempotency_key": "",
            "complaint_action": "idle",
            "agent_result": {
                "response": response,
                "status": "failed",
            },
            "final_response": response,
        }

    response = f"投诉已登记，编号：{complaint['complaint_id']}。"

    return {
        "current_agent": "complaint",
        "task_status": "completed",
        "pending_fields": [],
        "complaint_draft": None,
        "complaint_idempotency_key": "",
        "complaint_action": "idle",
        "agent_result": {
            "response": response,
            "status": "completed",
            "data": {
                "complaint_id": complaint["complaint_id"],
                "status": complaint["status"],
            },
        },
        "final_response": response,
    }


def route_after_collection(state: AgentState) -> str:
    if state.get("complaint_action") == "create":
        return "save"
    return "end"


def create_complaint_subgraph():
    graph = StateGraph(AgentState)

    graph.add_node("collect_or_confirm", collect_or_confirm)
    graph.add_node("save_complaint", save_confirmed_complaint)

    graph.set_entry_point("collect_or_confirm")
    graph.add_conditional_edges(
        "collect_or_confirm",
        route_after_collection,
        {
            "save": "save_complaint",
            "end": END,
        },
    )
    graph.add_edge("save_complaint", END)

    return graph.compile()