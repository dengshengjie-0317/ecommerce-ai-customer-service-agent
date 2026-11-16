import re
from typing import Any

from langgraph.graph import END, StateGraph

from core.state import AgentState
from storage.orders import find_order_for_user, list_orders_for_user


ORDER_ID_PATTERN = re.compile(r"\bORD-[A-Z0-9-]+\b", re.IGNORECASE)

ORDER_STATUS_LABELS = {
    "paid": "已付款，等待发货",
    "shipped": "已发货",
    "delivered": "已签收",
    "cancelled": "已取消",
}


async def lookup_order(state: AgentState) -> dict[str, Any]:
    """查询当前用户自己的演示订单，不让模型编造订单数据。"""

    user_id = state.get("user_id", "")
    messages = state.get("messages", [])

    if not user_id:
        return {
            "current_agent": "order",
            "task_status": "failed",
            "agent_result": {
                "response": "无法确认当前用户身份，暂时不能查询订单。",
                "status": "failed",
            },
            "final_response": "无法确认当前用户身份，暂时不能查询订单。",
        }

    if not messages:
        response = "没有收到订单查询内容，请告诉我你想查询哪个订单。"
        return {
            "current_agent": "order",
            "task_status": "awaiting_user",
            "pending_fields": ["order_id"],
            "agent_result": {
                "response": response,
                "status": "awaiting_user",
                "pending_fields": ["order_id"],
            },
            "final_response": response,
        }

    message = str(messages[-1].content)
    match = ORDER_ID_PATTERN.search(message)

    if match:
        order_id = match.group(0).upper()
        order = find_order_for_user(order_id, user_id)

        if order is None:
            response = "没有查到属于你账号的这个订单，请核对订单号后再试。"
            return {
                "current_agent": "order",
                "task_status": "failed",
                "agent_result": {
                    "response": response,
                    "status": "failed",
                },
                "final_response": response,
            }

        status = ORDER_STATUS_LABELS.get(order["status"], order["status"])
        response = (
            f"订单号：{order['order_id']}\n"
            f"商品：{order['product_name']}\n"
            f"订单状态：{status}\n"
            f"物流状态：{order['logistics_status'] or '暂无物流信息'}"
        )

        if order["estimated_delivery"]:
            response += f"\n预计送达：{order['estimated_delivery']}"

        result = {
            "response": response,
            "status": "completed",
            "data": {"order_id": order["order_id"]},
        }
        return {
            "current_agent": "order",
            "task_status": "completed",
            "pending_fields": [],
            "agent_result": result,
            "final_response": response,
        }

    orders = list_orders_for_user(user_id)

    if not orders:
        response = "当前账号下没有可查询的订单。"
        return {
            "current_agent": "order",
            "task_status": "completed",
            "pending_fields": [],
            "agent_result": {
                "response": response,
                "status": "completed",
            },
            "final_response": response,
        }

    if len(orders) == 1:
        order = orders[0]
        status = ORDER_STATUS_LABELS.get(order["status"], order["status"])
        response = (
            f"订单号：{order['order_id']}\n"
            f"商品：{order['product_name']}\n"
            f"订单状态：{status}\n"
            f"物流状态：{order['logistics_status'] or '暂无物流信息'}"
        )

        result = {
            "response": response,
            "status": "completed",
            "data": {"order_id": order["order_id"]},
        }
        return {
            "current_agent": "order",
            "task_status": "completed",
            "pending_fields": [],
            "agent_result": result,
            "final_response": response,
        }

    order_options = "\n".join(
        f"- {order['order_id']}：{order['product_name']}"
        for order in orders
    )
    response = f"你有多个订单，请提供要查询的订单号：\n{order_options}"

    return {
        "current_agent": "order",
        "task_status": "awaiting_user",
        "pending_fields": ["order_id"],
        "agent_result": {
            "response": response,
            "status": "awaiting_user",
            "pending_fields": ["order_id"],
        },
        "final_response": response,
    }


def create_order_subgraph():
    """构建只负责订单查询的 LangGraph 子图。"""

    graph = StateGraph(AgentState)
    graph.add_node("lookup_order", lookup_order)
    graph.set_entry_point("lookup_order")
    graph.add_edge("lookup_order", END)

    return graph.compile()