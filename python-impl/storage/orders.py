from contextlib import closing
from typing import Any

from storage.database import get_connection


def list_orders_for_user(user_id: str) -> list[dict[str, Any]]:
    """只列出属于指定用户的订单。"""

    with closing(get_connection()) as connection:
        rows = connection.execute(
            """
            SELECT
                order_id,
                product_name,
                status,
                amount,
                tracking_no,
                logistics_status,
                estimated_delivery,
                created_at
            FROM orders
            WHERE user_id = ?
            ORDER BY created_at DESC, order_id
            """,
            (user_id,),
        ).fetchall()

    return [dict(row) for row in rows]


def find_order_for_user(
    order_id: str,
    user_id: str,
) -> dict[str, Any] | None:
    """查找订单，同时验证订单属于该用户。"""

    with closing(get_connection()) as connection:
        row = connection.execute(
            """
            SELECT
                order_id,
                product_name,
                status,
                amount,
                tracking_no,
                logistics_status,
                estimated_delivery,
                created_at
            FROM orders
            WHERE order_id = ? AND user_id = ?
            """,
            (order_id, user_id),
        ).fetchone()

    return dict(row) if row else None