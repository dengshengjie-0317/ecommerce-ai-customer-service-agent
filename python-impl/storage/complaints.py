from typing import Any
from uuid import uuid4

from storage.database import get_connection


def create_complaint(
    user_id: str,
    category: str,
    description: str,
    idempotency_key: str,
    order_id: str | None = None,
) -> dict[str, Any]:
    """创建投诉；重复提交同一请求时返回同一条记录。"""

    user_id = user_id.strip()
    category = category.strip()
    description = description.strip()
    idempotency_key = idempotency_key.strip()
    order_id = order_id.strip().upper() if order_id else None

    if not user_id or not category or not description or not idempotency_key:
        raise ValueError("user_id、category、description 和 idempotency_key 不能为空")

    connection = get_connection()

    try:
        if order_id:
            owned_order = connection.execute(
                """
                SELECT 1 FROM orders
                WHERE order_id = ? AND user_id = ?
                """,
                (order_id, user_id),
            ).fetchone()

            if owned_order is None:
                raise ValueError("订单不存在或不属于当前用户")

        complaint_id = f"CMP-{uuid4().hex[:10].upper()}"

        connection.execute(
            """
            INSERT OR IGNORE INTO complaints (
                complaint_id,
                idempotency_key,
                user_id,
                order_id,
                category,
                description
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                complaint_id,
                idempotency_key,
                user_id,
                order_id,
                category,
                description,
            ),
        )

        row = connection.execute(
            """
            SELECT
                complaint_id,
                idempotency_key,
                user_id,
                order_id,
                category,
                description,
                status,
                created_at
            FROM complaints
            WHERE idempotency_key = ?
            """,
            (idempotency_key,),
        ).fetchone()

        if row is None:
            raise RuntimeError("投诉记录写入失败")

        complaint = dict(row)

        if complaint["user_id"] != user_id:
            raise PermissionError("该请求标识已被其他用户使用")

        same_request = (
            complaint["order_id"] == order_id
            and complaint["category"] == category
            and complaint["description"] == description
        )
        if not same_request:
            raise ValueError("同一个幂等键不能用于不同的投诉内容")

        connection.commit()
        return complaint

    except Exception:
        connection.rollback()
        raise

    finally:
        connection.close()


def list_complaints_for_user(user_id: str) -> list[dict[str, Any]]:
    """只列出属于指定用户的投诉记录。"""

    with get_connection() as connection:
        rows = connection.execute(
            """
            SELECT
                complaint_id,
                order_id,
                category,
                description,
                status,
                created_at
            FROM complaints
            WHERE user_id = ?
            ORDER BY created_at DESC, complaint_id
            """,
            (user_id.strip(),),
        ).fetchall()

    return [dict(row) for row in rows]


def find_complaint_for_user(
    complaint_id: str,
    user_id: str,
) -> dict[str, Any] | None:
    """查询投诉时同时检查用户归属。"""

    with get_connection() as connection:
        row = connection.execute(
            """
            SELECT
                complaint_id,
                order_id,
                category,
                description,
                status,
                created_at
            FROM complaints
            WHERE complaint_id = ? AND user_id = ?
            """,
            (complaint_id.strip().upper(), user_id.strip()),
        ).fetchone()

    return dict(row) if row else None