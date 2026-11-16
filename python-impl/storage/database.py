import sqlite3

from core.config import settings


DEMO_ORDERS = (
    (
        "ORD-DEMO-1001",
        "user_001",
        "无线蓝牙耳机",
        "shipped",
        199.00,
        "SF10000001",
        "运输中",
        "2026-11-14",
    ),
    (
        "ORD-DEMO-1002",
        "user_001",
        "机械键盘",
        "paid",
        399.00,
        "",
        "等待发货",
        "",
    ),
    (
        "ORD-DEMO-2001",
        "user_002",
        "显示器",
        "delivered",
        899.00,
        "YT20000001",
        "已签收",
        "2026-11-10",
    ),
)


def get_connection() -> sqlite3.Connection:
    settings.sqlite_db_path.parent.mkdir(parents=True, exist_ok=True)

    connection = sqlite3.connect(
        settings.sqlite_db_path,
        timeout=10,
    )
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    return connection


def initialize_database() -> None:
    with get_connection() as connection:
        connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS orders (
                order_id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                product_name TEXT NOT NULL,
                status TEXT NOT NULL,
                amount REAL NOT NULL,
                tracking_no TEXT NOT NULL DEFAULT '',
                logistics_status TEXT NOT NULL,
                estimated_delivery TEXT NOT NULL DEFAULT '',
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            );

            CREATE INDEX IF NOT EXISTS idx_orders_user_id
            ON orders(user_id);

            CREATE TABLE IF NOT EXISTS complaints (
                complaint_id TEXT PRIMARY KEY,
                idempotency_key TEXT NOT NULL UNIQUE,
                user_id TEXT NOT NULL,
                order_id TEXT,
                category TEXT NOT NULL,
                description TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'open',
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY(order_id) REFERENCES orders(order_id)
            );
            """
        )

        connection.executemany(
            """
            INSERT OR IGNORE INTO orders (
                order_id,
                user_id,
                product_name,
                status,
                amount,
                tracking_no,
                logistics_status,
                estimated_delivery
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            DEMO_ORDERS,
        )