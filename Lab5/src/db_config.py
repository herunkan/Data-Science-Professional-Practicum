from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv


@dataclass
class MySQLConfig:
    host: str
    port: int
    user: str
    password: str
    unix_socket: str | None = None


def load_mysql_config() -> MySQLConfig:
    load_dotenv()
    host = os.getenv("MYSQL_HOST", "127.0.0.1").strip()
    port = int(os.getenv("MYSQL_PORT", "3306").strip())
    user = os.getenv("MYSQL_USER", "").strip()
    password = os.getenv("MYSQL_PASSWORD", "").strip()
    unix_socket = os.getenv("MYSQL_SOCKET", "").strip() or None

    if not user or not password:
        raise ValueError(
            "Missing MySQL credentials. Set MYSQL_USER and MYSQL_PASSWORD in .env."
        )

    return MySQLConfig(
        host=host,
        port=port,
        user=user,
        password=password,
        unix_socket=unix_socket,
    )
