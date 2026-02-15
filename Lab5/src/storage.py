from __future__ import annotations

from typing import Dict, Iterable, List

import mysql.connector
import pandas as pd

from .db_config import load_mysql_config

def _connect(database: str | None = None):
    cfg = load_mysql_config()
    kwargs = {
        "user": cfg.user,
        "password": cfg.password,
        "autocommit": False,
    }
    if cfg.unix_socket:
        kwargs["unix_socket"] = cfg.unix_socket
    else:
        kwargs["host"] = cfg.host
        kwargs["port"] = cfg.port
    if database:
        kwargs["database"] = database
    return mysql.connector.connect(**kwargs)


def init_db(db_name: str) -> None:
    with _connect() as conn:
        cur = conn.cursor()
        cur.execute(f"CREATE DATABASE IF NOT EXISTS `{db_name}`")
        conn.commit()

    with _connect(db_name) as conn:
        cur = conn.cursor()
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS reddit_posts (
                post_id VARCHAR(32) PRIMARY KEY,
                fullname VARCHAR(64),
                created_utc BIGINT,
                title TEXT,
                selftext LONGTEXT,
                clean_text LONGTEXT,
                keywords TEXT,
                topic_hint VARCHAR(128),
                author_masked VARCHAR(128),
                score INT,
                num_comments INT,
                url TEXT,
                permalink TEXT,
                is_self TINYINT(1),
                over_18 TINYINT(1),
                stickied TINYINT(1),
                cluster_id INT NULL,
                centroid_distance DOUBLE NULL,
                fetched_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS cluster_keywords (
                cluster_id INT PRIMARY KEY,
                top_keywords TEXT,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
            )
            """
        )
        conn.commit()


def upsert_posts(db_name: str, rows: Iterable[Dict]) -> int:
    rows = list(rows)
    if not rows:
        return 0

    payload = []
    for r in rows:
        payload.append(
            (
                r.get("post_id"),
                r.get("fullname"),
                int(r.get("created_utc") or 0),
                r.get("title", ""),
                r.get("selftext", ""),
                r.get("clean_text", ""),
                r.get("keywords", ""),
                r.get("topic_hint", ""),
                r.get("author_masked", ""),
                int(r.get("score") or 0),
                int(r.get("num_comments") or 0),
                r.get("url", ""),
                r.get("permalink", ""),
                int(r.get("is_self") or 0),
                int(r.get("over_18") or 0),
                int(r.get("stickied") or 0),
                r.get("cluster_id"),
                r.get("centroid_distance"),
            )
        )

    with _connect(db_name) as conn:
        cur = conn.cursor()
        cur.executemany(
            """
            INSERT INTO reddit_posts (
                post_id, fullname, created_utc, title, selftext, clean_text,
                keywords, topic_hint, author_masked, score, num_comments,
                url, permalink, is_self, over_18, stickied, cluster_id, centroid_distance
            )
            VALUES (
                %s, %s, %s, %s, %s, %s,
                %s, %s, %s, %s, %s,
                %s, %s, %s, %s, %s, %s, %s
            )
            ON DUPLICATE KEY UPDATE
                fullname=VALUES(fullname),
                created_utc=VALUES(created_utc),
                title=VALUES(title),
                selftext=VALUES(selftext),
                clean_text=VALUES(clean_text),
                keywords=VALUES(keywords),
                topic_hint=VALUES(topic_hint),
                author_masked=VALUES(author_masked),
                score=VALUES(score),
                num_comments=VALUES(num_comments),
                url=VALUES(url),
                permalink=VALUES(permalink),
                is_self=VALUES(is_self),
                over_18=VALUES(over_18),
                stickied=VALUES(stickied),
                cluster_id=VALUES(cluster_id),
                centroid_distance=VALUES(centroid_distance)
            """,
            payload,
        )
        conn.commit()
    return len(rows)


def update_cluster_keywords(db_name: str, cluster_keywords: Dict[int, List[str]]) -> None:
    payload = [(int(cid), ",".join(words)) for cid, words in cluster_keywords.items()]
    with _connect(db_name) as conn:
        cur = conn.cursor()
        cur.executemany(
            """
            INSERT INTO cluster_keywords(cluster_id, top_keywords, updated_at)
            VALUES (%s, %s, CURRENT_TIMESTAMP)
            ON DUPLICATE KEY UPDATE
                top_keywords=VALUES(top_keywords),
                updated_at=CURRENT_TIMESTAMP
            """,
            payload,
        )
        conn.commit()


def load_posts_frame(db_name: str) -> pd.DataFrame:
    with _connect(db_name) as conn:
        cur = conn.cursor(dictionary=True)
        cur.execute(
            """
            SELECT post_id, created_utc, title, clean_text, keywords, topic_hint, author_masked,
                   score, num_comments, permalink, cluster_id, centroid_distance
            FROM reddit_posts
            WHERE permalink IS NOT NULL
              AND permalink LIKE '%%/comments/%%'
              AND (score > 0 OR num_comments > 0)
            ORDER BY created_utc DESC
            """
        )
        rows = cur.fetchall()
    return pd.DataFrame(rows)
