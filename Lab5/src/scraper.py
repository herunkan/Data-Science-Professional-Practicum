from __future__ import annotations

import re
import time
from typing import Dict, List, Optional, Set
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup

USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/123.0.0.0 Safari/537.36"
)


def _parse_score(score_text: str) -> int:
    if not score_text:
        return 0
    t = score_text.lower().strip()
    if "point" in t:
        m = re.search(r"([\d\.]+)\s*k", t)
        if m:
            return int(float(m.group(1)) * 1000)
        m = re.search(r"(\d+)", t)
        if m:
            return int(m.group(1))
    return 0


def _parse_num_comments(text: str) -> int:
    if not text:
        return 0
    m = re.search(r"(\d+)", text.replace(",", ""))
    return int(m.group(1)) if m else 0


def _parse_post(thing) -> Optional[Dict]:
    classes = set(thing.get("class", []))
    if "promotedlink" in classes or "promoted" in classes:
        return None

    post_id = thing.get("data-fullname", "").replace("t3_", "")
    fullname = thing.get("data-fullname", "")
    if not post_id:
        return None

    title_tag = thing.select_one("a.title")
    title = title_tag.get_text(strip=True) if title_tag else ""
    url = title_tag.get("href", "") if title_tag else ""
    if url.startswith("/"):
        url = urljoin("https://old.reddit.com", url)

    comments_tag = thing.select_one("a.comments")
    permalink = comments_tag.get("href", "") if comments_tag else ""
    if permalink.startswith("/"):
        permalink = urljoin("https://old.reddit.com", permalink)
    if "/comments/" not in permalink:
        return None

    selftext_tag = thing.select_one("div.expando .usertext-body")
    selftext = selftext_tag.get_text(" ", strip=True) if selftext_tag else ""

    author_tag = thing.select_one("a.author")
    author = author_tag.get_text(strip=True) if author_tag else "[deleted]"

    score_tag = thing.select_one("div.score.unvoted")
    score = _parse_score(score_tag.get_text(" ", strip=True) if score_tag else "")

    num_comments = _parse_num_comments(comments_tag.get_text(" ", strip=True) if comments_tag else "")
    created_utc = int(thing.get("data-timestamp", "0")) // 1000

    return {
        "post_id": post_id,
        "fullname": fullname,
        "created_utc": created_utc,
        "title": title,
        "selftext": selftext,
        "author": author,
        "score": score,
        "num_comments": num_comments,
        "url": url,
        "permalink": permalink,
        "is_self": 1 if "/comments/" in permalink else 0,
        "over_18": 0,
        "stickied": 0,
    }


def fetch_posts(
    subreddit_name: str,
    target_count: int,
    request_sleep_sec: float = 0.75,
    timeout_sec: int = 20,
) -> List[Dict]:
    """
    Fetch up to target_count posts from old.reddit.com using HTML pagination.
    This avoids API credentials while supporting large requests.
    """
    if target_count <= 0:
        return []

    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT})

    results: List[Dict] = []
    seen_ids: Set[str] = set()
    listing_urls = [
        f"https://old.reddit.com/r/{subreddit_name}/hot/",
        f"https://old.reddit.com/r/{subreddit_name}/new/",
        f"https://old.reddit.com/r/{subreddit_name}/top/?sort=top&t=month",
    ]

    for start_url in listing_urls:
        next_url = start_url
        while next_url and len(results) < target_count:
            resp = session.get(next_url, timeout=timeout_sec)
            if resp.status_code != 200:
                raise RuntimeError(f"Failed to fetch {next_url}: HTTP {resp.status_code}")

            soup = BeautifulSoup(resp.text, "html.parser")
            things = soup.select("div.thing")
            if not things:
                break

            for thing in things:
                post = _parse_post(thing)
                if not post:
                    continue
                if post["post_id"] in seen_ids:
                    continue
                seen_ids.add(post["post_id"])
                results.append(post)
                if len(results) >= target_count:
                    break

            next_btn = soup.select_one("span.next-button a")
            next_url = next_btn.get("href") if next_btn else None
            time.sleep(request_sleep_sec)
        if len(results) >= target_count:
            break

    return results[:target_count]
