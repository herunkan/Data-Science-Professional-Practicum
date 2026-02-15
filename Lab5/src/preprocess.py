from __future__ import annotations

import html
import re
from collections import Counter
from typing import Dict, List

from bs4 import BeautifulSoup

STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "been", "but", "by", "for", "from",
    "has", "have", "he", "her", "his", "i", "if", "in", "is", "it", "its", "just",
    "me", "my", "not", "of", "on", "or", "our", "so", "that", "the", "their", "them",
    "they", "this", "to", "too", "was", "we", "were", "what", "when", "which", "who",
    "will", "with", "you", "your",
}


def clean_text(text: str) -> str:
    text = html.unescape(text or "")
    text = BeautifulSoup(text, "html.parser").get_text(" ")
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = re.sub(r"/u/\w+", " [USER] ", text)
    text = re.sub(r"[^a-zA-Z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def mask_username(name: str) -> str:
    if not name or name in ("[deleted]", "None"):
        return "[deleted]"
    return f"user_{abs(hash(name)) % 1000000:06d}"


def extract_keywords(text: str, top_k: int = 8) -> List[str]:
    tokens = [t.lower() for t in text.split() if len(t) > 2]
    tokens = [t for t in tokens if t not in STOPWORDS and not t.isdigit()]
    counts = Counter(tokens)
    return [w for w, _ in counts.most_common(top_k)]


def preprocess_posts(posts: List[Dict]) -> List[Dict]:
    processed = []
    for p in posts:
        merged = f"{p.get('title', '')} {p.get('selftext', '')}"
        cleaned = clean_text(merged)
        keywords = extract_keywords(cleaned)
        topic_hint = keywords[0] if keywords else "general"
        processed.append(
            {
                **p,
                "author_masked": mask_username(p.get("author", "")),
                "clean_text": cleaned,
                "keywords": ",".join(keywords),
                "topic_hint": topic_hint,
            }
        )
    return processed
