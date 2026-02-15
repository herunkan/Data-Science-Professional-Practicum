from __future__ import annotations

import argparse
from pathlib import Path

from src.cluster import cluster_messages, posts_near_centroid, save_cluster_scatter
from src.preprocess import preprocess_posts
from src.scraper import fetch_posts
from src.storage import init_db, load_posts_frame, update_cluster_keywords, upsert_posts


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Lab5 Reddit scraping + preprocessing + clustering")
    p.add_argument("--subreddit", type=str, default="Fitness")
    p.add_argument("--num-posts", type=int, default=500)
    p.add_argument("--k-clusters", type=int, default=6)
    p.add_argument("--db-name", type=str, default="lab5_reddit")
    p.add_argument("--output-dir", type=str, default="outputs")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("[1/5] Initializing database...")
    init_db(args.db_name)

    print("[2/5] Fetching posts with BeautifulSoup scraper...")
    raw_posts = fetch_posts(
        subreddit_name=args.subreddit,
        target_count=args.num_posts,
        request_sleep_sec=0.75,
    )
    print(f"Fetched {len(raw_posts)} posts from r/{args.subreddit}.")

    print("[3/5] Preprocessing posts...")
    prepped = preprocess_posts(raw_posts)
    for row in prepped:
        row["cluster_id"] = None
        row["centroid_distance"] = None

    inserted = upsert_posts(args.db_name, prepped)
    print(f"Upserted {inserted} posts.")

    print("[4/5] Clustering messages...")
    frame = load_posts_frame(args.db_name)
    clustered, cluster_keywords, vectorizer, emb, model = cluster_messages(
        frame,
        k_clusters=args.k_clusters,
    )

    clustered_rows = []
    for row in clustered.itertuples(index=False):
        clustered_rows.append(
            {
                "post_id": row.post_id,
                "fullname": f"t3_{row.post_id}",
                "created_utc": int(row.created_utc),
                "title": row.title,
                "selftext": "",
                "clean_text": row.clean_text,
                "keywords": row.keywords if isinstance(row.keywords, str) else "",
                "topic_hint": row.topic_hint if isinstance(row.topic_hint, str) else "general",
                "author_masked": row.author_masked if isinstance(row.author_masked, str) else "[deleted]",
                "score": int(row.score) if row.score is not None else 0,
                "num_comments": int(row.num_comments) if row.num_comments is not None else 0,
                "url": "",
                "permalink": row.permalink if isinstance(row.permalink, str) else "",
                "is_self": 1,
                "over_18": 0,
                "stickied": 0,
                "cluster_id": int(row.cluster_id),
                "centroid_distance": float(row.centroid_distance),
            }
        )
    upsert_posts(args.db_name, clustered_rows)
    update_cluster_keywords(args.db_name, cluster_keywords)

    near = posts_near_centroid(clustered, top_n=5)
    near_file = out_dir / "posts_near_centroid.csv"
    near.to_csv(near_file, index=False)

    clustered_file = out_dir / "clustered_posts.csv"
    clustered.to_csv(clustered_file, index=False)

    save_cluster_scatter(
        embeddings=emb,
        labels=clustered["cluster_id"].to_numpy(),
        output_path=str(out_dir / "cluster_scatter.png"),
    )

    print("[5/5] Done.")
    print(f"Saved clustered data: {clustered_file}")
    print(f"Saved centroid-near posts: {near_file}")
    print(f"Saved cluster plot: {out_dir / 'cluster_scatter.png'}")
    print("")
    print("Top keywords per cluster:")
    for cid, kws in cluster_keywords.items():
        print(f"- Cluster {cid}: {', '.join(kws[:8])}")


if __name__ == "__main__":
    main()
