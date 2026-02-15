from __future__ import annotations

import argparse
import select
import sys
import time
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from run_pipeline import main as run_pipeline_main
from src.cluster import cluster_messages
from src.storage import load_posts_frame


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Periodic Lab5 updater + interactive cluster search")
    p.add_argument("interval_minutes", type=float, help="Update interval in minutes")
    p.add_argument("--db-name", type=str, default="lab5_reddit")
    p.add_argument("--output-dir", type=str, default="outputs")
    p.add_argument("--subreddit", type=str, default="Fitness")
    p.add_argument("--num-posts", type=int, default=1000)
    p.add_argument("--k-clusters", type=int, default=6)
    return p.parse_args()


def _run_update(args: argparse.Namespace) -> None:
    # Reuse run_pipeline through argv to keep one code path.
    argv = [
        "run_pipeline.py",
        "--subreddit",
        args.subreddit,
        "--num-posts",
        str(args.num_posts),
        "--k-clusters",
        str(args.k_clusters),
        "--db-name",
        args.db_name,
        "--output-dir",
        args.output_dir,
    ]
    old_argv = sys.argv
    sys.argv = argv
    try:
        run_pipeline_main()
    finally:
        sys.argv = old_argv


def _visualize_matched_cluster(df_clustered, matched_cluster: int, output_path: str) -> None:
    subset = df_clustered[df_clustered["cluster_id"] == matched_cluster].copy()
    subset = subset.sort_values("centroid_distance", ascending=True).head(20)
    if subset.empty:
        return

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    y = np.arange(len(subset))
    plt.figure(figsize=(10, 6))
    plt.barh(y, subset["score"].to_numpy(), color="tab:blue", alpha=0.8)
    plt.yticks(y, [t[:60] + ("..." if len(t) > 60 else "") for t in subset["title"].tolist()])
    plt.gca().invert_yaxis()
    plt.xlabel("Reddit score")
    plt.title(f"Top messages in matched cluster {matched_cluster}")
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def _query_cluster(args: argparse.Namespace, query: str) -> None:
    df = load_posts_frame(args.db_name)
    if df.empty:
        print("No posts in DB yet. Wait for first update.")
        return

    clustered, cluster_keywords, vectorizer, emb, model = cluster_messages(df, k_clusters=args.k_clusters)
    q_vec = vectorizer.transform([query])
    cluster_id = int(model.predict(q_vec)[0])

    print(f"\nClosest cluster: {cluster_id}")
    print(f"Cluster keywords: {', '.join(cluster_keywords.get(cluster_id, [])[:10])}")
    subset = clustered[clustered["cluster_id"] == cluster_id].copy()
    if subset.empty:
        print("No posts in matched cluster.")
        return

    subset_vec = vectorizer.transform(subset["clean_text"].fillna("").tolist())
    sims = cosine_similarity(q_vec, subset_vec).ravel()
    subset["query_similarity"] = sims
    closest = subset.sort_values("query_similarity", ascending=False).head(5)
    print("Top related messages:")
    for row in closest.itertuples(index=False):
        print(f"- ({row.score}) {row.title}")
        print(f"  {row.permalink}")

    plot_path = Path(args.output_dir) / f"query_cluster_{cluster_id}.png"
    _visualize_matched_cluster(clustered, cluster_id, str(plot_path))
    print(f"Saved cluster view: {plot_path}\n")


def main() -> None:
    args = parse_args()
    interval_sec = max(10.0, args.interval_minutes * 60.0)
    next_update = 0.0

    print("Automation started.")
    print("Type keywords/message and press Enter to query nearest cluster.")
    print("Type 'quit' to stop.\n")

    while True:
        now = time.time()
        if now >= next_update:
            print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Fetching/processing/updating DB...")
            try:
                _run_update(args)
                print("Update cycle completed.")
            except Exception as exc:
                print(f"Update cycle failed: {exc}")
            next_update = time.time() + interval_sec
            print(f"Next update in {int(interval_sec)} seconds.")

        wait = max(0.2, min(1.0, next_update - time.time()))
        ready, _, _ = select.select([sys.stdin], [], [], wait)
        if ready:
            query = sys.stdin.readline().strip()
            if not query:
                continue
            if query.lower() in {"quit", "exit"}:
                print("Stopping automation.")
                break
            try:
                _query_cluster(args, query)
            except Exception as exc:
                print(f"Query failed: {exc}")


if __name__ == "__main__":
    main()
