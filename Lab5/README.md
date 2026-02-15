# Lab 5 - Reddit Scraping, Preprocessing, Clustering, and Automation

This implementation uses the BeautifulSoup route in the assignment resources:

- https://www.datacamp.com/tutorial/scraping-reddit-python-scrapy

Target subreddit used here: `r/Fitness`.

## What this solution includes

1. **Data collection with BeautifulSoup**
   - Scrapes `r/Fitness` posts from `old.reddit.com`.
   - Accepts post count as an argument (`--num-posts`).
   - Handles large requests by repeatedly following pagination links (`next-button`) across pages.
2. **Data preprocessing**
   - Removes HTML tags and special characters.
   - Masks usernames for privacy.
   - Extracts keywords and topic hints.
3. **Storage**
   - Stores cleaned records in **MySQL** (as required by the lab PDF).
   - Saves cluster metadata and nearest-to-centroid records.
4. **Forum analysis + clustering**
   - Embedding via TF-IDF vectors.
   - Clustering via KMeans.
   - Outputs nearest messages to each centroid.
   - Produces cluster visualization.
5. **Automation + interactive query**
   - Runs periodic update cycles by interval (minutes).
   - While idle, accepts keyword/message input and returns closest cluster with a plot.

## Project structure

- `run_pipeline.py` - one-shot fetch/process/store/cluster run
- `automation.py` - interval scheduler + interactive cluster query
- `src/scraper.py` - BeautifulSoup scraping logic
- `src/preprocess.py` - text cleaning, user masking, keyword extraction
- `src/storage.py` - MySQL schema + upsert utilities
- `src/cluster.py` - embedding, clustering, centroid-near posts, plots
- `meeting_notes_template.md` - fill-in template for team discussion notes
- `report_outline.md` - report structure template

## Setup

```bash
cd Lab5
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Create `.env` from template and fill your MySQL credentials:

```bash
cp .env.example .env
```

Example `.env`:

```bash
MYSQL_HOST=127.0.0.1
MYSQL_PORT=3306
MYSQL_USER=root
MYSQL_PASSWORD=your_mysql_password
```

## Run pipeline

```bash
python3 run_pipeline.py --subreddit workout --num-posts 500 --k-clusters 6 --db-name lab5_reddit
```

Large request example:

```bash
python3 run_pipeline.py --subreddit workout --num-posts 5000 --k-clusters 8 --db-name lab5_reddit
```

Outputs:

- `outputs/clustered_posts.csv`
- `outputs/posts_near_centroid.csv`
- `outputs/cluster_scatter.png`
- MySQL schema/database: `lab5_reddit` (or name provided via `--db-name`)

## Run periodic automation + interactive query

Run every 5 minutes:

```bash
python3 automation.py 5 --subreddit Fitness --num-posts 1500 --k-clusters 8 --db-name lab5_reddit
```

While it is idle, type keywords in terminal, for example:

```text
protein intake for muscle gain
home workout for beginners
```

It prints closest cluster, representative posts, and saves:

- `outputs/query_cluster_<id>.png`

## Notes on assignment mapping

- **Large request handling (5000 posts / long runtime):**
  implemented by iterative pagination over HTML pages with repeated HTTP requests.
- **Preprocessing requirements:**
  HTML/special-char removal, privacy masking, keyword extraction included.
- **Clustering requirements:**
  vector abstraction (TF-IDF), KMeans clustering, centroid-nearest posts, and visualization included.
- **Automation requirement:**
  interval-based updates with interactive nearest-cluster query implemented.

