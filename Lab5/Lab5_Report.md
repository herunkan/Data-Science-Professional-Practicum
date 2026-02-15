# DSCI 560 - Lab 5 Report

**Team Name:** Group6 the Coach  
**Course:** DSCI 560 - Data Science Practicum  
**Lab:** Lab 5 - Web Scraping, Preprocessing, Clustering, and Real-Time Updates

## 1) Initial Setup

This project was implemented in Python on local macOS with MySQL storage.

Libraries used:
- `requests`, `beautifulsoup4` for scraping
- `pandas` for data processing
- `scikit-learn` for embedding + clustering
- `matplotlib` for visualization
- `mysql-connector-python` for DB writes/reads

## 2) Data Collection and Storage

### Topic selection

- Initial target was `r/Fitness`, then switched to `r/workout` for better topical consistency.
- Final analysis in this report is based on `r/workout`.

### Scraping method

- Used BeautifulSoup route by crawling `old.reddit.com`.
- Pagination uses repeated "next" links to handle larger requests (no single-page limit dependency).
- Script accepts post count parameter (`--num-posts`).

### Database storage

- MySQL database: `lab5_workout`
- Main table: `reddit_posts`
- Supporting table: `cluster_keywords`
- Upsert behavior ensures repeated runs update existing rows rather than duplicate inserts.

## 3) Data Preprocessing

Preprocessing pipeline includes:
- HTML tag removal and text normalization
- URL/special-character cleanup
- Username masking (`author_masked`) for privacy
- Keyword extraction per message
- Topic hint generation

Additional filtering to improve quality:
- Promoted/sponsored posts filtered
- Non-comment permalink entries filtered
- Low-signal junk rows reduced before clustering

## 4) Forum Analysis and Clustering

### Embedding (message abstraction)

- TF-IDF vectorization (`TfidfVectorizer`) was used to map each cleaned message into fixed-dimensional vectors.
- This satisfies the required embedding step before clustering.

### Clustering

- K-Means clustering with `K=8`
- Outputs generated:
  - cluster assignment per message
  - centroid distance per message
  - posts nearest each centroid
  - 2D cluster scatter plot via PCA projection

### Representative cluster keywords (from current run)

- Cluster 0: `workout, routine, good, beginner, body, day, split, after`
- Cluster 1: `can, should, anyone, workouts, only, does, training, leg`
- Cluster 2: `gym, how, going, got, anyone, else, should, most`
- Cluster 3: `how, guys, workout, why, exercise, think, actually, body`
- Cluster 4: `day, split, how, can, routine, workout, help, any`
- Cluster 5: `advice, need, help, body, gym, weight, stuck, split`
- Cluster 6: `out, working, how, work, after, lost, help, back`
- Cluster 7: `how, routine, weight, muscle, start, can, day, best`

## 5) Automation

Real-time/periodic update implemented in `automation.py`:
- Accepts interval in minutes (example: `python3 automation.py 5 ...`)
- Runs recurring fetch -> preprocess -> DB update -> recluster cycles
- While waiting between cycles, terminal accepts free-text query
- Query is mapped to closest cluster and top related posts are shown
- Query-specific visualization is saved (`query_cluster_<id>.png`)

## 6) Results and Artifacts

### Runtime result summary (latest run)

- Requested posts: **500**
- Stored/processed rows in output CSV: **1277**
- Number of clusters: **8**

### Generated files

- `outputs/clustered_posts.csv`
- `outputs/posts_near_centroid.csv`
- `outputs/cluster_scatter.png`
- `outputs/query_cluster_0.png`
- `outputs/query_cluster_1.png`
- `outputs/query_cluster_2.png`


These screenshots capture the command-line execution and interactive query behavior.

## 7) Team Discussions

Discussion minutes template is prepared in:
- `meeting_notes_template.md`

Final deliverable file name:
- `meeting_notes_group6_the_coach.pdf`

## 8) Challenges and Decisions
- BeautifulSoup approach was used as Reddit API registration can be avoided.
- `r/Fitness` content was heavy with recurring threads/noise, so subreddit choice was switched to `r/workout`.

## 9) Conclusion

The solution fulfills the lab workflow end-to-end: scraping, preprocessing, MySQL storage, vector abstraction, clustering, nearest-centroid extraction, and periodic automation with interactive querying. The resulting clusters for `r/workout` are meaningfully workout-related and suitable for report visualizations and discussion.
