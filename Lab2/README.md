# Lab 2 — Multi-Source Data Collection

## 📋 Overview

This lab demonstrates data collection techniques from multiple heterogeneous sources within the **Fitness and Exercise Science** domain. The project aggregates data from local files, web scraping, and public APIs.

---

## 📂 Structure

```
Lab2/
├── Data/
│   ├── megaGymDataset.csv              # Source exercise database
│   └── collected_fitness_data/
│       ├── exercise_database.csv        # Processed exercise data
│       ├── reddit_posts.csv             # Scraped Reddit fitness posts
│       ├── pubmed_articles.csv          # Academic articles from PubMed
│       └── collection_summary.csv       # Collection statistics
├── Scipts/
│   └── lab2-solution.py                 # Main data collection script
└── lab2-s26.pdf                         # Lab assignment instructions
```

---

## 🗂️ Data Sources

### 1. Local CSV — Exercise Database
- **Source**: megaGymDataset.csv
- **Content**: Comprehensive exercise database with attributes including:
  - Exercise title and description
  - Target body part
  - Equipment required
  - Difficulty level
  - Exercise type

### 2. Reddit — r/Fitness Community
- **Method**: JSON API / Web scraping (old.reddit.com fallback)
- **Content**: Recent posts from the fitness subreddit
- **Fields**: Title, author, score, comments, timestamps, URLs

### 3. PubMed — Academic Articles
- **Method**: NCBI E-utilities API
- **Search Terms**: "resistance training", "exercise physiology", "strength training"
- **Content**: Peer-reviewed research articles
- **Fields**: PMID, title, authors, journal, publication date

---

## 🚀 Usage

```bash
# Navigate to scripts directory
cd Lab2/Scipts

# Run the data collection script
python lab2-solution.py
```

### Prerequisites

```bash
pip install pandas requests beautifulsoup4
```

---

## ⚙️ Features

| Feature | Description |
|---------|-------------|
| **Multi-source Collection** | Aggregates data from 3 distinct source types |
| **Error Handling** | Graceful fallbacks for API failures |
| **Rate Limiting** | Respectful delays between API calls |
| **Data Validation** | Structured output with consistent schemas |
| **Summary Generation** | Automatic statistics and collection reports |

---

## 📊 Output

The script generates a summary report showing:
- Total records collected per source
- Success/failure status for each data source
- File sizes and column counts
- Collection timestamps

---

## 🔧 Technical Details

### API Endpoints Used

| Source | Endpoint |
|--------|----------|
| Reddit | `https://www.reddit.com/r/Fitness.json` |
| PubMed Search | `https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi` |
| PubMed Summary | `https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi` |

### Headers & Best Practices

- Custom User-Agent headers to avoid blocking
- JSON preference for structured responses
- Timeout handling for network reliability
- Polite crawling with delays between requests

---

## 📄 Sample Output

```
COLLECTION SUMMARY
======================================================================
Collection Results:
- Total sources attempted: 3
- Successful collections: 3/3
- Total records collected: XXX

Detailed breakdown:
         Source              Filename  Records  Columns  Size_kb     Status
Exercise Database  exercise_database.csv     XXX       X     XX.XX  Success ✓
     Reddit Posts       reddit_posts.csv      XX       X      X.XX  Success ✓
  Pubmed Articles    pubmed_articles.csv      XX       X      X.XX  Success ✓
```

