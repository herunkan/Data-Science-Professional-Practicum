# Oil Wells Data Wrangling
Herun Kan

## Overview

This project extracts structured data from 77 scanned oil-well PDFs, enriches it with web-scraped information from drillingedge.com, stores everything in MySQL, and displays an interactive map of well locations using Leaflet.js served by Apache.

## Prerequisites

Refer to requirements.txt

| Tool | Install |
|------|---------|
| Python 3.10+ | pre-installed (conda base) |
| Tesseract OCR | `brew install tesseract` |
| MySQL 8+ | `brew install mysql` (or existing install) |
| Apache httpd | pre-installed on macOS |

## Setup

```bash
cd Lab6

# 1. Install Python packages
pip install -r requirements.txt

# 2. Configure MySQL credentials
cp .env.example .env   # edit with your credentials
# (or verify .env already has correct MYSQL_USER, MYSQL_PASSWORD, etc.)

# 3. Verify Tesseract
tesseract --version

# 4. Verify MySQL is running
mysql -u root -p -e "SELECT 1"
```

## Running the Pipeline

### Full pipeline (all steps)

```bash
python run_pipeline.py
```

This pipeline runs:
1. **OCR** — processes all PDFs in `Scanned_pdfs/`, outputs to `ocr_output/`
2. **Extract & Store** — parses text for well info + stimulation data, inserts into MySQL
3. **Web Scrape** — queries drillingedge.com for additional well data
4. **Preprocess** — cleans data, fills missing values with 0/N/A
5. **Export** — writes `web/data/wells.json` for the map

###  To Run individual steps

```bash
python run_pipeline.py --only ocr
python run_pipeline.py --only extract
python run_pipeline.py --only scrape
python run_pipeline.py --only preprocess
python run_pipeline.py --only export
```

### Skip steps

```bash
# Skip OCR if already processed
python run_pipeline.py --skip-ocr

# Skip web scraping
python run_pipeline.py --skip-scrape
```

## Viewing the Map

```bash
# Create symlink to Apache document root
sudo ln -s "$(pwd)/web" /Library/WebServer/Documents/lab6

# Start Apache
sudo apachectl start

# Open in browser
open http://localhost/lab6/
```

## Project Structure

```
Lab6/
├── Scanned_pdfs/          # Source scanned PDFs (77 files)
├── ocr_output/            # OCR-processed PDFs
├── src/
│   ├── ocr_batch.py       # Batch OCR with ocrmypdf
│   ├── extract_fields.py  # Regex-based field extraction
│   ├── storage.py         # MySQL schema & CRUD operations
│   ├── scrape_web.py      # drillingedge.com scraper
│   ├── preprocess.py      # Data cleaning & normalization
│   └── export_data.py     # MySQL -> JSON export
├── web/
│   ├── index.html         # Leaflet map page
│   ├── style.css          # Page styling
│   └── data/wells.json    # Exported well data (generated)
├── run_pipeline.py        # Main orchestrator
├── requirements.txt       # Python dependencies
├── .env                   # MySQL credentials (not committed)
└── README.md
```

## Database Schema

**wells** — one row per well
- `ndic_file_number` (PK), `api_number`, `well_name`, `operator`
- `latitude`, `longitude`, `county`, `field_name`, `section`, `township`, `range_val`
- `well_status`, `well_type`, `closest_city`, `oil_produced`, `gas_produced` (from web scraping)

**stimulation_data** — one or more rows per well
- `ndic_file_number` (FK), `formation`, `treatment_type`, `stimulation_stages`
- `volume`, `volume_units`, `lbs_proppant`, `max_treatment_pressure`, `max_treatment_rate`
