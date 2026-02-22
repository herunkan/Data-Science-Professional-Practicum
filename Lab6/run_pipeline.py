#!/usr/bin/env python3
"""Lab 6 pipeline: OCR -> Extract -> Store -> Scrape -> Preprocess -> Export."""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

from src.ocr_batch import run_batch
from src.extract_fields import extract_from_pdf
from src.storage import init_db, upsert_well, insert_stimulation, get_wells_needing_scrape, update_web_scraped
from src.scrape_web import scrape_single_well, DELAY
import time
from src.preprocess import preprocess_wells
from src.export_data import export_to_json


def step_ocr(input_dir, output_dir):
    print("\n=== Step 1: OCR Batch Processing ===")
    processed = run_batch(input_dir, output_dir)
    print(f"OCR complete: {len(processed)} PDFs processed.\n")
    return processed


def step_extract_and_store(ocr_dir):
    print("\n=== Step 2: Extract Fields & Store in MySQL ===")
    init_db()
    pdfs = sorted(f for f in os.listdir(ocr_dir) if f.lower().endswith(".pdf"))
    success, fail = 0, 0
    for i, fname in enumerate(pdfs, 1):
        path = os.path.join(ocr_dir, fname)
        print(f"  [{i}/{len(pdfs)}] {fname} ...", end=" ")
        try:
            result = extract_from_pdf(path)
            well = result["well_info"]
            stim = result["stimulation_data"]
            if not well["ndic_file_number"]:
                print(f"no NDIC number found, skipping")
                fail += 1
                continue
            upsert_well(well)
            insert_stimulation(well["ndic_file_number"], stim)
            n_stim = len(stim)
            print(f"ok (API={well['api_number'] or '?'}, stim_records={n_stim}, text={result['raw_text_length']} chars)")
            success += 1
        except Exception as exc:
            print(f"ERROR: {exc}")
            fail += 1
    print(f"Extraction done: {success} succeeded, {fail} failed.\n")


def step_scrape(skip=False):
    if skip:
        print("\n=== Step 3: Web Scraping — SKIPPED ===\n")
        return
    print("\n=== Step 3: Scrape drillingedge.com ===")
    wells = get_wells_needing_scrape()
    print(f"  {len(wells)} wells need web data.")
    for i, w in enumerate(wells, 1):
        ndic = w["ndic_file_number"]
        api = w.get("api_number")
        name = w.get("well_name")
        county = w.get("county")
        print(f"  [{i}/{len(wells)}] NDIC {ndic} (API={api}) ...", end=" ", flush=True)
        data = scrape_single_well(api, name, county)
        found = sum(1 for v in data.values() if v)
        print(f"{found}/5 fields found")
        update_web_scraped(ndic, data)
        time.sleep(DELAY)
    print("Web scraping complete.\n")


def step_preprocess():
    print("\n=== Step 4: Preprocessing ===")
    preprocess_wells()


def step_export():
    print("\n=== Step 5: Export to JSON ===")
    export_to_json()


def main():
    parser = argparse.ArgumentParser(description="Lab 6 Pipeline")
    parser.add_argument("--input-dir", default="Scanned_pdfs", help="Folder with source PDFs")
    parser.add_argument("--ocr-dir", default="ocr_output", help="Folder for OCR output")
    parser.add_argument("--skip-ocr", action="store_true", help="Skip OCR step (use existing ocr_output)")
    parser.add_argument("--skip-scrape", action="store_true", help="Skip web scraping step")
    parser.add_argument("--only", choices=["ocr", "extract", "scrape", "preprocess", "export"],
                        help="Run only a single step")
    args = parser.parse_args()

    if args.only == "ocr":
        step_ocr(args.input_dir, args.ocr_dir)
    elif args.only == "extract":
        step_extract_and_store(args.ocr_dir)
    elif args.only == "scrape":
        step_scrape()
    elif args.only == "preprocess":
        step_preprocess()
    elif args.only == "export":
        step_export()
    else:
        if not args.skip_ocr:
            step_ocr(args.input_dir, args.ocr_dir)
        step_extract_and_store(args.ocr_dir)
        step_scrape(skip=args.skip_scrape)
        step_preprocess()
        step_export()

    print("\n=== Pipeline complete ===")


if __name__ == "__main__":
    main()
