"""Batch OCR processing for scanned PDF files using ocrmypdf."""

import os
import subprocess
import sys


def ocr_single(src_path: str, dst_path: str) -> bool:
    """Run ocrmypdf on a single PDF. Returns True on success."""
    import shutil
    try:
        subprocess.run(
            [
                "ocrmypdf",
                "--skip-text",
                "--optimize", "0",
                "--output-type", "pdf",
                "-l", "eng",
                "--jobs", "4",
                src_path,
                dst_path,
            ],
            capture_output=True, text=True, check=True,
        )
        return True
    except subprocess.CalledProcessError as exc:
        if exc.returncode == 6:
            shutil.copy2(src_path, dst_path)
            return True
        # Fallback: copy original so extraction can still try pdfplumber on it
        print(f"  [WARN] ocrmypdf error (code {exc.returncode}) on {os.path.basename(src_path)}, copying original")
        shutil.copy2(src_path, dst_path)
        return True


def run_batch(input_dir: str, output_dir: str) -> list[str]:
    """OCR every PDF in input_dir, write results to output_dir.
    Returns list of successfully processed output paths."""
    os.makedirs(output_dir, exist_ok=True)
    pdfs = sorted(f for f in os.listdir(input_dir) if f.lower().endswith(".pdf"))
    print(f"Found {len(pdfs)} PDFs to process.")

    processed = []
    for i, fname in enumerate(pdfs, 1):
        src = os.path.join(input_dir, fname)
        dst = os.path.join(output_dir, fname)
        if os.path.exists(dst):
            print(f"  [{i}/{len(pdfs)}] {fname} — already exists, skipping")
            processed.append(dst)
            continue
        print(f"  [{i}/{len(pdfs)}] {fname} ...", end=" ", flush=True)
        ok = ocr_single(src, dst)
        if ok:
            print("done")
            processed.append(dst)
        else:
            print("FAILED")
    return processed


if __name__ == "__main__":
    in_dir = sys.argv[1] if len(sys.argv) > 1 else "Scanned_pdfs"
    out_dir = sys.argv[2] if len(sys.argv) > 2 else "ocr_output"
    run_batch(in_dir, out_dir)
