"""Data preprocessing: clean extracted and scraped data before final storage."""

import re
from src.storage import get_connection


def _clean_text(val: str | None) -> str:
    if val is None:
        return "N/A"
    val = re.sub(r"<[^>]+>", "", val)          # strip HTML tags
    val = re.sub(r"[^\x20-\x7E]+", " ", val)  # non-printable / special chars -> space
    val = re.sub(r"\s{2,}", " ", val).strip()
    return val if val else "N/A"


def _clean_numeric(val) -> str:
    if val is None:
        return "0"
    s = str(val).strip().replace(",", "")
    if s == "" or s.lower() == "none":
        return "0"
    return s


def preprocess_wells():
    """Clean all wells and stimulation_data rows in-place."""
    conn = get_connection()
    cur = conn.cursor(dictionary=True)

    # --- Wells ---
    cur.execute("SELECT * FROM wells")
    wells = cur.fetchall()
    text_cols = ["api_number", "well_name", "operator", "county",
                 "field_name", "section", "township", "range_val",
                 "well_status", "well_type", "closest_city",
                 "oil_produced", "gas_produced"]

    for w in wells:
        updates = {}
        for col in text_cols:
            cleaned = _clean_text(w.get(col))
            if cleaned != (w.get(col) or ""):
                updates[col] = cleaned
        if w.get("latitude") is None:
            updates["latitude"] = 0
        if w.get("longitude") is None:
            updates["longitude"] = 0
        if updates:
            set_clause = ", ".join(f"{k} = %({k})s" for k in updates)
            updates["ndic"] = w["ndic_file_number"]
            cur.execute(
                f"UPDATE wells SET {set_clause} WHERE ndic_file_number = %(ndic)s",
                updates,
            )

    # --- Stimulation data ---
    cur.execute("SELECT * FROM stimulation_data")
    stim_rows = cur.fetchall()
    stim_text = ["formation", "treatment_type", "volume_units"]
    stim_num = ["stimulation_stages", "volume", "lbs_proppant",
                "max_treatment_pressure", "max_treatment_rate", "top_ft", "bottom_ft"]

    for s in stim_rows:
        updates = {}
        for col in stim_text:
            cleaned = _clean_text(s.get(col))
            if cleaned != (s.get(col) or ""):
                updates[col] = cleaned
        for col in stim_num:
            cleaned = _clean_numeric(s.get(col))
            if cleaned != str(s.get(col) or ""):
                updates[col] = cleaned
        if updates:
            set_clause = ", ".join(f"{k} = %({k})s" for k in updates)
            updates["sid"] = s["id"]
            cur.execute(
                f"UPDATE stimulation_data SET {set_clause} WHERE id = %(sid)s",
                updates,
            )

    conn.commit()
    cur.close()
    conn.close()
    print("Preprocessing complete.")
