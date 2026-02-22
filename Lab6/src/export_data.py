"""Export well data from MySQL to JSON for the web page."""

import json
import os
import decimal
import datetime
from src.storage import get_all_wells, get_stimulation_for_well


def _json_default(obj):
    if isinstance(obj, decimal.Decimal):
        return float(obj)
    if isinstance(obj, (datetime.date, datetime.datetime)):
        return obj.isoformat()
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


def export_to_json(output_path: str = None):
    if output_path is None:
        output_path = os.path.join(os.path.dirname(__file__), "..", "web", "data", "wells.json")

    wells = get_all_wells()
    result = []
    for w in wells:
        lat = float(w["latitude"]) if w["latitude"] else 0
        lon = float(w["longitude"]) if w["longitude"] else 0
        stim = get_stimulation_for_well(w["ndic_file_number"])
        entry = {
            "ndic_file_number": w["ndic_file_number"],
            "api_number": w.get("api_number") or "N/A",
            "well_name": w.get("well_name") or "N/A",
            "operator": w.get("operator") or "N/A",
            "latitude": lat,
            "longitude": lon,
            "county": w.get("county") or "N/A",
            "field_name": w.get("field_name") or "N/A",
            "section": w.get("section") or "N/A",
            "township": w.get("township") or "N/A",
            "range_val": w.get("range_val") or "N/A",
            "well_status": w.get("well_status") or "N/A",
            "well_type": w.get("well_type") or "N/A",
            "closest_city": w.get("closest_city") or "N/A",
            "oil_produced": w.get("oil_produced") or "N/A",
            "gas_produced": w.get("gas_produced") or "N/A",
            "stimulation_data": [],
        }
        for s in stim:
            entry["stimulation_data"].append({
                "formation": s.get("formation") or "N/A",
                "treatment_type": s.get("treatment_type") or "N/A",
                "stimulation_stages": s.get("stimulation_stages") or "0",
                "volume": s.get("volume") or "0",
                "volume_units": s.get("volume_units") or "N/A",
                "lbs_proppant": s.get("lbs_proppant") or "0",
                "max_treatment_pressure": s.get("max_treatment_pressure") or "0",
                "max_treatment_rate": s.get("max_treatment_rate") or "0",
                "top_ft": s.get("top_ft") or "0",
                "bottom_ft": s.get("bottom_ft") or "0",
            })
        result.append(entry)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2, default=_json_default)
    print(f"Exported {len(result)} wells to {output_path}")
    return output_path


if __name__ == "__main__":
    export_to_json()
