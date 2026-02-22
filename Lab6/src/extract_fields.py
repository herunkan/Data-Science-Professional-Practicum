"""Extract structured well information and stimulation data from OCR'd PDF text."""

import os
import re
import pdfplumber


ND_COUNTIES = {
    "mckenzie", "williams", "mountrail", "dunn", "burke", "divide",
    "bottineau", "renville", "ward", "mclean", "mercer", "oliver",
    "morton", "stark", "billings", "golden valley", "slope", "bowman",
    "adams", "hettinger", "grant", "sioux", "emmons", "burleigh",
    "kidder", "stutsman", "wells", "sheridan", "foster", "eddy",
    "benson", "ramsey", "nelson", "griggs", "steele", "traill",
    "cass", "barnes", "ransom", "richland", "sargent", "dickey",
    "lamoure", "logan", "mcintosh", "mchenry", "pierce", "rolette",
    "towner", "cavalier", "pembina", "walsh", "grand forks",
}

COMPANY_WORDS = {"company", "inc", "corp", "llc", "petroleum", "oil",
                 "energy", "resources", "exploration", "oasis", "continental",
                 "slawson", "whiting", "hess", "marathon", "statoil", "burlington",
                 "conocophillips", "eog", "qep", "sm energy", "liberty"}


def _dms_to_decimal(degrees: float, minutes: float, seconds: float, direction: str) -> float:
    dec = abs(degrees) + minutes / 60.0 + seconds / 3600.0
    if direction.upper() in ("S", "W"):
        dec = -dec
    return round(dec, 6)


_DMS_PATTERNS = [
    # N 48° 01' 29.869"  or  48° 01' 29.869" N
    re.compile(r'([NSEW])\s*(\d{2,3})\s*[\xb0\u02da\u00ba°]\s*(\d{1,2})\s*[\x27\u2018\u2019\u2032\']\s*([\d.]+)\s*[\x22\u201c\u201d\u2033\"]*', re.IGNORECASE),
    re.compile(r'(\d{2,3})\s*[\xb0\u02da\u00ba°]\s*(\d{1,2})\s*[\x27\u2018\u2019\u2032\']\s*([\d.]+)\s*[\x22\u201c\u201d\u2033\"]*\s*([NSEW])', re.IGNORECASE),
]
_DECIMAL_COORD = re.compile(r'(?:Lat(?:itude)?|Lon(?:gitude)?)\s*[:=]?\s*(-?\d{2,3}\.\d{3,})')


def _parse_coordinates(text: str):
    """Return (lat, lon) as decimals, or (None, None).
    Takes the first valid pair found (surface location)."""
    lat, lon = None, None
    for pat in _DMS_PATTERNS:
        for m in pat.finditer(text):
            groups = m.groups()
            if len(groups) == 4:
                try:
                    if groups[0].upper() in "NSEW":
                        d_str = groups[0]
                        deg, mn, sec = float(groups[1]), float(groups[2]), float(groups[3])
                    else:
                        deg, mn, sec = float(groups[0]), float(groups[1]), float(groups[2])
                        d_str = groups[3]
                except (ValueError, TypeError):
                    continue
                try:
                    val = _dms_to_decimal(deg, mn, sec, d_str)
                except (ValueError, TypeError):
                    continue
                if d_str.upper() in ("N", "S") and 45 < abs(val) < 50 and lat is None:
                    lat = val
                elif d_str.upper() in ("E", "W") and 100 < abs(val) < 106 and lon is None:
                    lon = val
        if lat and lon:
            return lat, lon

    for m in _DECIMAL_COORD.finditer(text):
        v = float(m.group(1))
        if 45 < abs(v) < 50 and lat is None:
            lat = v
        elif 100 < abs(v) < 106 and lon is None:
            lon = -abs(v)
    return lat, lon


# ---------------------------------------------------------------------------
# Well-info extraction — multi-strategy
# ---------------------------------------------------------------------------

def _extract_from_spill_report(text: str) -> dict:
    """Try the Spill/Incident Report format (cleanest key:value pairs)."""
    info = {}
    m = re.search(r'Well\s*Operator\s*:\s*(.+)', text, re.IGNORECASE)
    if m:
        info["operator"] = m.group(1).strip()
    m = re.search(r'(?:Well\s*(?:or\s*)?Facility\s*Name|Well\s*Name)\s*:\s*(.+)', text, re.IGNORECASE)
    if m:
        val = m.group(1).strip()
        if re.search(r'\d', val) and len(val) < 80:
            info["well_name"] = val
    m = re.search(r'NDIC\s*(?:File\s*)?(?:Number|No\.?|#)\s*:\s*(\d+)', text, re.IGNORECASE)
    if m:
        info["ndic_file_number"] = m.group(1)
    m = re.search(r'Field\s*Name\s*:\s*([A-Z][A-Za-z ]+)', text, re.IGNORECASE)
    if m:
        info["field_name"] = m.group(1).strip()
    m = re.search(r'County\s*:\s*([A-Z][A-Za-z ]+)', text, re.IGNORECASE)
    if m:
        val = m.group(1).strip()
        if val.lower() in ND_COUNTIES:
            info["county"] = val
    for pat in [
        re.compile(r'Section\s*:\s*(\d+)', re.IGNORECASE),
        re.compile(r'Township\s*:\s*(\d+N?)', re.IGNORECASE),
        re.compile(r'Range\s*:\s*(\d+W?)', re.IGNORECASE),
    ]:
        m = pat.search(text)
        if m:
            key = pat.pattern.split(r'\s')[0].lower().replace("section", "section").replace("township", "township").replace("range", "range_val")
            if "section" in pat.pattern.lower():
                info["section"] = m.group(1)
            elif "township" in pat.pattern.lower():
                info["township"] = m.group(1)
            elif "range" in pat.pattern.lower():
                info["range_val"] = m.group(1)
    return info


def _extract_from_completion_form(text: str) -> dict:
    """Try the completion/recompletion report format."""
    info = {}
    # Well Name and Number followed by actual name on next content
    m = re.search(r'Well\s*Name\s*(?:and\s*Number)?\s*[\|\[\]]*\s*(?:Spacing[^\n]*)?\n\s*([A-Z][A-Z0-9\- ]+\d+[A-Z]*H?\b)', text, re.IGNORECASE)
    if m:
        info["well_name"] = m.group(1).strip()
    # Operator followed by company name on next line or same line
    for m in re.finditer(r'Operator\s*(?:Telephone[^\n]*)?\n\s*(.+)', text, re.IGNORECASE):
        val = m.group(1).strip().split('\n')[0]
        if any(w in val.lower() for w in COMPANY_WORDS) and len(val) < 100:
            info["operator"] = val
            break
    # Section/Township/Range from tabular format
    m = re.search(r'(?:SESE|SWSE|SESW|SWSW|NWSE|NWSW|NWNE|NENE|SENW|NESW|NENW|SWNW)\s+(\d+)\s+(\d+)\s*N?\s+(\d+)\s*W?', text)
    if m:
        info.setdefault("section", m.group(1))
        info.setdefault("township", m.group(2))
        info.setdefault("range_val", m.group(3))
    # S/T/R pattern
    m = re.search(r'(\d+)-(\d+)N-(\d+)W', text)
    if m:
        info.setdefault("section", m.group(1))
        info.setdefault("township", m.group(2))
        info.setdefault("range_val", m.group(3))
    # County from "County" field near known county names
    for county in ND_COUNTIES:
        if re.search(rf'\b{county}\s*county\b', text, re.IGNORECASE) or \
           re.search(rf'County\s*[:|\n]\s*{county}\b', text, re.IGNORECASE):
            info.setdefault("county", county.title())
            break
    # Field from "Field\n" or "Field:" pattern
    m = re.search(r'Field\s*(?:Name)?\s*\n\s*([A-Z][A-Z ]+)\b', text)
    if m:
        val = m.group(1).strip()
        if len(val) > 1 and val.lower() not in {"inspector", "pool", "gas", "oil", "well", "poo"}:
            info.setdefault("field_name", val)
    return info


def _extract_from_directional_survey(text: str) -> dict:
    """Try directional survey certification format for API and coordinates."""
    info = {}
    m = re.search(r'API\s*(?:Number|#|No)?\s*[:=]?\s*(33[-\s]?\d{3}[-\s]?\d{5})', text, re.IGNORECASE)
    if m:
        info["api_number"] = m.group(1).replace(" ", "-")
    if not info.get("api_number"):
        m = re.search(r'\b(33-\d{3}-\d{5})\b', text)
        if m:
            info["api_number"] = m.group(1)
    return info


def _extract_from_verbal_approval(text: str) -> dict:
    """Try Verbal Approval format for well name, operator, field, etc."""
    info = {}
    m = re.search(r'Well\s*Name\s*Inspector\n\s*([A-Z][A-Z0-9\- ]+\d[A-Z]*H?\b)', text, re.IGNORECASE)
    if m:
        info["well_name"] = m.group(1).strip()
    m = re.search(r'OPERATOR\s*\n\s*Operator\s*Representative.*\n\s*([A-Z][A-Z &,.\-]+(?:INC|LLC|COMPANY|CORP))', text, re.IGNORECASE)
    if m:
        info["operator"] = m.group(1).strip()
    m = re.search(r'Field\s*\n\s*\d+\s+Feet.*\n\s*([A-Z][A-Z ]+)\b', text)
    if m:
        val = m.group(1).strip()
        if val.lower() not in {"inspector", "pool", "gas", "oil", "well"}:
            info["field_name"] = val
    return info


def extract_well_info(text: str, filename: str) -> dict:
    """Merge results from multiple extraction strategies, prioritizing cleaner sources."""
    result = {
        "ndic_file_number": None,
        "api_number": None,
        "well_name": None,
        "operator": None,
        "latitude": None,
        "longitude": None,
        "county": None,
        "field_name": None,
        "section": None,
        "township": None,
        "range_val": None,
    }

    # NDIC from filename always
    m = re.search(r'W(\d+)', filename)
    if m:
        result["ndic_file_number"] = m.group(1)

    # Coordinates — all wells are in ND (Northern/Western hemisphere)
    lat, lon = _parse_coordinates(text)
    if lat is not None:
        result["latitude"] = abs(lat)
    if lon is not None:
        result["longitude"] = -abs(lon)

    # Try strategies in priority order (cleanest first)
    for strategy in [_extract_from_spill_report,
                     _extract_from_directional_survey,
                     _extract_from_completion_form,
                     _extract_from_verbal_approval]:
        partial = strategy(text)
        for k, v in partial.items():
            if v and not result.get(k):
                result[k] = v

    return result


# ---------------------------------------------------------------------------
# Stimulation-data extraction
# ---------------------------------------------------------------------------

_FORMATION_NAMES = ["middle bakken", "bakken", "three forks", "lodgepole",
                    "mission canyon", "duperow", "birdbear", "nisku"]


def extract_stimulation_data(text: str) -> list[dict]:
    """Extract stimulation records from completion report pages."""
    records = []

    stim_section = re.search(
        r'(?:Well\s*Specific\s*)?Stimulat.*?(?=CORES\s*CUT|Drill\s*Stem\s*Test|$)',
        text, re.IGNORECASE | re.DOTALL,
    )
    if not stim_section:
        stim_section = re.search(
            r'Stimulat.*?Formation.*?(?:Barrels|PSI|BBL|proppant)',
            text, re.IGNORECASE | re.DOTALL,
        )

    search_text = stim_section.group(0) if stim_section else ""

    formation = None
    for fn in _FORMATION_NAMES:
        if re.search(rf'\b{fn}\b', search_text, re.IGNORECASE):
            formation = fn.title()
            break
    if not formation:
        m = re.search(r'Stimulated?\s*Formation?\s*[:\n|]+\s*([A-Za-z ]+?)(?:\s*(?:\d|Top|Bottom|Stimulat|Volume|$))', search_text, re.IGNORECASE)
        if m:
            formation = m.group(1).strip()

    if not formation:
        return []

    treatment = None
    m = re.search(r'(?:Sand\s*Frac|Acid(?:izing)?|Frac(?:ture)?|Hybrid|Sliding\s*Sleeve)', search_text, re.IGNORECASE)
    if m:
        treatment = m.group(0).strip()

    stages = None
    m = re.search(r'(?:Stimulation\s*)?Stages?\s*[:\|\[\]]*\s*(\d+)', search_text, re.IGNORECASE)
    if m:
        stages = m.group(1)

    volume = None
    m = re.search(r'(\d{3,})\s*(?:_+\s*)?(?:Barrels|bbls?|gallons)', search_text, re.IGNORECASE)
    if m:
        volume = m.group(1)

    volume_units = None
    m = re.search(r'(Barrels|Gallons)', search_text, re.IGNORECASE)
    if m:
        volume_units = m.group(1).title()

    proppant = None
    m = re.search(r'(?:Lbs|Pounds?)\s*Proppant\s*[:\|\[\]]*\s*([\d,]+)', search_text, re.IGNORECASE)
    if not m:
        # Fallback: look for large number between treatment type and pressure
        m = re.search(r'(?:Sand\s*Frac|Acid|Frac)\s+[\d.]+\s*%?\s*([\d,]{4,})', search_text, re.IGNORECASE)
    if not m:
        # Fallback: look for "# of" or large number near "Sand" or "Mesh"
        m = re.search(r'([\d,]{5,})(?:\s*#|\s*(?:of|lbs?|pounds?))', search_text, re.IGNORECASE)
    if m:
        proppant = m.group(1).replace(",", "")

    max_pressure = None
    m = re.search(r'(?:Maximum\s*)?(?:Treatment\s*)?Pressure\s*\(?PSI?\)?\s*[:\|\[\]]*\s*([\d,]+)', search_text, re.IGNORECASE)
    if not m:
        m = re.search(r'(\d{4,})\s*(?:PSI|psi)', search_text)
    if m:
        max_pressure = m.group(1).replace(",", "")

    max_rate = None
    m = re.search(r'(?:Maximum\s*)?(?:Treatment\s*)?Rate\s*\(?(?:BBLS?|BBL)\s*/\s*Min\)?\s*[:\|\[\]]*\s*([\d,.]+)', search_text, re.IGNORECASE)
    if m:
        max_rate = m.group(1).replace(",", "")

    top_ft = None
    m = re.search(r'Top\s*\(?Ft?\)?\s*[:\|\[\]]*\s*Bottom.*?\n[^\n]*?(\d{4,})\s*[\|\[\]\s]+\s*(\d{4,})', search_text, re.IGNORECASE)
    if m:
        top_ft = m.group(1)
        bottom_ft = m.group(2)
    else:
        bottom_ft = None

    records.append({
        "formation": formation,
        "treatment_type": treatment,
        "stimulation_stages": stages,
        "volume": volume,
        "volume_units": volume_units,
        "lbs_proppant": proppant,
        "max_treatment_pressure": max_pressure,
        "max_treatment_rate": max_rate,
        "top_ft": top_ft,
        "bottom_ft": bottom_ft,
    })
    return records


# ---------------------------------------------------------------------------
# Full PDF extraction
# ---------------------------------------------------------------------------

def extract_from_pdf(pdf_path: str) -> dict:
    """Extract all text from a PDF, then parse well info and stimulation data."""
    text_parts = []
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                t = page.extract_text()
                if t:
                    text_parts.append(t)
    except Exception as exc:
        print(f"  [WARN] pdfplumber failed on {pdf_path}: {exc}")

    full_text = "\n".join(text_parts)
    filename = os.path.basename(pdf_path)

    well = extract_well_info(full_text, filename)
    stim = extract_stimulation_data(full_text)

    return {"well_info": well, "stimulation_data": stim, "raw_text_length": len(full_text)}
