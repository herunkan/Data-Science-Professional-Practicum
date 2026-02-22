"""Scrape additional well information from drillingedge.com."""

import re
import time
import requests
from bs4 import BeautifulSoup

BASE_URL = "https://www.drillingedge.com"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                  "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
}
DELAY = 2


def _make_slug(name: str) -> str:
    return re.sub(r'[^a-z0-9]+', '-', name.lower()).strip('-')


def _county_slug(county: str | None) -> str:
    if not county:
        return "mckenzie-county"
    c = county.lower().strip()
    if "williams" in c:
        return "williams-county"
    if "mountrail" in c:
        return "mountrail-county"
    if "dunn" in c:
        return "dunn-county"
    return f"{_make_slug(c)}-county"


def _extract_dl_fields(soup) -> dict:
    """Parse definition-list-style or table-row key/value pairs."""
    data = {}
    text = soup.get_text(" ", strip=True)

    # Well Status
    m = re.search(r'Well\s*Status\s*\n?\s*(.+?)(?:\n|Well\s*Type)', text)
    if m:
        val = m.group(1).strip()
        if val.lower() not in ("members only", ""):
            data["well_status"] = val[:100]

    # Well Type
    m = re.search(r'Well\s*Type\s*\n?\s*(.+?)(?:\n|Township)', text)
    if m:
        val = m.group(1).strip()
        if val.lower() not in ("members only", ""):
            data["well_type"] = val[:100]

    # Closest City
    m = re.search(r'Closest\s*City\s*\n?\s*(.+?)(?:\n|Latitude)', text)
    if m:
        val = m.group(1).strip()
        if val.lower() not in ("members only", ""):
            data["closest_city"] = val[:200]

    # Oil Production
    m = re.search(r'Total\s*Oil\s*Prod\s*\n?\s*([\d,]+)', text)
    if m:
        data["oil_produced"] = m.group(1).strip()[:100]

    # Gas Production
    m = re.search(r'Total\s*Gas\s*Prod\s*\n?\s*([\d,]+)', text)
    if m:
        data["gas_produced"] = m.group(1).strip()[:100]

    return data


def scrape_well_page(url: str) -> dict:
    info = {
        "well_status": None, "well_type": None,
        "closest_city": None, "oil_produced": None, "gas_produced": None,
    }
    try:
        resp = requests.get(url, headers=HEADERS, timeout=15)
        if resp.status_code != 200:
            return info
    except requests.RequestException:
        return info

    soup = BeautifulSoup(resp.text, "html.parser")

    # Strategy 1: parse table rows
    for tr in soup.find_all("tr"):
        cells = tr.find_all(["td", "th"])
        if len(cells) >= 2:
            label = cells[0].get_text(strip=True).lower()
            value = cells[1].get_text(strip=True)
            if value.lower() == "members only":
                continue
            if "well status" in label and not info["well_status"]:
                info["well_status"] = value[:100]
            elif "well type" in label and not info["well_type"]:
                info["well_type"] = value[:100]
            elif "closest city" in label and not info["closest_city"]:
                info["closest_city"] = value[:200]
            elif "total oil" in label and not info["oil_produced"]:
                info["oil_produced"] = value[:100]
            elif "total gas" in label and not info["gas_produced"]:
                info["gas_produced"] = value[:100]

    # Strategy 2: text-based fallback
    fallback = _extract_dl_fields(soup)
    for k, v in fallback.items():
        if not info.get(k):
            info[k] = v

    return info


def scrape_single_well(api_number: str | None, well_name: str | None,
                       county: str | None = None) -> dict:
    """Construct URL and scrape a single well page."""
    empty = {"well_status": None, "well_type": None, "closest_city": None,
             "oil_produced": None, "gas_produced": None}
    if not api_number or not well_name:
        return empty

    # Clean well name for URL slug
    clean_name = re.sub(r'\s+(?:Well File No|Directional|Number:?).*', '', well_name, flags=re.IGNORECASE).strip()
    slug = _make_slug(clean_name)
    county_s = _county_slug(county)

    api_clean = api_number.replace(" ", "-")
    if len(api_clean) == 11 and "-" not in api_clean:
        api_clean = f"{api_clean[:2]}-{api_clean[2:5]}-{api_clean[5:]}"

    url = f"{BASE_URL}/north-dakota/{county_s}/wells/{slug}/{api_clean}"
    result = scrape_well_page(url)

    # If first attempt fails, try without county specificity
    found = sum(1 for v in result.values() if v)
    if found == 0:
        url2 = f"{BASE_URL}/north-dakota/mckenzie-county/wells/{slug}/{api_clean}"
        if url2 != url:
            time.sleep(DELAY)
            result = scrape_well_page(url2)

    return result
