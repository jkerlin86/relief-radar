"""Download every raw input from its official source.

Sources (all public, no keys):
1. CFPB Consumer Complaint Database bulk file (structured records).
2. CFPB FOIA Reading Room "Consumer Complaint Database Narratives
   Archive": 21 full-record export zips carrying the narrative text the
   live bulk file no longer ships.
3. SEC EDGAR company tickers (CIK mapping) with the declared
   User-Agent required by SEC fair-access rules.

Writes data/raw/DOWNLOAD_MANIFEST.json with sizes and SHA-256 hashes.
Nothing is downloaded from any third party or mirror.
"""

from __future__ import annotations

import hashlib
import json
import time
import urllib.request
from pathlib import Path

from relief_radar.config import load_config, project_path

BULK_URL = "https://files.consumerfinance.gov/ccdb/complaints.csv.zip"

FOIA_BASE = "https://files.consumerfinance.gov/f/documents/"
FOIA_FILES = [
    "CCDB_Export_1_December_2011_through_April_2018.zip",
    "CCDB_Export_2_May_2018_through_April_2021.zip",
    "CCDB_Export_3_May_2021_through_October_2022.zip",
    "CCDB_Export_4_November_2022_through_August_2023.zip",
    "CCDB_Export_5_September_2023_through_March_2024.zip",
    "CCDB_Export_6_April_2024_through_July_2024.zip",
    "CCDB_Export_7_August_2024_through_October_2024.zip",
    "CCDB_Export_8_November_2024_through_December_2024.zip",
    "CCDB_Export_9_January_2025_through_February_2025.zip",
    "CCDB_Export_10_March_2025_through_April_2025.zip",
    "CCDB_Export_11_May_2025_through_June_2025.zip",
    "CCDB_Export_12_July_2025_through_August_2025.zip",
    "CCDB_Export_13_September_2025_through_October_2025.zip",
    "CCDB_Export_14_November_2025_through_December_2025.zip",
    "CCDB_Export_15_January_2026_through_February_2026.zip",
    "CCDB_Export_16_March_2026.zip",
    "CCDB_Export_17_April_2026.zip",
    "CCDB_Export_18_May_2026.zip",
    "CCDB_Export_19_June_2026.zip",
    "CCDB_Export_20_July_2026.zip",
    "CCDB_Export_21_August_2026.zip",
]

SEC_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _fetch(url: str, dest: Path, user_agent: str) -> dict:
    if dest.exists() and dest.stat().st_size > 0:
        return {"url": url, "path": str(dest), "status": "already_present",
                "bytes": dest.stat().st_size, "sha256": _sha256(dest)}
    req = urllib.request.Request(url, headers={"User-Agent": user_agent})
    try:
        with urllib.request.urlopen(req) as resp, open(dest, "wb") as fh:
            while block := resp.read(1 << 20):
                fh.write(block)
        return {"url": url, "path": str(dest), "status": "downloaded",
                "bytes": dest.stat().st_size, "sha256": _sha256(dest)}
    except Exception as exc:  # noqa: BLE001 — failure is recorded, never hidden
        return {"url": url, "path": str(dest), "status": f"failed: {exc}"}


def main() -> None:
    cfg = load_config()
    ua = cfg["edgar"]["user_agent"]
    manifest: dict = {"files": []}
    bulk = project_path(cfg["snapshot"]["bulk_zip"])
    bulk.parent.mkdir(parents=True, exist_ok=True)
    manifest["files"].append(
        _fetch(BULK_URL, bulk, "Mozilla/5.0 (compatible; ReliefRadar/0.1)")
    )
    foia_dir = project_path(cfg["snapshot"]["foia_dir"])
    foia_dir.mkdir(parents=True, exist_ok=True)
    for name in FOIA_FILES:
        manifest["files"].append(
            _fetch(FOIA_BASE + name, foia_dir / name,
                   "Mozilla/5.0 (compatible; ReliefRadar/0.1)")
        )
        time.sleep(0.2)
    tickers = project_path(cfg["snapshot"]["sec_tickers"])
    entry = _fetch(SEC_TICKERS_URL, tickers, ua)
    if entry.get("status") == "downloaded":
        head = tickers.read_bytes()[:1]
        if head != b"{":
            entry["status"] = "failed: response was not JSON (access page)"
            tickers.unlink(missing_ok=True)
    manifest["files"].append(entry)
    out = project_path("data", "raw", "DOWNLOAD_MANIFEST.json")
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=1)
    print(json.dumps(manifest, indent=1)[:1500])


if __name__ == "__main__":
    main()
