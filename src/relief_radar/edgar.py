"""Minimal, polite SEC EDGAR client.

Honours SEC fair access: declared User-Agent (from config), at most
the configured requests per second, no key. Only the endpoints the
alignment step needs: company tickers and XBRL company facts.
"""

from __future__ import annotations

import json
import time
import urllib.request

TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
FACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik:010d}.json"


class EdgarAccessError(RuntimeError):
    """Raised when EDGAR does not return usable JSON."""


class EdgarClient:
    def __init__(self, user_agent: str, max_requests_per_second: float = 5.0):
        self.user_agent = user_agent
        self.min_interval = 1.0 / max_requests_per_second
        self._last_request = 0.0

    def _get_json(self, url: str) -> dict:
        wait = self.min_interval - (time.monotonic() - self._last_request)
        if wait > 0:
            time.sleep(wait)
        req = urllib.request.Request(
            url, headers={"User-Agent": self.user_agent, "Accept": "application/json"}
        )
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                payload = resp.read()
        except Exception as exc:
            raise EdgarAccessError(f"GET {url} failed: {exc}") from exc
        finally:
            self._last_request = time.monotonic()
        try:
            return json.loads(payload)
        except json.JSONDecodeError as exc:
            raise EdgarAccessError(
                f"GET {url} did not return JSON (access page or block)"
            ) from exc

    def company_tickers(self) -> list[dict]:
        data = self._get_json(TICKERS_URL)
        return list(data.values())

    def company_facts(self, cik: int) -> dict:
        return self._get_json(FACTS_URL.format(cik=cik))


def annual_facts(facts: dict, concept: str, taxonomy: str = "us-gaap") -> list[dict]:
    """Extract annual (10-K, FY) values for one XBRL concept.

    Returns one row per fiscal year: the value from the latest-filed
    10-K observation for that year. Pure function of the facts payload,
    so it is unit-testable on fixtures.
    """
    units = facts.get("facts", {}).get(taxonomy, {}).get(concept, {}).get("units", {})
    rows: list[dict] = []
    for unit, entries in units.items():
        by_year: dict[int, dict] = {}
        for e in entries:
            if e.get("form") != "10-K" or e.get("fp") != "FY":
                continue
            year = int(e["fy"])
            current = by_year.get(year)
            if current is None or e["filed"] > current["filed"]:
                by_year[year] = {"year": year, "value": float(e["val"]),
                                 "filed": e["filed"], "unit": unit}
        rows.extend(by_year.values())
    return sorted(rows, key=lambda r: r["year"])
