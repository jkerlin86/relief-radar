# Complaint-vs-disclosure alignment

## Status: blocked in this build environment

SEC company_tickers.json could not be downloaded from this environment (SEC returned an access/rate page, HTTP 403, to the declared User-Agent). No third-party mirror was used. The matcher is implemented and unit-tested on fixtures; run scripts/download_data.py from a network where EDGAR fair access succeeds, then re-run this script.

The CFPB side is complete: 7,639 distinct normalized company names covering 18,367,254 primary-population complaints (reports/parts/cfpb_company_table.csv). The matcher (src/relief_radar/crosswalk.py) implements exact, containment, and fuzzy tiers with ambiguity flags and is unit-tested on fixtures; the EDGAR facts extractor (src/relief_radar/edgar.py) is likewise fixture-tested. What is missing is only the SEC reference data, which must come from the SEC itself.

No alignment statistics are reported, because none were measured. Any number here would be fabricated.
