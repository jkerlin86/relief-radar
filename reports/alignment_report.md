# Complaint-vs-disclosure alignment

## Match audit

- Companies matched: 348 of 7,639 (4.56%).
- Complaint volume covered: 64.89% of the primary population.
- Ambiguous matches flagged: 102.
- Alignment panel rows: 407 (status: completed).

The panel (reports/parts/alignment_panel.parquet) aligns annual complaint volume and relief rates with filed financial facts for the largest confidently matched filers. It is descriptive context only: complaint volume tracks company size and customer counts, and no causal reading is offered.

## Completion note (2026-11-11)

This step was completed on 2026-11-11, after the access route was established. At publication (2026-11-10) SEC EDGAR returned HTTP 403 to this build environment and this report recorded the alignment as blocked with no statistics, because none had been measured. The run above used only the SEC itself (company_tickers.json and data.sec.gov company facts), with the declared User-Agent configured in configs/default.json and the client throttled to at most 5 requests per second; no third-party mirror was used. All statistics in this report and in reports/parts/crosswalk_status.json were measured in that run. Two configuration-level fixes were required and are documented in the commit: the configured User-Agent was set to the declared name-and-email form, and the tickers parser was taught the "title" field name the SEC actually publishes (the original fixtures assumed "name"). The largest unmatched name is EXPERIAN INFORMATION SOLUTIONS (4,308,344 complaints): the credit bureaus' operating subsidiaries are not themselves SEC filers in the tickers file, which is the main reason company-count match rate is low while complaint-volume coverage is high.
