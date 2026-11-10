"""Name -> CIK crosswalk and complaint-vs-disclosure alignment.

Builds the CFPB company table from the primary population, matches it
against the SEC tickers file when that file is available, and (when
matching succeeds) aligns annual complaint volume and relief rates with
filed financial facts for the largest confidently matched filers.

If EDGAR data cannot be obtained, the script records exactly that in
reports/parts/crosswalk_status.json and stops — it never substitutes a
third-party mirror for the SEC source.
"""

from __future__ import annotations

import json

import pandas as pd

from relief_radar.config import load_config, project_path
from relief_radar.crosswalk import match_companies, prepare_tickers
from relief_radar.edgar import EdgarAccessError, EdgarClient, annual_facts

ALIGNMENT_CONCEPTS = ["Revenues", "NetIncomeLoss", "Assets"]
TOP_FILERS = 15


def company_table() -> pd.DataFrame:
    from relief_radar.data import load_primary_arrow

    grouped = (
        load_primary_arrow()
        .group_by(["company_norm"])
        .aggregate([("complaint_id", "count"), ("target", "sum")])
        .to_pandas()
        .rename(columns={"complaint_id_count": "complaints", "target_sum": "relief_n"})
    )
    grouped["relief_rate"] = grouped["relief_n"] / grouped["complaints"]
    return grouped.sort_values("complaints", ascending=False).reset_index(drop=True)


def main() -> None:
    cfg = load_config()
    parts = project_path("reports", "parts")
    status: dict = {}
    companies = company_table()
    companies.to_csv(parts / "cfpb_company_table.csv", index=False)
    status["cfpb_companies"] = len(companies)
    status["cfpb_primary_complaints"] = int(companies["complaints"].sum())

    tickers_path = project_path(cfg["snapshot"]["sec_tickers"])
    if not tickers_path.exists():
        status["status"] = "blocked"
        status["reason"] = (
            "SEC company_tickers.json could not be downloaded from this "
            "environment (SEC returned an access/rate page, HTTP 403, to "
            "the declared User-Agent). No third-party mirror was used. "
            "The matcher is implemented and unit-tested on fixtures; "
            "run scripts/download_data.py from a network where EDGAR "
            "fair access succeeds, then re-run this script."
        )
        with open(parts / "crosswalk_status.json", "w", encoding="utf-8") as fh:
            json.dump(status, fh, indent=1)
        print(json.dumps(status, indent=1))
        return

    raw = pd.read_json(tickers_path).T.reset_index(drop=True)
    tickers = prepare_tickers(raw)
    matches = match_companies(companies["company_norm"].tolist(), tickers)
    merged = companies.merge(matches, on="company_norm", how="left")
    merged.to_parquet(parts / "crosswalk.parquet", index=False)
    matched = merged[merged["tier"] != "unmatched"]
    status["status"] = "matched"
    status["companies_matched"] = len(matched)
    status["company_match_rate"] = float(len(matched) / len(merged))
    status["complaint_volume_matched"] = int(matched["complaints"].sum())
    status["complaint_volume_share_matched"] = float(
        matched["complaints"].sum() / merged["complaints"].sum()
    )
    status["by_tier"] = (
        matched.groupby("tier")["complaints"].agg(["size", "sum"]).to_dict("index")
    )
    status["ambiguous_matches"] = int(matched["ambiguous"].sum())
    top_unmatched = merged[merged["tier"] == "unmatched"].head(25)
    status["top_unmatched"] = top_unmatched[
        ["company_norm", "complaints"]
    ].to_dict(orient="records")

    # Alignment for the largest confidently matched filers.
    client = EdgarClient(
        cfg["edgar"]["user_agent"], cfg["edgar"]["max_requests_per_second"]
    )
    confident = matched[~matched["ambiguous"]].head(TOP_FILERS)
    alignment_rows = []
    try:
        from relief_radar.data import load_primary_arrow

        annual = (
            load_primary_arrow()
            .group_by(["company_norm", "year"])
            .aggregate([("complaint_id", "count"), ("target", "sum")])
            .to_pandas()
            .rename(
                columns={"complaint_id_count": "complaints", "target_sum": "relief_n"}
            )
        )
        annual["relief_rate"] = annual["relief_n"] / annual["complaints"]
        for _, crow in confident.iterrows():
            facts = client.company_facts(int(crow["cik"]))
            comp = annual[annual["company_norm"] == crow["company_norm"]]
            for concept in ALIGNMENT_CONCEPTS:
                for fact in annual_facts(facts, concept):
                    match = comp[comp["year"] == fact["year"]]
                    if len(match):
                        alignment_rows.append(
                            {
                                "company_norm": crow["company_norm"],
                                "sec_name": crow["sec_name"],
                                "cik": int(crow["cik"]),
                                "year": fact["year"],
                                "concept": concept,
                                "filed_value": fact["value"],
                                "complaints": int(match["complaints"].iloc[0]),
                                "relief_rate": float(match["relief_rate"].iloc[0]),
                            }
                        )
        status["alignment"] = "completed"
    except EdgarAccessError as exc:
        status["alignment"] = f"blocked: {exc}"
    if alignment_rows:
        pd.DataFrame(alignment_rows).to_parquet(
            parts / "alignment_panel.parquet", index=False
        )
    status["alignment_rows"] = len(alignment_rows)
    with open(parts / "crosswalk_status.json", "w", encoding="utf-8") as fh:
        json.dump(status, fh, indent=1, default=str)
    print(json.dumps(status, indent=1, default=str)[:2000])


if __name__ == "__main__":
    main()
