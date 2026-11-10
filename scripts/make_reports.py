"""Render reports/results.json and the markdown reports from parts.

Every number in every document comes from a measured part file
(reports/parts/*.json, reports/snapshot_gate.json). This script types
none of them.
"""

from __future__ import annotations

import json

from relief_radar.config import project_path

PARTS = project_path("reports", "parts")
REPORTS = project_path("reports")


def _load(path) -> dict | None:
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def pct(x: float, digits: int = 2) -> str:
    return f"{100 * x:.{digits}f}%"


def main() -> None:
    gate = _load(REPORTS / "snapshot_gate.json") or {}
    build = _load(PARTS / "dataset_build.json") or {}
    models = _load(PARTS / "model_results.json")
    selection = _load(PARTS / "selection_audit.json")
    narrative = _load(PARTS / "narrative_results.json")
    crosswalk = _load(PARTS / "crosswalk_status.json")

    combined = {
        "project": "D1 — Relief Radar: CFPB Complaint Outcomes x SEC Disclosure",
        "snapshot_gate": gate,
        "dataset_build": build,
        "structured_modelling": models,
        "selection_audit": selection,
        "narrative_arm": narrative,
        "crosswalk": crosswalk,
    }
    with open(REPORTS / "results.json", "w", encoding="utf-8") as fh:
        json.dump(combined, fh, indent=1, default=str)

    if models:
        _executive_summary(gate, build, models, selection, narrative, crosswalk)
        _model_card(models, selection)
    if selection:
        _selection_report(selection)
    _alignment_report(crosswalk)
    print("reports rendered")


def _executive_summary(gate, build, models, selection, narrative, crosswalk) -> None:
    pop = models["population"]
    lr = models["models"]["logistic_regression"]["test"]
    gb = models["models"]["hist_gradient_boosting"]["test"]
    lines = [
        "# Executive summary — Relief Radar",
        "",
        "## The question",
        "",
        ("When a consumer complaint arrives at the CFPB, how much can be "
        "known at that moment about whether it will end in monetary "
        "relief — the closest observable proxy for a substantiated "
        "problem? And can the consumer's own written narrative, an "
        "opt-in and now-closed archive, be trusted to speak for "
        "complaints in general?"),
        "",
        "## What was built",
        "",
        (f"- A point-in-time outcome model on the full CFPB snapshot: "
        f"{gate.get('rows', 0):,} complaints, of which "
        f"{pop['primary_rows']:,} are in the primary population "
        f"(unambiguous closures) with a monetary-relief rate of "
        f"{pct(pop['primary_base_rate'])}. Features are only what is "
        "known at receipt (see docs/leakage_audit.md); evaluation is on "
        "a later era than training, never a shuffled split."),
        ("- A narrative-selection audit comparing the narrative-bearing "
        "minority with all complaints in the publication window."),
        ("- A narrative arm testing whether consumer text adds signal "
        "on top of the structured fields."),
        ("- A CFPB-to-EDGAR name crosswalk and complaint-vs-disclosure "
        "alignment (status in reports/alignment_report.md)."),
        "",
        "## Headline results (held-out era, 2025+ test sample)",
        "",
        (f"- Logistic regression baseline: ROC AUC {lr['roc_auc']:.3f}, "
        f"PR AUC {lr['pr_auc']:.3f} against a base rate of "
        f"{pct(lr['base_rate'])}, Brier {lr['brier']:.4f}."),
        (f"- Gradient boosting: ROC AUC {gb['roc_auc']:.3f}, "
        f"PR AUC {gb['pr_auc']:.3f}, Brier {gb['brier']:.4f}."),
        (f"- Era drift is real: the full-population relief rate in the "
        f"training era was {pct(pop['train_era_base_rate'])} and in "
        f"the test era {pct(pop['test_era_base_rate'])}. Absolute "
        "probabilities must be re-anchored per era; ranking is what "
        "carries across."),
    ]
    subgroups = models["models"]["hist_gradient_boosting"].get(
        "subgroups_product", []
    )
    if subgroups:
        top = max(subgroups, key=lambda r: r["n"])
        aucs = [r["roc_auc"] for r in subgroups if r["positives"] >= 5]
        relief_heavy = [
            r for r in subgroups if r["base_rate"] >= 0.10 and r["positives"] >= 5
        ]
        lines += [
            (
                f"- Read the headline AUC with the product mix: the "
                f"largest test subgroup, {top['product']}, is "
                f"{top['n']:,} of the sampled test rows at a relief rate "
                f"of {pct(top['base_rate'])}, and near-separable volume "
                f"like that flatters the pooled ranking. Within-product "
                f"AUCs for gradient boosting span "
                f"{min(aucs):.2f} to {max(aucs):.2f}; on the products "
                f"where relief is common ("
                + ", ".join(
                    f"{r['product']} {r['roc_auc']:.2f}"
                    for r in sorted(
                        relief_heavy, key=lambda r: -r["n"]
                    )
                )
                + "), discrimination is real but far more modest. The "
                "subgroup table is the operationally honest read."
            ),
        ]
    if selection:
        sp = selection["population"]
        lines += [
            "",
            "## The narrative layer, audited",
            "",
            (f"- In the publication window, {pct(sp['narrative_share'])} "
            f"of complaints carry a narrative "
            f"({sp['narrative_rows_in_window']:,} of "
            f"{sp['primary_rows_in_window']:,})."),
            (f"- Monetary-relief rate: {pct(sp['relief_rate_narrative'])} "
            f"for narrative-bearing complaints vs "
            f"{pct(sp['relief_rate_no_narrative'])} without. "
            "reports/selection_audit.md decomposes how much of that gap "
            "is composition (product and era mix) and how much survives "
            "inside year x product cells."),
        ]
    if narrative:
        nm = narrative["models"]
        diff = narrative["combined_minus_structured_auc"]
        lines += [
            "",
            "## Does the text help?",
            "",
            (f"- On identical narrative-bearing rows, structured-only "
            f"logistic regression scores ROC AUC "
            f"{nm['structured_only_logistic']['roc_auc']:.3f}; "
            f"text-only {nm['text_only_logistic']['roc_auc']:.3f}; "
            f"combined {nm['combined_logistic']['roc_auc']:.3f}. The "
            f"combined-minus-structured AUC difference is "
            f"{diff['point_estimate']:+.3f} (bootstrap 95% CI "
            f"{diff['bootstrap_95ci'][0]:+.3f} to "
            f"{diff['bootstrap_95ci'][1]:+.3f})."),
        ]
    lines += [
        "",
        "## Limits that bind every conclusion",
        "",
        ("- Complaints are self-selected; the CFPB states the database "
        "is not a statistical sample. Volume confounds harm with size "
        "and awareness. Nothing here estimates consumer harm."),
        ("- The narrative corpus is closed (publication ceased "
        "2026-08-14) and opt-in; text findings describe that archive, "
        "not future complaints."),
        ("- Monetary relief is a proxy outcome: it records what the "
        "company reported closing with, not an adjudication."),
    ]
    with open(REPORTS / "executive_summary.md", "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")


def _model_card(models, selection) -> None:
    m = models["models"]
    lines = [
        "# Model card — Relief Radar outcome models",
        "",
        "## Intended use",
        "",
        ("Triage and early-warning ranking of incoming consumer "
        "complaints by their likelihood of closing with monetary "
        "relief, for consumer-operations and conduct-risk teams. Not "
        "for use in decisions about individual consumers, and not a "
        "measure of whether any complaint is meritorious."),
        "",
        "## Training and evaluation protocol",
        "",
        (f"- Train: received before {models['protocol']['train'].split('< ')[-1]} "
        f"({models['samples']['train_n']:,} sampled rows, base rate "
        f"{pct(models['samples']['train_base_rate'])})."),
        (f"- Test: received on or after 2025-01-01 "
        f"({models['samples']['test_n']:,} sampled rows, base rate "
        f"{pct(models['samples']['test_base_rate'])}). Calendar 2024 "
        "is a buffer year used only for drift checks."),
        ("- Features: receipt-time fields only; company/product history "
        "as expanding aggregates over strictly earlier months. Full "
        "leakage audit in docs/leakage_audit.md."),
        ("- Imbalance: no resampling or class weighting in the headline "
        "models (a class-weighted sensitivity run is included in "
        "results.json); ranking and calibration metrics are reported "
        "instead of accuracy."),
        "",
        "## Performance on the held-out era",
        "",
        "| Model | ROC AUC | PR AUC | Brier | Max calibration gap |",
        "|---|---|---|---|---|",
    ]
    for name in ["logistic_regression", "hist_gradient_boosting"]:
        t = m[name]["test"]
        lines.append(
            f"| {name} | {t['roc_auc']:.3f} | {t['pr_auc']:.3f} | "
            f"{t['brier']:.4f} | {m[name]['calibration_max_abs_gap']:.3f} |"
        )
    lines += [
        "",
        ("Subgroup tables (product, year, channel) are in "
        "reports/parts/subgroups_*.csv and summarised in results.json. "
        "Performance varies by product; the model must not be read as "
        "uniformly reliable across the catalogue."),
        "",
        "## Limitations",
        "",
        ("- Trained on a self-selected complaint stream; scores rank "
        "complaints, they do not estimate harm or fault."),
        ("- Relief rates drift by era, so calibrated probabilities are "
        "era-specific even when ranking transfers."),
        ("- Legacy response labels that do not distinguish monetary "
        "from non-monetary relief are excluded from training (counted "
        "in the dataset build report)."),
    ]
    with open(REPORTS / "model_card.md", "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")


def _selection_report(selection) -> None:
    sp = selection["population"]
    cell = selection.get("within_cell_relief_diff", {})
    years = selection["tables"]["year"]
    first, last = years[-1], years[0]
    lines = [
        "# Narrative-selection audit",
        "",
        "## Scope",
        "",
        (f"Complaints received {selection['window']['start']} to "
        f"{selection['window']['end']} — the window in which a "
        "published narrative was possible. Outside it, 'no narrative' "
        "is structural, not a consumer choice, and including those "
        "complaints would manufacture a selection effect."),
        "",
        "## How big is the narrative layer?",
        "",
        (f"{sp['narrative_rows_in_window']:,} of "
        f"{sp['primary_rows_in_window']:,} primary-population "
        f"complaints ({pct(sp['narrative_share'])}) carry a narrative. "
        f"The share is not stable over time: it is "
        f"{pct(first['narrative_share'])} in {first['year']} and "
        f"{pct(last['narrative_share'])} in {last['year']} (the archive "
        "thins sharply as publication ends in 2026; see "
        "figures/narrative_share_by_year.png and "
        "reports/parts/selection_by_year.csv). Narrative rates also "
        "differ by product, channel, and company size — full tables in "
        "reports/parts/selection_by_*.csv."),
        "",
        "## Do narrative-bearing complaints end differently?",
        "",
        (f"Raw relief rates: {pct(sp['relief_rate_narrative'])} with a "
        f"narrative vs {pct(sp['relief_rate_no_narrative'])} without. "
        "That raw gap mixes composition (which products and eras "
        "produce narratives) with any within-cell difference."),
    ]
    if cell:
        lines += [
            "",
            (f"Inside {cell['cells']} year x product cells, the "
            f"volume-weighted mean relief-rate difference is "
            f"{pct(cell['volume_weighted_mean_diff'])} and the median "
            f"cell difference is {pct(cell['median_diff'])}; narrative "
            f"complaints have the higher rate in "
            f"{pct(cell['share_cells_narrative_higher'], 0)} of cells."),
        ]
    lines += [
        "",
        "## Conclusion",
        "",
        ("The narrative layer can speak for the complaints that carry "
        "narratives — a self-selected, opt-in minority whose mix of "
        "products, eras, and channels differs measurably from the full "
        "stream. It cannot speak for all complaints: any statistic "
        "computed on narratives alone inherits that selection, and the "
        "layer closes entirely after August 2026, so narrative-based "
        "monitoring has no future data. Findings from the narrative "
        "arm of this project are reported as findings about "
        "narrative-bearing complaints, full stop."),
    ]
    with open(REPORTS / "selection_audit.md", "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")


def _alignment_report(crosswalk) -> None:
    lines = ["# Complaint-vs-disclosure alignment", ""]
    if not crosswalk:
        lines.append("Not run.")
    elif crosswalk.get("status") == "blocked":
        lines += [
            "## Status: blocked in this build environment",
            "",
            crosswalk.get("reason", ""),
            "",
            (f"The CFPB side is complete: {crosswalk.get('cfpb_companies', 0):,} "
            f"distinct normalized company names covering "
            f"{crosswalk.get('cfpb_primary_complaints', 0):,} "
            "primary-population complaints (reports/parts/"
            "cfpb_company_table.csv). The matcher "
            "(src/relief_radar/crosswalk.py) implements exact, "
            "containment, and fuzzy tiers with ambiguity flags and is "
            "unit-tested on fixtures; the EDGAR facts extractor "
            "(src/relief_radar/edgar.py) is likewise fixture-tested. "
            "What is missing is only the SEC reference data, which "
            "must come from the SEC itself."),
            "",
            ("No alignment statistics are reported, because none were "
            "measured. Any number here would be fabricated."),
        ]
    else:
        lines += [
            "## Match audit",
            "",
            (f"- Companies matched: {crosswalk.get('companies_matched', 0):,} "
            f"of {crosswalk.get('cfpb_companies', 0):,} "
            f"({pct(crosswalk.get('company_match_rate', 0))})."),
            (f"- Complaint volume covered: "
            f"{pct(crosswalk.get('complaint_volume_share_matched', 0))} "
            "of the primary population."),
            (f"- Ambiguous matches flagged: "
            f"{crosswalk.get('ambiguous_matches', 0):,}."),
            (f"- Alignment panel rows: {crosswalk.get('alignment_rows', 0):,} "
            f"(status: {crosswalk.get('alignment', 'n/a')})."),
            "",
            ("The panel (reports/parts/alignment_panel.parquet) aligns "
            "annual complaint volume and relief rates with filed "
            "financial facts for the largest confidently matched "
            "filers. It is descriptive context only: complaint volume "
            "tracks company size and customer counts, and no causal "
            "reading is offered."),
        ]
    with open(REPORTS / "alignment_report.md", "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
