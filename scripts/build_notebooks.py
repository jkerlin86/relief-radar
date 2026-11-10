"""Build the two project notebooks (executed separately with nbclient).

Notebook 01 explores the pinned snapshot and processed population from
the build artefacts. Notebook 02 walks through the modelling protocol
and results from reports/parts/. Both read only small artefacts, so
they execute quickly on this machine and in CI-like environments.
"""

from __future__ import annotations

import nbformat as nbf

from relief_radar.config import project_path


def md(text: str):
    return nbf.v4.new_markdown_cell(text)


def code(text: str):
    return nbf.v4.new_code_cell(text)


def notebook_01() -> nbf.NotebookNode:
    nb = nbf.v4.new_notebook(metadata={"kernelspec": {"name": "python3", "display_name": "Python 3"}})
    nb.cells = [
        md(
            "# Relief Radar 01 — The snapshot and the population\n"
            "\n"
            "Every number here is read from build artefacts "
            "(`reports/snapshot_gate.json`, `reports/parts/dataset_build.json`, "
            "`reports/parts/model_results.json`), so this notebook re-executes "
            "in seconds and can never drift from the pipeline outputs."
        ),
        code(
            "import json\n"
            "import pandas as pd\n"
            "import matplotlib.pyplot as plt\n"
            "\n"
            "gate = json.load(open('../reports/snapshot_gate.json'))\n"
            "build = json.load(open('../reports/parts/dataset_build.json'))\n"
            "print('rows:', f\"{gate['rows']:,}\")\n"
            "print('narrative column in bulk file:', gate['has_narrative_column'])\n"
            "print('date range:', gate['date_received_min'], 'to', gate['date_received_max'])\n"
            "print('bulk zip sha256:', gate['bulk_zip_sha256'][:16], '...')\n"
            "gate['columns']"
        ),
        md("## How complaints close (full file)"),
        code(
            "resp = pd.Series(gate['response_distribution']).sort_values(ascending=True)\n"
            "resp.plot(kind='barh', figsize=(8, 4))\n"
            "plt.title('Company response to consumer: all 19,045,222 complaints')\n"
            "plt.tight_layout(); plt.show()\n"
            "resp"
        ),
        md(
            "## The primary population\n"
            "\n"
            "The modelling population keeps only unambiguous closures: "
            "\"Closed with monetary relief\" (positive) against \"Closed with "
            "explanation\", \"Closed with non-monetary relief\" and \"Closed "
            "without relief\". In-progress, untimely, legacy ambiguous labels "
            "and missing responses are excluded and counted, never dropped "
            "silently (see docs/leakage_audit.md)."
        ),
        code(
            "comp = build['complaints']\n"
            "print('primary population:', f\"{comp['primary_population_rows']:,}\")\n"
            "print('positives:', f\"{comp['positives_monetary_relief']:,}\")\n"
            "print('base rate:', round(100 * comp['positives_monetary_relief'] / comp['primary_population_rows'], 3), '%')\n"
            "print('unique raw company names:', f\"{comp['unique_companies_raw']:,}\")\n"
            "years = pd.Series(comp['rows_by_year']).rename('complaints')\n"
            "years.index = years.index.astype(int)\n"
            "years.plot(kind='bar', figsize=(9, 3.5))\n"
            "plt.title('Complaints by year received (all rows)')\n"
            "plt.tight_layout(); plt.show()\n"
            "years"
        ),
        md(
            "## Era drift in the outcome itself\n"
            "\n"
            "The monetary-relief base rate collapses across eras as complaint "
            "volume shifts toward credit reporting. This is why the evaluation "
            "is a temporal split with era base rates reported beside every "
            "sample metric, and why a single headline AUC must be read with "
            "the within-product subgroups (notebook 02)."
        ),
        code(
            "models = json.load(open('../reports/parts/model_results.json'))\n"
            "era = pd.DataFrame(models['era_base_rates_full_population']).T\n"
            "era.index = era.index.astype(int)\n"
            "era['relief_rate_pct'] = 100 * era['relief_rate']\n"
            "era[['n', 'relief_rate_pct']].plot(y='relief_rate_pct', figsize=(9, 3.5))\n"
            "plt.title('Monetary-relief rate by year (primary population)')\n"
            "plt.ylabel('% closed with monetary relief')\n"
            "plt.tight_layout(); plt.show()\n"
            "era"
        ),
        md(
            "## The narrative layer\n"
            "\n"
            "The live bulk file no longer carries narratives (publication "
            "ceased 2026-08-14). The narrative corpus used here is the CFPB "
            "FOIA Reading Room archive: narrative-bearing means a non-empty "
            "narrative in that archive for the Complaint ID."
        ),
        code(
            "narr = build['narratives']\n"
            "print('narrative rows extracted:', f\"{narr['narrative_rows']:,}\")\n"
            "print('duplicate IDs dropped:', narr['duplicate_ids_dropped'])\n"
            "audit = json.load(open('../reports/parts/selection_audit.json'))\n"
            "pop = audit['population']\n"
            "for k, v in pop.items():\n"
            "    print(k, ':', v)"
        ),
    ]
    return nb


def notebook_02() -> nbf.NotebookNode:
    nb = nbf.v4.new_notebook(metadata={"kernelspec": {"name": "python3", "display_name": "Python 3"}})
    nb.cells = [
        md(
            "# Relief Radar 02 — Modelling walkthrough\n"
            "\n"
            "Protocol: train on complaints received before 2024, buffer year "
            "2024 for drift context, test on 2025 onward. Features are known "
            "at receipt only; company identity enters through point-in-time "
            "history aggregates. Full audit: docs/leakage_audit.md."
        ),
        code(
            "import json\n"
            "import pandas as pd\n"
            "import matplotlib.pyplot as plt\n"
            "\n"
            "res = json.load(open('../reports/parts/model_results.json'))\n"
            "rows = []\n"
            "for name, block in res['models'].items():\n"
            "    t = block['test']\n"
            "    rows.append({'model': name, 'roc_auc': t['roc_auc'], 'pr_auc': t['pr_auc'],\n"
            "                 'brier': t['brier'], 'log_loss': t['log_loss'],\n"
            "                 'calibration_max_gap': block['calibration_max_abs_gap']})\n"
            "headline = pd.DataFrame(rows).set_index('model')\n"
            "print('test era:', res['protocol']['test'], '| test n:', f\"{res['samples']['test_n']:,}\",\n"
            "      '| test base rate:', round(100 * res['samples']['test_base_rate'], 3), '%')\n"
            "headline"
        ),
        md(
            "## Calibration (test era)\n"
            "\n"
            "At a 0.5% base rate, calibration is the difference between a "
            "useful ranking and a useful probability. The class-weighted "
            "sensitivity model ranks about as well but its probabilities are "
            "uncalibrated by construction; it is reported as a sensitivity, "
            "never as a headline."
        ),
        code(
            "fig, ax = plt.subplots(figsize=(6, 5))\n"
            "for name in ['logistic_regression', 'hist_gradient_boosting']:\n"
            "    cal = pd.read_csv(f'../reports/parts/calibration_{name}.csv')\n"
            "    ax.plot(cal['mean_predicted'], cal['observed_rate'], marker='o', label=name)\n"
            "ax.plot([0, 0.6], [0, 0.6], 'k--', lw=1, label='perfect')\n"
            "ax.set_xlabel('mean predicted'); ax.set_ylabel('observed rate')\n"
            "ax.legend(); ax.set_title('Calibration on the 2025+ test sample (quantile bins)')\n"
            "plt.tight_layout(); plt.show()"
        ),
        md(
            "## Subgroup honesty check\n"
            "\n"
            "The headline AUC is flattered by product mix: the test era is "
            "dominated by credit-reporting complaints, which almost never end "
            "in monetary relief. Within-product discrimination is the honest "
            "read."
        ),
        code(
            "sub = pd.read_csv('../reports/parts/subgroups_hist_gradient_boosting_product.csv')\n"
            "sub[['product', 'n', 'positives', 'base_rate', 'roc_auc', 'pr_auc']].sort_values('n', ascending=False)"
        ),
        md("## Does the consumer's own text help? (narrative arm)"),
        code(
            "arm = json.load(open('../reports/parts/narrative_results.json'))\n"
            "comp = pd.DataFrame(arm['models']).T\n"
            "print('sample:', arm['samples'])\n"
            "print('bootstrap, combined minus structured AUC:', arm['combined_minus_structured_auc'])\n"
            "comp[['roc_auc', 'pr_auc', 'brier']]"
        ),
        md(
            "## Selection audit and crosswalk status\n"
            "\n"
            "Narratives are opt-in: the audit quantifies how far the "
            "narrative-bearing minority can speak for all complaints. The "
            "EDGAR crosswalk status is reported exactly as measured."
        ),
        code(
            "audit = json.load(open('../reports/parts/selection_audit.json'))\n"
            "print('audit population:', audit['population'])\n"
            "print('within-cell check:', audit.get('within_cell_relief_diff'))\n"
            "cross = json.load(open('../reports/parts/crosswalk_status.json'))\n"
            "print('crosswalk:', cross.get('status'), '| CFPB companies:', cross.get('cfpb_companies'))"
        ),
    ]
    return nb


def main() -> None:
    out = project_path("notebooks")
    out.mkdir(parents=True, exist_ok=True)
    nbf.write(notebook_01(), out / "01_snapshot_and_population.ipynb")
    nbf.write(notebook_02(), out / "02_modelling_walkthrough.ipynb")
    print("notebooks written")


if __name__ == "__main__":
    main()
