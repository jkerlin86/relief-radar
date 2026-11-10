"""Narrative arm: does the consumer's own text add predictive signal?

Population: narrative-bearing complaints in the primary population
(narrative text from the FOIA archive, joined on Complaint ID).
Sampling: stratified by year x outcome, proportional allocation, fixed
seed, caps from config — the sample design and the sample-vs-population
base rates are reported in the output, never hidden.

Comparisons on identical rows (paired):
- structured-only (same features as the main model),
- text-only (logistic regression on MiniLM embeddings),
- combined (structured + PCA-reduced embeddings).
A percentile bootstrap over test rows gives a CI for the combined minus
structured AUC difference.
"""

from __future__ import annotations

import json
import os

# Environment workaround (documented, not a data decision): this
# sandbox's NO_PROXY contains bracketed IPv6 entries that httpx
# (huggingface_hub's client) cannot parse, which otherwise crashes the
# encoder's model download. Narrowing NO_PROXY in-process restores it.
os.environ["no_proxy"] = "localhost,127.0.0.1"
os.environ["NO_PROXY"] = "localhost,127.0.0.1"

from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from relief_radar.config import load_config, project_path
from relief_radar.evaluate import binary_metrics
from relief_radar.features import (
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    add_features,
    apply_category_caps,
    build_history_tables,
    fit_category_caps,
)
from relief_radar.models import make_histgb_pipeline, make_logistic_pipeline

COLUMNS = [
    "complaint_id",
    "date_received",
    "year",
    "month",
    "month_of_year",
    "day_of_week",
    "product",
    "sub_product",
    "issue",
    "sub_issue",
    "company_norm",
    "state",
    "zip_missing",
    "tag_older_american",
    "tag_servicemember",
    "submitted_via",
    "target",
    "in_primary_population",
]


def load_narrative_texts(wanted_ids: set[int]) -> pd.DataFrame:
    """Stream narratives.parquet, keeping only wanted Complaint IDs."""
    path = project_path("data", "processed", "narratives.parquet")
    pf = pq.ParquetFile(path)
    frames = []
    for batch in pf.iter_batches(
        columns=["complaint_id", "narrative"], batch_size=50_000
    ):
        part = batch.to_pandas()
        part = part[part["complaint_id"].isin(wanted_ids)]
        if len(part):
            frames.append(part)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(
        columns=["complaint_id", "narrative"]
    )


def stratified_sample(
    df: pd.DataFrame, cap: int, rng: np.random.Generator
) -> pd.DataFrame:
    """Proportional allocation across year x outcome strata."""
    if len(df) <= cap:
        return df.copy()
    strata = df.groupby(["year", "target"], observed=True)
    parts = []
    for _key, group in strata:
        take = max(1, round(cap * len(group) / len(df)))
        take = min(take, len(group))
        idx = rng.choice(group.index.to_numpy(), size=take, replace=False)
        parts.append(df.loc[idx])
    return pd.concat(parts)


PASS1_DIR = ("data", "processed", "narrative_pass1")


def pass1() -> None:
    """Population pass in an isolated process (memory discipline).

    The Arrow population pass peaks near 3 GB on this machine and the
    allocator does not return the pages; running it as a subprocess
    that exits guarantees the modelling phase starts from a clean,
    small footprint. Writes history tables + the narrative-bearing
    light frame + population facts under data/processed/narrative_pass1/.
    """
    import pyarrow as pa
    import pyarrow.compute as pc

    from relief_radar.data import load_primary_arrow, monthly_stats

    table = load_primary_arrow()
    narrative_ids = pd.read_parquet(
        project_path("data", "processed", "narratives.parquet"),
        columns=["complaint_id"],
    )["complaint_id"]
    # History features must describe ALL complaints (a company's history
    # at receipt includes complaints without narratives), so the history
    # tables are built from the full primary population, as in the main
    # modelling run — not from the narrative subset.
    history = build_history_tables(
        monthly_stats(table, "company_norm"),
        monthly_stats(table, "product"),
        float(pc.mean(table["target"]).as_py()),
    )
    narr = (
        table.filter(
            pc.is_in(
                table["complaint_id"], value_set=pa.array(narrative_ids.to_numpy())
            )
        )
        .select(["complaint_id", "date_received", "year", "target"])
        .to_pandas()
    )
    del table
    out_dir = project_path(*PASS1_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    history["company"].to_parquet(out_dir / "history_company.parquet", index=False)
    history["product"].to_parquet(out_dir / "history_product.parquet", index=False)
    narr.to_parquet(out_dir / "narrative_light.parquet", index=False)
    facts = {
        "narrative_bearing_primary_rows": len(narr),
        "base_rate": float(narr["target"].mean()),
    }
    with open(out_dir / "population.json", "w", encoding="utf-8") as fh:
        json.dump(facts, fh, indent=1)
    print(json.dumps(facts, indent=1))


def load_pass1() -> tuple[dict, pd.DataFrame, dict]:
    """Run pass 1 in a subprocess if its artifacts are missing."""
    import subprocess
    import sys

    out_dir = project_path(*PASS1_DIR)
    marker = out_dir / "population.json"
    if not marker.exists():
        subprocess.run(
            [sys.executable, str(Path(__file__)), "--pass1"],
            check=True,
            cwd=project_path(),
        )
    history = {
        "company": pd.read_parquet(out_dir / "history_company.parquet"),
        "product": pd.read_parquet(out_dir / "history_product.parquet"),
    }
    narr = pd.read_parquet(
        out_dir / "narrative_light.parquet",
        columns=["complaint_id", "year", "target"],
    )
    with open(marker, encoding="utf-8") as fh:
        facts = json.load(fh)
    return history, narr, facts


ENCODE_DIR = ("data", "processed", "narrative_encode_chunks")
ENCODE_CHUNK = 2000
# The encoder truncates at 256 tokens internally; 2,000 characters
# always contain those first 256 tokens for English text, so this
# pre-truncation bounds tokenizer cost without changing what the
# model can see. Recorded in the sidecar and the results.
MAX_TEXT_CHARS = 2000


def _sample_ids(cfg: dict, arm: dict):
    """Deterministic stratified sample shared by encode and main."""
    _history, narr, _facts = load_pass1()
    rng = np.random.default_rng(cfg["seed"])
    # Era boundaries fall on year lines (train_end 2024-01-01,
    # test_start 2025-01-01), so year masks are exact.
    train_pop = narr[narr["year"] <= 2023]
    test_pop = narr[narr["year"] >= 2025]
    train_light = stratified_sample(train_pop, arm["train_sample_n"], rng)
    test_light = stratified_sample(test_pop, arm["test_sample_n"], rng)
    stats = {
        "train_n": len(train_light),
        "train_base_rate": float(train_light["target"].mean()),
        "test_n": len(test_light),
        "test_base_rate": float(test_light["target"].mean()),
        "train_population_n": len(train_pop),
        "test_population_n": len(test_pop),
    }
    return (
        train_light["complaint_id"].tolist(),
        test_light["complaint_id"].tolist(),
        stats,
    )


def encode_phase() -> None:
    """Embed the sampled narratives, checkpointed per chunk.

    Runs as a subprocess: a first uncheckpointed attempt encoded for
    over an hour on this throttled VM and died with the session before
    writing anything. Chunks of 2,000 texts are saved as they finish,
    so a rerun resumes instead of restarting.
    """
    cfg = load_config()
    arm = cfg["narrative_arm"]
    train_ids, test_ids, stats = _sample_ids(cfg, arm)
    import gc

    gc.collect()
    texts = load_narrative_texts(set(train_ids) | set(test_ids))
    text_map = dict(
        zip(texts["complaint_id"].tolist(), texts["narrative"].tolist(), strict=True)
    )
    del texts
    train_ids = [i for i in train_ids if i in text_map]
    test_ids = [i for i in test_ids if i in text_map]
    stats["train_n_with_text"] = len(train_ids)
    stats["test_n_with_text"] = len(test_ids)

    # Encoder runtime: fastembed (ONNX Runtime) executing the published
    # all-MiniLM-L6-v2 weights. The PyTorch/sentence-transformers stack
    # could not be installed reliably in this environment (wheel
    # downloads failed through the egress proxy repeatedly); the model
    # weights are the same. Recorded in the build log and QA report.
    from fastembed import TextEmbedding

    encoder = TextEmbedding(model_name=arm["encoder"])
    out_dir = project_path(*ENCODE_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    def encode_list(ids: list[int], tag: str) -> np.ndarray:
        # Length-sorted encoding: ONNX batches pad to the longest text
        # in the batch, so grouping similar lengths removes most
        # padding waste (measured: unsorted chunks varied 5x in cost).
        # Embeddings are per-text independent, so sorting changes
        # nothing semantically; the permutation is inverted on return.
        texts = [str(text_map[i])[:MAX_TEXT_CHARS] for i in ids]
        order = sorted(range(len(ids)), key=lambda k: len(texts[k]))
        sorted_texts = [texts[k] for k in order]
        arrays = []
        for start in range(0, len(ids), ENCODE_CHUNK):
            chunk_file = out_dir / f"{tag}_sorted_{start}.npz"
            if chunk_file.exists():
                arrays.append(np.load(chunk_file)["emb"])
                print(f"{tag} chunk {start}: cached", flush=True)
                continue
            emb = np.array(
                list(
                    encoder.embed(
                        sorted_texts[start : start + ENCODE_CHUNK],
                        batch_size=arm["encode_batch_size"],
                    )
                ),
                dtype=np.float32,
            )
            np.savez_compressed(chunk_file, emb=emb)
            arrays.append(emb)
            print(f"{tag} chunk {start}: encoded", flush=True)
        emb_sorted = np.concatenate(arrays)
        emb = np.empty_like(emb_sorted)
        for pos, orig in enumerate(order):
            emb[orig] = emb_sorted[pos]
        return emb

    emb_train = encode_list(train_ids, "train")
    emb_test = encode_list(test_ids, "test")
    np.savez_compressed(
        project_path("data", "processed", "narrative_embeddings.npz"),
        train=emb_train,
        test=emb_test,
        train_ids=np.array(train_ids),
        test_ids=np.array(test_ids),
    )
    sidecar = {
        "samples": stats,
        "train_ids": train_ids,
        "test_ids": test_ids,
        "max_text_chars": MAX_TEXT_CHARS,
        "encoder": arm["encoder"],
    }
    with open(
        project_path("data", "processed", "narrative_encode_sidecar.json"),
        "w",
        encoding="utf-8",
    ) as fh:
        json.dump(sidecar, fh)
    print("encode phase complete", flush=True)


def load_embeddings():
    """Run the encode phase in a subprocess if embeddings are missing."""
    import subprocess
    import sys

    npz_path = project_path("data", "processed", "narrative_embeddings.npz")
    if not npz_path.exists():
        subprocess.run(
            [sys.executable, str(Path(__file__)), "--encode"],
            check=True,
            cwd=project_path(),
        )
    data = np.load(npz_path)
    with open(
        project_path("data", "processed", "narrative_encode_sidecar.json"),
        encoding="utf-8",
    ) as fh:
        sidecar = json.load(fh)
    return data, sidecar


def main() -> None:
    cfg = load_config()
    arm = cfg["narrative_arm"]

    from relief_radar.data import fetch_rows_by_ids

    history, _narr, pop_facts = load_pass1()
    emb_data, sidecar = load_embeddings()
    train_ids = sidecar["train_ids"]
    test_ids = sidecar["test_ids"]
    results: dict = {
        "population": pop_facts,
        "sample_design": {
            "strata": "year x target, proportional allocation",
            "train_cap": arm["train_sample_n"],
            "test_cap": arm["test_sample_n"],
            "seed": cfg["seed"],
            "encoder": arm["encoder"],
        },
        "samples": sidecar["samples"],
        "text_truncation_chars": sidecar["max_text_chars"],
        "encoder_runtime": "onnxruntime via fastembed (see encode_phase note)",
    }
    full = fetch_rows_by_ids(set(train_ids) | set(test_ids), COLUMNS)
    train = full[full["complaint_id"].isin(set(train_ids))].drop(
        columns=["in_primary_population"]
    )
    test = full[full["complaint_id"].isin(set(test_ids))].drop(
        columns=["in_primary_population"]
    )
    del full

    train_f = add_features(
        train, history, prior_strength=cfg["models"]["company_prior_strength"]
    )
    test_f = add_features(
        test, history, prior_strength=cfg["models"]["company_prior_strength"]
    )
    caps = fit_category_caps(
        train_f, ["sub_product", "issue", "sub_issue"],
        cfg["models"]["top_k_categories"],
    )
    train_f = apply_category_caps(train_f, caps)
    test_f = apply_category_caps(test_f, caps)
    feature_cols = CATEGORICAL_FEATURES + NUMERIC_FEATURES
    y_train = train_f["target"].astype(int).to_numpy()
    y_test = test_f["target"].astype(int).to_numpy()

    # Align stored embeddings (ordered by the encode phase's ID lists)
    # to the fetched feature frames.
    pos_train = {cid: k for k, cid in enumerate(train_ids)}
    pos_test = {cid: k for k, cid in enumerate(test_ids)}
    emb_train = emb_data["train"][
        [pos_train[c] for c in train_f["complaint_id"].tolist()]
    ]
    emb_test = emb_data["test"][[pos_test[c] for c in test_f["complaint_id"].tolist()]]
    assert emb_train.shape[0] == len(train_f)
    assert emb_test.shape[0] == len(test_f)

    models_out: dict = {}
    structured = make_logistic_pipeline(
        CATEGORICAL_FEATURES, NUMERIC_FEATURES, cfg["models"]["logistic"]
    )
    structured.fit(train_f[feature_cols], y_train)
    p_struct = structured.predict_proba(test_f[feature_cols])[:, 1]
    models_out["structured_only_logistic"] = binary_metrics(y_test, p_struct)

    text_lr = Pipeline(
        [("scale", StandardScaler()),
         ("clf", LogisticRegression(C=1.0, max_iter=1000))]
    )
    text_lr.fit(emb_train, y_train)
    p_text = text_lr.predict_proba(emb_test)[:, 1]
    models_out["text_only_logistic"] = binary_metrics(y_test, p_text)

    pca = PCA(n_components=arm["embedding_pca_components"], random_state=cfg["seed"])
    emb_train_pca = pca.fit_transform(emb_train)
    emb_test_pca = pca.transform(emb_test)
    results["embedding_pca_explained_variance"] = float(
        pca.explained_variance_ratio_.sum()
    )
    pc_cols = [f"emb_{i}" for i in range(arm["embedding_pca_components"])]
    train_c = train_f[feature_cols].copy()
    test_c = test_f[feature_cols].copy()
    for j, col in enumerate(pc_cols):
        train_c[col] = emb_train_pca[:, j]
        test_c[col] = emb_test_pca[:, j]
    combined = make_logistic_pipeline(
        CATEGORICAL_FEATURES, NUMERIC_FEATURES + pc_cols, cfg["models"]["logistic"]
    )
    combined.fit(train_c, y_train)
    p_comb = combined.predict_proba(test_c)[:, 1]
    models_out["combined_logistic"] = binary_metrics(y_test, p_comb)

    gbm = make_histgb_pipeline(
        CATEGORICAL_FEATURES, NUMERIC_FEATURES, cfg["models"]["hist_gb"]
    )
    gbm.fit(train_f[feature_cols], y_train)
    p_gbm = gbm.predict_proba(test_f[feature_cols])[:, 1]
    models_out["structured_only_histgb"] = binary_metrics(y_test, p_gbm)

    # Paired bootstrap: combined vs structured-only AUC difference.
    boot_rng = np.random.default_rng(cfg["seed"] + 1)
    n = len(y_test)
    diffs = []
    for _ in range(300):
        idx = boot_rng.integers(0, n, size=n)
        if y_test[idx].sum() in (0, len(idx)):
            continue
        diffs.append(
            roc_auc_score(y_test[idx], p_comb[idx])
            - roc_auc_score(y_test[idx], p_struct[idx])
        )
    results["combined_minus_structured_auc"] = {
        "point_estimate": float(
            roc_auc_score(y_test, p_comb) - roc_auc_score(y_test, p_struct)
        ),
        "bootstrap_95ci": [
            float(np.percentile(diffs, 2.5)),
            float(np.percentile(diffs, 97.5)),
        ],
        "bootstrap_reps": len(diffs),
    }
    results["models"] = models_out
    with open(
        project_path("reports", "parts", "narrative_results.json"),
        "w",
        encoding="utf-8",
    ) as fh:
        json.dump(results, fh, indent=1, default=str)
    print(json.dumps(results, indent=1, default=str)[:2500])


if __name__ == "__main__":
    import sys

    if "--pass1" in sys.argv:
        pass1()
    elif "--encode" in sys.argv:
        encode_phase()
    else:
        main()
