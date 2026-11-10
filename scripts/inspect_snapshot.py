"""Re-derive the gate facts from the pinned CFPB bulk file.

Streams the bulk zip and writes reports/snapshot_gate.json: columns,
row count, parsed date-received range, the full "Company response to
consumer" distribution, and whether a narrative column is present.
This is the machine-readable counterpart of the portfolio gate note.
"""

from __future__ import annotations

import json
import zipfile
from collections import Counter

import pandas as pd

from relief_radar.config import load_config, project_path


def main() -> None:
    cfg = load_config()
    zip_path = project_path(cfg["snapshot"]["bulk_zip"])
    rows = 0
    response: Counter = Counter()
    date_min: str | None = None
    date_max: str | None = None
    columns: list[str] = []
    with zipfile.ZipFile(zip_path) as zf:
        member = zf.namelist()[0]
        with zf.open(member) as fh:
            header = pd.read_csv(fh, nrows=0)
            columns = list(header.columns)
        with zf.open(member) as fh:
            reader = pd.read_csv(
                fh,
                usecols=["Date received", "Company response to consumer"],
                dtype=str,
                chunksize=300_000,
            )
            for chunk in reader:
                rows += len(chunk)
                response.update(
                    chunk["Company response to consumer"]
                    .fillna("(missing)")
                    .tolist()
                )
                dates = pd.to_datetime(chunk["Date received"], format="mixed")
                lo = dates.min().date().isoformat()
                hi = dates.max().date().isoformat()
                date_min = lo if date_min is None or lo < date_min else date_min
                date_max = hi if date_max is None or hi > date_max else date_max
    facts = {
        "columns": columns,
        "has_narrative_column": "Consumer complaint narrative" in columns,
        "rows": rows,
        "date_received_min": date_min,
        "date_received_max": date_max,
        "response_distribution": dict(response.most_common()),
        "bulk_zip_sha256": cfg["snapshot"]["bulk_zip_sha256"],
    }
    out = project_path("reports", "snapshot_gate.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(facts, fh, indent=1)
    print(json.dumps(facts, indent=1))


if __name__ == "__main__":
    main()
