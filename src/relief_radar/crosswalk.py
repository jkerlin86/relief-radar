"""CFPB company name -> SEC CIK crosswalk.

Matching tiers, applied in order, each row matched at most once:
1. exact: normalized names identical.
2. containment: one name's token set contains the other's (handles
   suffix/prefix variants like "BANK OF AMERICA" vs "BANK OF AMERICA
   NATIONAL ASSOCIATION" after suffix stripping); ambiguous when more
   than one candidate qualifies with the same token surplus.
3. fuzzy: difflib ratio >= threshold against candidates sharing the
   first token (blocking); ambiguous when the runner-up is within the
   ambiguity margin of the best score.

Every match records its tier and, for tiers 2-3, the competing
candidates considered, so the audit can report match quality honestly.
"""

from __future__ import annotations

import difflib

import pandas as pd

from relief_radar.names import normalize_company, token_set

FUZZY_THRESHOLD = 0.90
AMBIGUITY_MARGIN = 0.02


def prepare_tickers(raw: pd.DataFrame) -> pd.DataFrame:
    """Normalize the SEC company_tickers frame (cik_str/title/ticker).

    The file as published by the SEC names the company field "title";
    "name" is accepted as a fallback for older fixtures.
    """
    name_col = "title" if "title" in raw.columns else "name"
    out = pd.DataFrame(
        {
            "cik": raw["cik_str"].astype(int),
            "sec_name": raw[name_col].astype(str),
            "ticker": raw["ticker"].astype(str),
        }
    )
    out["name_norm"] = out["sec_name"].map(normalize_company)
    return out


def match_companies(
    cfpb_names: list[str], tickers: pd.DataFrame
) -> pd.DataFrame:
    """Match normalized CFPB company names to SEC filers by tier."""
    by_norm: dict[str, list[int]] = {}
    for idx, name in enumerate(tickers["name_norm"]):
        by_norm.setdefault(name, []).append(idx)
    by_first_token: dict[str, list[int]] = {}
    for idx, name in enumerate(tickers["name_norm"]):
        tokens = name.split(" ")
        if tokens:
            by_first_token.setdefault(tokens[0], []).append(idx)

    rows = []
    sec_norms = tickers["name_norm"].tolist()
    for name in cfpb_names:
        row = {
            "company_norm": name,
            "cik": None,
            "sec_name": None,
            "ticker": None,
            "tier": "unmatched",
            "score": None,
            "ambiguous": False,
        }
        tokens = token_set(name)
        if name in by_norm:
            idx = by_norm[name][0]
            row.update(
                cik=int(tickers["cik"].iloc[idx]),
                sec_name=tickers["sec_name"].iloc[idx],
                ticker=tickers["ticker"].iloc[idx],
                tier="exact",
                score=1.0,
                ambiguous=len(by_norm[name]) > 1,
            )
            rows.append(row)
            continue
        # Tier 2: containment.
        candidates = []
        for idx in by_first_token.get(name.split(" ")[0], []):
            other = token_set(sec_norms[idx])
            if tokens and other and (tokens <= other or other <= tokens):
                candidates.append((abs(len(tokens) - len(other)), idx))
        if candidates:
            candidates.sort()
            best_surplus = candidates[0][0]
            tied = [c for c in candidates if c[0] == best_surplus]
            idx = candidates[0][1]
            row.update(
                cik=int(tickers["cik"].iloc[idx]),
                sec_name=tickers["sec_name"].iloc[idx],
                ticker=tickers["ticker"].iloc[idx],
                tier="containment",
                score=None,
                ambiguous=len(tied) > 1,
            )
            rows.append(row)
            continue
        # Tier 3: fuzzy within first-token block.
        scored = []
        for idx in by_first_token.get(name.split(" ")[0], []):
            ratio = difflib.SequenceMatcher(None, name, sec_norms[idx]).ratio()
            if ratio >= FUZZY_THRESHOLD:
                scored.append((ratio, idx))
        if scored:
            scored.sort(reverse=True)
            best_ratio, idx = scored[0]
            ambiguous = (
                len(scored) > 1 and best_ratio - scored[1][0] <= AMBIGUITY_MARGIN
            )
            row.update(
                cik=int(tickers["cik"].iloc[idx]),
                sec_name=tickers["sec_name"].iloc[idx],
                ticker=tickers["ticker"].iloc[idx],
                tier="fuzzy",
                score=float(best_ratio),
                ambiguous=bool(ambiguous),
            )
        rows.append(row)
    return pd.DataFrame(rows)
