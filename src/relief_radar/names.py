"""Company-name normalization shared by the dataset build and the crosswalk.

Normalization is deliberately conservative: case folding, punctuation
removal, whitespace collapsing, and stripping of common corporate
suffixes. It never merges names that differ in substantive tokens.
"""

from __future__ import annotations

import re

_SUFFIXES = {
    "INC",
    "LLC",
    "LLP",
    "LP",
    "CORP",
    "CORPORATION",
    "CO",
    "COMPANY",
    "INCORPORATED",
    "LTD",
    "LIMITED",
    "PLC",
    "NA",
    "NATIONAL ASSOCIATION",
    "FSB",
    "SSB",
}

_PUNCT = re.compile(r"[^A-Z0-9 ]+")
_SPACES = re.compile(r"\s+")


def normalize_company(name: str | None) -> str:
    """Return a normalized comparison key for a company name."""
    if not name:
        return ""
    # Drop periods first so "N.A." becomes the single token "NA"
    # (a strippable suffix) instead of two stray tokens "N" and "A".
    text = _PUNCT.sub(" ", str(name).upper().replace(".", ""))
    text = _SPACES.sub(" ", text).strip()
    tokens = text.split(" ")
    # Strip trailing corporate suffixes (possibly stacked, e.g. "INC LLC").
    changed = True
    while changed and tokens:
        changed = False
        for width in (2, 1):
            tail = " ".join(tokens[-width:])
            if len(tokens) > width and tail in _SUFFIXES:
                tokens = tokens[:-width]
                changed = True
                break
    return " ".join(tokens)


def token_set(name: str) -> frozenset[str]:
    """Token set of an already-normalized name."""
    return frozenset(name.split(" ")) if name else frozenset()
