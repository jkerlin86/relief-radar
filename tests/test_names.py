from relief_radar.names import normalize_company, token_set


def test_suffix_stripping():
    assert normalize_company("Bank of America, N.A.") == "BANK OF AMERICA"
    assert normalize_company("ACME BANK INC") == "ACME BANK"
    assert normalize_company("Widgets LLC") == "WIDGETS"
    assert normalize_company("First National Bank, FSB") == "FIRST NATIONAL BANK"


def test_punctuation_and_case():
    assert normalize_company("jpmorgan chase & co.") == "JPMORGAN CHASE"
    assert normalize_company("  Wells   Fargo ") == "WELLS FARGO"


def test_empty_and_none():
    assert normalize_company("") == ""
    assert normalize_company(None) == ""


def test_substantive_tokens_kept():
    # Only suffixes are stripped; real name tokens survive.
    assert normalize_company("Capital One") == "CAPITAL ONE"
    assert token_set("CAPITAL ONE") == frozenset({"CAPITAL", "ONE"})
