import pandas as pd
import pytest

from src.data_loader import load_dns_csv, normalize_label


def test_normalize_label_accepts_expected_binary_spellings():
    assert normalize_label("benign") == 0
    assert normalize_label("normal") == 0
    assert normalize_label("tunnelling") == 1
    assert normalize_label("malicious") == 1
    assert pd.isna(normalize_label("unknown-class"))


def test_load_dns_csv_maps_columns_and_normalizes_values():
    frame = pd.DataFrame({
        "qname": ["WWW.Example.COM.", "tunnel.example.net"],
        "class": ["benign", "malicious"],
        "qbytes": ["42", "not-a-number"],
        "rbytes": [120, None],
    })
    mapped = load_dns_csv(frame, {
        "domain": "qname", "label": "class", "query_size": "qbytes",
        "response_size": "rbytes", "timestamp": None, "response_count": None,
    })
    assert mapped.loc[0, "domain"] == "www.example.com."
    assert mapped["label"].tolist() == [0, 1]
    assert mapped.loc[0, "query_size"] == 42
    assert mapped.loc[1, "query_size"] == 0
    assert mapped.loc[1, "response_size"] == 0


def test_load_dns_csv_requires_a_domain_column():
    with pytest.raises(ValueError):
        load_dns_csv(pd.DataFrame({"other": [1]}), {"domain": "missing"})
