import pandas as pd
from src.feature_extraction import extract_dns_features, shannon_entropy


def test_entropy_and_domain_features():
    assert shannon_entropy("aaaa") == 0.0
    assert shannon_entropy("abcd") == 2.0
    records = pd.DataFrame({
        "domain": ["www.example.com", "x7k2m9q.example.net", "www.example.com"],
        "query_size": [50, 65, 52],
        "response_size": [120, 0, 130],
        "response_count": [1, 0, 1],
    })
    features = extract_dns_features(records)
    assert len(features) == 3
    assert features.loc[1, "domain_length"] > features.loc[0, "domain_length"]
    assert features.loc[0, "query_frequency"] == 2
    assert features.select_dtypes("number").notna().all().all()
