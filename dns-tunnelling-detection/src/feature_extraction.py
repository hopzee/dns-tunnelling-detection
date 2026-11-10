from __future__ import annotations

import math
from collections import Counter

import numpy as np
import pandas as pd


def shannon_entropy(value: str) -> float:
    text = str(value or "")
    if not text:
        return 0.0
    counts = Counter(text)
    total = len(text)
    return float(-sum((count / total) * math.log2(count / total) for count in counts.values()))


def extract_dns_features(records: pd.DataFrame) -> pd.DataFrame:
    """Create one numeric feature row per DNS observation, using only traffic-derived fields."""
    if "domain" not in records.columns:
        raise ValueError("Input records must contain a 'domain' column.")
    df = records.copy().reset_index(drop=True)
    domains = df["domain"].fillna("").astype(str).str.lower().str.rstrip(".")
    labels = domains.map(lambda d: [part for part in d.split(".") if part])
    lengths = domains.map(len)
    label_counts = labels.map(len)
    max_label = labels.map(lambda parts: max((len(x) for x in parts), default=0))
    avg_label = labels.map(lambda parts: float(np.mean([len(x) for x in parts])) if parts else 0.0)

    chars = domains.map(lambda d: "".join(ch for ch in d if ch.isalnum()))
    char_lengths = chars.map(len).replace(0, 1)
    digit_counts = chars.map(lambda d: sum(ch.isdigit() for ch in d))
    alpha_counts = chars.map(lambda d: sum(ch.isalpha() for ch in d))
    hyphen_counts = domains.map(lambda d: d.count("-"))
    entropy = domains.map(shannon_entropy)

    query_size = pd.to_numeric(df.get("query_size", 0), errors="coerce")
    response_size = pd.to_numeric(df.get("response_size", 0), errors="coerce")
    response_count = pd.to_numeric(df.get("response_count", 0), errors="coerce")
    if not isinstance(query_size, pd.Series):
        query_size = pd.Series(float(query_size), index=df.index)
    if not isinstance(response_size, pd.Series):
        response_size = pd.Series(float(response_size), index=df.index)
    if not isinstance(response_count, pd.Series):
        response_count = pd.Series(float(response_count), index=df.index)
    query_size = query_size.fillna(0).clip(lower=0)
    response_size = response_size.fillna(0).clip(lower=0)
    response_count = response_count.fillna(0).clip(lower=0)

    # Per-domain frequency and aggregate response behaviour are computed within this uploaded dataset.
    frequency = domains.map(domains.value_counts())
    domain_group = pd.DataFrame({"domain": domains, "response_size": response_size, "response_count": response_count})
    response_mean = domain_group.groupby("domain")["response_size"].transform("mean")
    response_total = domain_group.groupby("domain")["response_count"].transform("sum")
    query_response_ratio = response_total / frequency.replace(0, 1)

    features = pd.DataFrame({
        "domain_length": lengths,
        "max_label_length": max_label,
        "average_label_length": avg_label,
        "label_count": label_counts,
        "shannon_entropy": entropy,
        "digit_ratio": digit_counts / char_lengths,
        "alphabetic_ratio": alpha_counts / char_lengths,
        "hyphen_count": hyphen_counts,
        "query_frequency": frequency,
        "query_size_bytes": query_size,
        "response_size_bytes": response_size,
        "mean_response_size_per_domain": response_mean,
        "response_count_per_query": query_response_ratio,
        "response_to_query_size_ratio": response_size / query_size.replace(0, np.nan),
        "has_response_size": response_size.gt(0).astype(float),
    })
    return features.replace([np.inf, -np.inf], np.nan).fillna(0).astype(float)
