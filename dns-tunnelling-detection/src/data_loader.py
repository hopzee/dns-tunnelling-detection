from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd


def normalize_label(value: Any):
    """Map common binary labels to 0=benign and 1=malicious; unknown values become NaN."""
    if pd.isna(value):
        return float("nan")
    if isinstance(value, (int, float)):
        if value in (0, 1):
            return int(value)
    text = str(value).strip().lower()
    benign = {"0", "benign", "normal", "legitimate", "clean", "good"}
    malicious = {"1", "malicious", "tunnel", "tunnelling", "tunneling", "attack", "bad"}
    if text in benign:
        return 0
    if text in malicious:
        return 1
    return float("nan")


def load_dns_csv(source: pd.DataFrame | str | Path, mapping: dict) -> pd.DataFrame:
    """Normalize selected CSV columns into a consistent event schema."""
    df = source.copy() if isinstance(source, pd.DataFrame) else pd.read_csv(source)
    domain_col = mapping.get("domain")
    if not domain_col or domain_col not in df.columns:
        raise ValueError("A valid domain/query-name column must be selected.")
    out = pd.DataFrame(index=df.index)
    out["domain"] = df[domain_col].astype("string").fillna("").str.strip().str.lower()
    for target in ("query_size", "response_size", "response_count"):
        col = mapping.get(target)
        out[target] = pd.to_numeric(df[col], errors="coerce") if col in df.columns else 0.0
    time_col = mapping.get("timestamp")
    out["timestamp"] = df[time_col] if time_col in df.columns else pd.NaT
    label_col = mapping.get("label")
    if label_col and label_col in df.columns:
        out["label"] = df[label_col].map(normalize_label)
    out = out[out["domain"].ne("")].copy()
    if out.empty:
        raise ValueError("No non-empty domain names were found.")
    for col in ("query_size", "response_size", "response_count"):
        out[col] = out[col].fillna(0).clip(lower=0)
    return out.reset_index(drop=True)


def load_dns_pcap(path: str | Path) -> pd.DataFrame:
    """Extract basic conventional DNS observations from a PCAP/PCAPNG file using Scapy."""
    try:
        from scapy.all import DNS, DNSQR, IP, IPv6, UDP, TCP, rdpcap
    except ImportError as exc:
        raise ImportError("Install scapy to read PCAP files: pip install scapy") from exc

    rows = []
    for packet in rdpcap(str(path)):
        if DNS not in packet or DNSQR not in packet:
            continue
        if IP in packet:
            src, dst = packet[IP].src, packet[IP].dst
        elif IPv6 in packet:
            src, dst = packet[IPv6].src, packet[IPv6].dst
        else:
            continue
        dns = packet[DNS]
        qname = packet[DNSQR].qname.decode("utf-8", errors="replace").rstrip(".")
        rows.append({
            "timestamp": float(packet.time),
            "src_ip": src,
            "dst_ip": dst,
            "protocol": "UDP" if UDP in packet else ("TCP" if TCP in packet else "OTHER"),
            "domain": qname,
            "query_size": len(bytes(packet)),
            "response_size": len(bytes(packet)) if int(dns.qr) == 1 else 0,
            "response_count": 1 if int(dns.qr) == 1 else 0,
        })
    if not rows:
        raise ValueError("No conventional DNS query-name records were extracted from this capture.")
    return pd.DataFrame(rows)
