from __future__ import annotations

import io
import json
from pathlib import Path

import pandas as pd
import streamlit as st

from src.data_loader import load_dns_csv, load_dns_pcap, normalize_label
from src.feature_extraction import extract_dns_features
from src.model_training import train_models, save_models
from src.evaluation import evaluate_model
from src.detection import score_records

st.set_page_config(page_title="DNS Tunnelling Detection", page_icon="🛡️", layout="wide")

st.title("DNS Tunnelling Detection")
st.caption("Statistical traffic analysis and unsupervised anomaly detection")
st.info(
    "Offline, dataset-driven research prototype. Models learn from benign DNS traffic only. "
    "This application does not capture live traffic, block connections, or prove performance in a live network."
)

with st.sidebar:
    st.header("Project scope")
    st.write("• Conventional DNS over IPv4, UDP/TCP port 53")
    st.write("• Isolation Forest and One-Class SVM")
    st.write("• Benign-only model training")
    st.write("• Held-out evaluation using labelled records")
    st.divider()
    st.caption("Do not upload sensitive production traffic unless you are authorised to do so.")

tab_train, tab_evaluate, tab_detect, tab_guide = st.tabs(
    ["1. Train models", "2. Evaluate models", "3. Detect anomalies", "Dataset guide"]
)

def _column_mapping(df: pd.DataFrame, key_prefix: str):
    cols = ["(not available)"] + list(df.columns)
    domain = st.selectbox("Domain / DNS query-name column", list(df.columns), key=f"{key_prefix}_domain")
    label = st.selectbox("Label column (optional here)", cols, key=f"{key_prefix}_label")
    query_size = st.selectbox("Query size column", cols, key=f"{key_prefix}_qsize")
    response_size = st.selectbox("Response size column", cols, key=f"{key_prefix}_rsize")
    timestamp = st.selectbox("Timestamp column", cols, key=f"{key_prefix}_time")
    response_count = st.selectbox("Response count column", cols, key=f"{key_prefix}_rcount")
    return {
        "domain": domain,
        "label": None if label == cols[0] else label,
        "query_size": None if query_size == cols[0] else query_size,
        "response_size": None if response_size == cols[0] else response_size,
        "timestamp": None if timestamp == cols[0] else timestamp,
        "response_count": None if response_count == cols[0] else response_count,
    }

with tab_train:
    st.subheader("Train using benign DNS traffic")
    st.write("Upload a CSV containing benign DNS observations. The training upload must contain benign records only.")
    train_file = st.file_uploader("Benign training CSV", type=["csv"], key="train_csv")
    if train_file:
        try:
            raw_train = pd.read_csv(train_file)
            st.write(f"Loaded {len(raw_train):,} rows and {len(raw_train.columns)} columns.")
            mapping = _column_mapping(raw_train, "train")
            if st.button("Extract features and train both models", type="primary"):
                clean = load_dns_csv(raw_train, mapping)
                if mapping["label"]:
                    labels = clean["label"].map(normalize_label)
                    if not labels.eq(0).all():
                        st.error("Training data contains non-benign labels. Use a benign-only training file.")
                        st.stop()
                features = extract_dns_features(clean)
                if len(features) < 10:
                    st.error("At least 10 usable DNS observations are required for training.")
                    st.stop()
                models = train_models(features)
                st.session_state["dns_models"] = models
                st.session_state["dns_feature_columns"] = list(features.columns)
                st.session_state["dns_training_features"] = features
                save_models(models, Path("models"))
                st.success(f"Both models trained on {len(features):,} benign observations.")
                st.dataframe(features.head(10), use_container_width=True)
        except Exception as exc:
            st.error(f"Could not process training CSV: {exc}")

with tab_evaluate:
    st.subheader("Evaluate on separate held-out data")
    if "dns_models" not in st.session_state:
        st.warning("Train the models in the first tab before evaluation.")
    eval_file = st.file_uploader("Held-out evaluation CSV (benign and malicious)", type=["csv"], key="eval_csv")
    if eval_file and "dns_models" in st.session_state:
        try:
            raw_eval = pd.read_csv(eval_file)
            st.write(f"Loaded {len(raw_eval):,} rows.")
            mapping = _column_mapping(raw_eval, "eval")
            if not mapping["label"]:
                st.error("An evaluation label column is required. Labels are used only for evaluation, never training.")
            else:
                if st.button("Run evaluation", type="primary"):
                    clean = load_dns_csv(raw_eval, mapping)
                    y = clean["label"].map(normalize_label)
                    if y.isna().any():
                        st.error("Some evaluation labels could not be interpreted. Use benign/normal/0 and malicious/tunnel/1 labels.")
                        st.stop()
                    X = extract_dns_features(clean)
                    results = {}
                    cols = st.columns(2)
                    for (name, model), col in zip(st.session_state["dns_models"].items(), cols):
                        result = evaluate_model(model, X, y)
                        results[name] = result
                        with col:
                            st.markdown(f"**{name}**")
                            m1, m2 = st.columns(2)
                            m1.metric("Recall / detection rate", f"{result['recall']:.3f}")
                            m2.metric("Precision", f"{result['precision']:.3f}")
                            m1.metric("F1-score", f"{result['f1']:.3f}")
                            m2.metric("False positive rate", f"{result['fpr']:.3f}")
                            st.write("Confusion matrix (rows: actual 0/1; columns: predicted 0/1)")
                            st.dataframe(pd.DataFrame(result["confusion_matrix"], index=["Actual benign", "Actual malicious"], columns=["Predicted benign", "Predicted malicious"]))
                    comparison = pd.DataFrame(results).T[["recall", "precision", "f1", "fpr"]]
                    st.subheader("Model comparison")
                    st.dataframe(comparison.style.format("{:.3f}"), use_container_width=True)
                    st.session_state["dns_last_evaluation"] = comparison
        except Exception as exc:
            st.error(f"Evaluation failed: {exc}")

with tab_detect:
    st.subheader("Score DNS observations")
    if "dns_models" not in st.session_state:
        st.warning("Train the models first.")
    detect_file = st.file_uploader("CSV to inspect", type=["csv"], key="detect_csv")
    if detect_file and "dns_models" in st.session_state:
        try:
            raw = pd.read_csv(detect_file)
            mapping = _column_mapping(raw, "detect")
            selected = st.selectbox(
                "Model",
                list(st.session_state["dns_models"].keys()),
                key="detect_model",
            )
            if st.button("Analyse uploaded DNS records"):
                clean = load_dns_csv(raw, mapping)
                X = extract_dns_features(clean)
                scored = score_records(st.session_state["dns_models"][selected], X)
                output = pd.concat([clean.reset_index(drop=True), scored.reset_index(drop=True)], axis=1)
                st.metric("Potential anomalies", int(output["prediction"].eq("Anomaly").sum()))
                st.dataframe(output, use_container_width=True)
                st.download_button(
                    "Download scored records",
                    output.to_csv(index=False).encode("utf-8"),
                    file_name="dns_detection_results.csv",
                    mime="text/csv",
                )
        except Exception as exc:
            st.error(f"Detection failed: {exc}")

with tab_guide:
    st.subheader("Expected dataset structure")
    st.markdown("""
Each CSV row should represent one DNS query/observation. At minimum, provide a domain or
DNS query-name field. The column mapper lets you select your actual column names.

**Recommended fields**
- Domain / query name (required): e.g. `www.example.com`
- Label (evaluation only): benign/normal/0 or malicious/tunnel/1
- Query size and response size (bytes), where available
- Timestamp, where available
- Response count, where available

**Workflow**
1. Prepare a benign-only CSV for training.
2. Upload a separate held-out CSV containing both benign and malicious examples.
3. Select the label column only in the evaluation upload.
4. Compare recall, precision, F1-score and false positive rate.

The pipeline derives domain length, label lengths, Shannon character entropy, character ratios,
per-domain query frequency, response-size statistics and query/response characteristics.
Unavailable optional fields are handled conservatively and are documented in the results.

**PCAP/PCAPNG:** packet extraction is available through the Python helper in `src/data_loader.py`
when Scapy is installed. Convert extracted records to CSV for use in the interface.
""")
    st.code("streamlit run app.py", language="bash")
