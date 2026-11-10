# DNS Tunnelling Detection

An offline, dataset-driven Intrusion Detection System (IDS) research prototype for analysing conventional DNS traffic using statistical feature extraction and two unsupervised anomaly-detection algorithms: **Isolation Forest** and **One-Class SVM**.

## Research alignment

The application implements the stated project objectives:

1. Preprocess public cybersecurity datasets containing benign DNS traffic and DNS tunnelling examples.
2. Extract statistical features including Shannon character entropy, query frequency, query-to-response characteristics, domain label lengths and response characteristics.
3. Train Isolation Forest and One-Class SVM using **benign DNS observations only**.
4. Evaluate against a separate held-out dataset containing benign and malicious observations.
5. Compare recall/detection rate, precision, F1-score and false positive rate (FPR).

The intended scope is historical, publicly available data and conventional IPv4 DNS over UDP/TCP, particularly port 53. This is not a live monitoring or automated prevention system.

## Features

- CSV upload with flexible column mapping.
- Benign-only model training.
- Statistical DNS feature extraction.
- Isolation Forest and One-Class SVM.
- Held-out labelled evaluation and confusion matrices.
- Per-record anomaly scoring and downloadable CSV output.
- Optional PCAP/PCAPNG extraction helper using Scapy.

## Requirements

- Python 3.10 or later recommended.
- pip.

## Installation

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt
```

## Run the application

```bash
streamlit run app.py
```

Streamlit will display a local URL in the terminal, usually `http://localhost:8501`.

## Dataset format

The user interface accepts CSV files. Each row should represent one DNS query/observation. A domain or DNS query-name column is required. Other columns are optional and mapped in the interface.

Recommended columns:

| Meaning | Example column | Required |
|---|---|---|
| Domain / query name | `query_name`, `domain`, `qname` | Yes |
| Binary class label | `label`, `class`, `is_malicious` | Evaluation only |
| Query size in bytes | `query_size` | No |
| Response size in bytes | `response_size` | No |
| Timestamp | `timestamp` | No |
| Response count | `response_count` | No |

Labels recognised by the application include benign/normal/legitimate/0 and malicious/tunnel/tunnelling/attack/1. Verify the meaning and encoding of the dataset labels before evaluation.

### Training and evaluation discipline

1. Prepare a training CSV containing benign observations only.
2. Upload it in **Train models**. If a label column is provided, every value must map to benign.
3. Prepare a separate held-out evaluation CSV containing both benign and malicious observations.
4. Select the label column only for evaluation.
5. Compare both models using recall, precision, F1 and FPR.

Labels are not used to fit the scaler or either model. The evaluation function uses labels only after model predictions have been produced. Do not train on the evaluation dataset.

## Feature engineering

The pipeline currently extracts:

- Total domain-name length.
- Maximum and average label length.
- Number of domain labels.
- Shannon character entropy.
- Digit and alphabetic character ratios.
- Hyphen count.
- Per-domain query frequency within the uploaded dataset.
- Query size and response size.
- Mean response size per domain.
- Response count per query.
- Response-to-query size ratio.
- Response-size availability indicator.

Some datasets do not contain all traffic metadata. Missing optional fields are represented conservatively as zero. This can affect model quality; document missing fields in the research evaluation and do not interpret unavailable measurements as observed zero traffic. The current query-frequency and per-domain response aggregates are calculated within the uploaded file, so training, evaluation and detection files should represent comparable observation windows and collection procedures. They are not persistent rolling counts across files.

## PCAP / PCAPNG extraction helper

The helper `load_dns_pcap(path)` in `src/data_loader.py` uses Scapy to extract basic DNS query-name observations from a conventional DNS capture. It is an offline file reader only. For research use, validate extracted fields and capture semantics against the dataset documentation before training. The Streamlit interface currently accepts CSV files directly; convert PCAP-extracted observations to CSV before uploading.

## Tests

Run:

```bash
pytest -q
```

The tests cover feature extraction, entropy, model fitting and held-out metric calculation. They use small synthetic fixtures only for software tests; they are **not** research datasets and do not represent measured detection performance.

## Model artifacts and privacy

Trained model bundles are saved under `models/` as Joblib files when training runs locally. These generated binary artifacts are excluded from Git by `.gitignore`. Do not upload confidential traffic or personal data to a public repository.

## GitHub and Streamlit Community Cloud

1. Create a GitHub repository for this project.
2. Upload the project files, excluding `.venv`, private data, and generated model artifacts.
3. In Streamlit Community Cloud, create an app from the repository.
4. Set the main file path to `app.py`.
5. Deploy the app. Streamlit installs dependencies from `requirements.txt`.

Check the current Streamlit Community Cloud instructions before deployment, as platform settings may change.

## Limitations and responsible interpretation

- The system is offline and dataset-driven; it does not capture live network packets or continuously monitor a network.
- It does not block traffic or automatically respond to detected anomalies.
- It does not implement DoH or DoT analysis.
- It does not generate DNS tunnelling traffic or conduct penetration testing.
- An anomaly is a deviation from the learned benign baseline, not proof of malicious intent.
- Results depend on dataset quality, representativeness, feature availability, class balance and the chosen model parameters.
- Do not claim that evaluation on historical public datasets proves performance in a live enterprise environment.
- No performance figures are supplied by this repository. Report only results obtained from the selected and documented datasets.

## Project structure

```text
dns-tunnelling-detection/
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
├── src/
│   ├── __init__.py
│   ├── data_loader.py
│   ├── preprocessing.py
│   ├── feature_extraction.py
│   ├── model_training.py
│   ├── evaluation.py
│   └── detection.py
├── models/
│   └── .gitkeep
├── data/
│   └── .gitkeep
├── reports/
│   └── .gitkeep
└── tests/
    ├── __init__.py
    ├── test_features.py
    └── test_preprocessing.py
```
