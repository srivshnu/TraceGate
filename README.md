# TraceGate — Audit Intelligence Platform

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat&logo=python&logoColor=white)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.63-FF4B4B?style=flat&logo=streamlit&logoColor=white)](https://streamlit.io)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-Random_Forest-F7931E?style=flat&logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![Gemini](https://img.shields.io/badge/Gemini_1.5_Flash-LLM-4285F4?style=flat&logo=google&logoColor=white)](https://deepmind.google/technologies/gemini/)

> A real-time ML-powered invoice fraud detection and audit explanation platform built for internal auditors and field teams.

---

## Architecture

```
Simulator Daemon  →  SQLite DB  →  Random Forest ML  →  Streamlit Live UI
                                                      ↘  Gemini LLM Explainer
```

- **Data Layer:** SQLite database (`data/tracegate.db`) updated every 60 seconds by the simulator daemon
- **ML Layer:** Random Forest Regressor trained on vehicle telemetry (engine temp, RPM variance) to predict expected repair cost
- **Anomaly Detection:** Z-Score statistical deviation flags invoices > 2.5 standard deviations from model predictions
- **Explainability:** Google Gemini LLM generates a plain-English 5–6 sentence audit report for each flagged transaction
- **UI:** Streamlit dashboard with live auto-refresh, KPI cards, anomaly alerts, bar chart, and formatted feed table

---

## Project Structure

```
TraceGate/
├── .env                        # GEMINI_API_KEY goes here
├── .streamlit/config.toml      # Dark professional theme
├── pyproject.toml              # Package config & pytest settings
├── requirements.txt
├── pipeline.py                 # One-time: trains ML model & seeds DB
├── simulator.py                # Entry point: live transaction daemon
├── src/
│   ├── config.py               # Centralised paths, env vars, logging
│   ├── data/
│   │   └── generator.py        # Synthetic telemetry + invoice data
│   ├── models/
│   │   └── anomaly_detector.py # Random Forest training & DB seeding
│   ├── services/
│   │   ├── llm_explainer.py    # Gemini API service
│   │   └── prompts.json        # Structured JSON prompt schema
│   ├── simulation/
│   │   └── simulator.py        # Live transaction generator (60s loop)
│   └── app/
│       └── main.py             # Streamlit dashboard
└── tests/
    ├── conftest.py
    ├── test_data_generator.py
    ├── test_anomaly_detector.py
    └── test_llm_prompt.py
```

---

## Setup & Run

### 1. Create the Virtual Environment
```bash
python -m venv .venv
.\.venv\Scripts\Activate.ps1      # Windows PowerShell
pip install -r requirements.txt
```

### 2. Add your Gemini API Key
Edit `.env`:
```
GEMINI_API_KEY=your_key_here
```

### 3. Run the ML Pipeline (one time)
```bash
python pipeline.py
```

### 4. Start the Live Simulator (Terminal 1)
```bash
python simulator.py
```

### 5. Launch the Dashboard (Terminal 2)
```bash
streamlit run src/app/main.py
```

### 6. Run Tests
```bash
pytest
```

---

## Power BI Integration

Since TraceGate uses SQLite as its data store, integrating Power BI is native:

- **Direct:** Power BI Desktop → Get Data → ODBC → connect to `data/tracegate.db`
- **Export:** Use the Streamlit Excel download button to export audit packs, then import into Power BI via a Folder connector for automated ingestion
