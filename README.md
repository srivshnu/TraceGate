# TraceGate ML Audit Application

An enterprise-grade ML system for auditing internal financial discrepancies and identifying fraud using Random Forest and LLMs.

## Project Structure
This repository uses an industry-standard structural layout for Machine Learning engineering:
- `src/config.py`: Centralized environment & path configuration.
- `src/data/`: Data synthesis and preprocessing.
- `src/models/`: Machine learning training, predictions, and persistence.
- `src/services/`: External API integrations (Gemini LLM).
- `src/app/`: Streamlit web interface.
- `pipeline.py`: A unified execution pipeline.

## How to Run

1. **Install Requirements:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Add your API Key:**
   Update the `.env` file with your `GEMINI_API_KEY`.

3. **Run the Backend ML Pipeline:**
   Generates mock data, trains the Random Forest model, and predicts anomalies.
   ```bash
   python pipeline.py
   ```

4. **Launch the UI Dashboard:**
   ```bash
   streamlit run src/app/main.py
   ```
