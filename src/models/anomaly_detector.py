import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from src.config import RAW_DATA_DIR, PROCESSED_DATA_DIR, setup_logger

logger = setup_logger(__name__)

class AnomalyDetector:
    """Trains an ML model on sensory ground truth to predict actual costs and flag deviations."""
    
    def __init__(self, z_score_threshold: float = 2.5):
        self.model = RandomForestRegressor(n_estimators=50, random_state=42)
        self.z_score_threshold = z_score_threshold
        
    def _load_data(self) -> pd.DataFrame:
        logger.info("Loading telemetry and invoice datasets.")
        tel_path = RAW_DATA_DIR / "telemetry.csv"
        inv_path = RAW_DATA_DIR / "invoices.csv"
        
        if not tel_path.exists() or not inv_path.exists():
            raise FileNotFoundError("Raw data files not found. Please run the data generator first.")
            
        tel = pd.read_csv(tel_path)
        inv = pd.read_csv(inv_path)
        return pd.merge(inv, tel, on="vin")
        
    def run_pipeline(self):
        """Executes the training and anomaly detection pipeline."""
        df = self._load_data()
        
        # Train ML Model
        logger.info("Training Random Forest Regressor on telemetry inputs...")
        X = df[["engine_temp_c", "rpm_variance"]]
        y = df["billed_cost"]
        self.model.fit(X, y)
        
        # Predict Expected Costs
        logger.info("Executing predictions and evaluating discrepancies...")
        df["ml_predicted_cost"] = self.model.predict(X)
        df["discrepancy"] = df["billed_cost"] - df["ml_predicted_cost"]
        
        # Statistical Profiling (Z-Score)
        df["z_score"] = df["discrepancy"] / df["discrepancy"].std()
        df["status"] = np.where(df["z_score"] > self.z_score_threshold, "ANOMALY", "NORMAL")
        
        anomalies = df[df["status"] == "ANOMALY"]
        
        # Persist Flagged Artifacts
        output_path = PROCESSED_DATA_DIR / "flagged_anomalies.csv"
        anomalies.to_csv(output_path, index=False)
        
        logger.info(f"Anomaly Detection Complete. Flagged {len(anomalies)} suspicious records.")
        logger.info(f"Persisted anomalies to {output_path}")
