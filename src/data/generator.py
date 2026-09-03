import pandas as pd
import numpy as np
from faker import Faker
import random
from src.config import RAW_DATA_DIR, setup_logger

logger = setup_logger(__name__)

class DataGenerator:
    """Generates synthetic telemetry and invoice data to simulate field operations and fraud."""
    
    def __init__(self, num_records: int = 5000, random_seed: int = 42):
        self.num_records = num_records
        self.fake = Faker()
        
        # Set seeds for reproducibility
        np.random.seed(random_seed)
        random.seed(random_seed)
        
        # Dealership DL-402 is designated as the target anomaly (fraud)
        self.dealers = [f"DL-{i:03d}" for i in range(100, 110)] + ["DL-402"]

    def _generate_telemetry(self) -> pd.DataFrame:
        logger.info(f"Synthesizing telemetry data for {self.num_records} vehicles.")
        return pd.DataFrame({
            "vin": [self.fake.unique.bothify(text='?#?####??#???????').upper() for _ in range(self.num_records)],
            "engine_temp_c": np.random.normal(90, 5, self.num_records), 
            "rpm_variance": np.random.normal(200, 50, self.num_records)
        })

    def _generate_invoices(self, telemetry: pd.DataFrame) -> pd.DataFrame:
        logger.info("Synthesizing repair invoice data.")
        # Baseline cost calculation based on physical sensor health
        expected_cost = (telemetry["engine_temp_c"] - 90).abs() * 50 + (telemetry["rpm_variance"] * 2)
        
        invoices = pd.DataFrame({
            "vin": telemetry["vin"],
            "dealership_id": [random.choice(self.dealers) for _ in range(self.num_records)],
            "billed_cost": expected_cost + np.random.normal(100, 50, self.num_records)
        })
        
        # Inject deliberate fraud for DL-402 (Massive overbilling for healthy sensors)
        logger.info("Injecting structural anomalies (fraud) for dealership DL-402...")
        fraud_idx = invoices['dealership_id'] == 'DL-402'
        invoices.loc[fraud_idx, 'billed_cost'] += np.random.uniform(3000, 5000, sum(fraud_idx))
        
        return invoices

    def run(self):
        """Executes the data generation pipeline and persists to the raw data directory."""
        telemetry = self._generate_telemetry()
        invoices = self._generate_invoices(telemetry)
        
        telemetry_path = RAW_DATA_DIR / "telemetry.csv"
        invoices_path = RAW_DATA_DIR / "invoices.csv"
        
        telemetry.to_csv(telemetry_path, index=False)
        invoices.to_csv(invoices_path, index=False)
        
        logger.info(f"Successfully generated and saved datasets to {RAW_DATA_DIR}")
