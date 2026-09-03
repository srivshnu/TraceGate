import pytest
import pandas as pd
from src.data.generator import DataGenerator

def test_telemetry_generation():
    """Test that telemetry data is generated with correct shapes and columns."""
    gen = DataGenerator(num_records=100, random_seed=42)
    df = gen._generate_telemetry()
    
    assert len(df) == 100
    assert "vin" in df.columns
    assert "engine_temp_c" in df.columns
    assert "rpm_variance" in df.columns

def test_fraud_injection_logic():
    """Test that the targeted dealership has statistically higher billed costs due to fraud."""
    gen = DataGenerator(num_records=1000, random_seed=42)
    telemetry = gen._generate_telemetry()
    invoices = gen._generate_invoices(telemetry)
    
    fraud_dealer = invoices[invoices['dealership_id'] == 'DL-402']
    normal_dealers = invoices[invoices['dealership_id'] != 'DL-402']
    
    if not fraud_dealer.empty and not normal_dealers.empty:
        assert fraud_dealer['billed_cost'].mean() > normal_dealers['billed_cost'].mean()
