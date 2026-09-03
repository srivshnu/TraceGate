import pytest
import sys
from pathlib import Path

# Ensure the project root is on the path for all tests
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


@pytest.fixture(scope="session")
def sample_transaction():
    """A reusable, realistic transaction dict for testing."""
    return {
        "vin": "A1B2345CD6EFGH789",
        "dealership_id": "DL-402",
        "engine_temp_c": 91.2,
        "rpm_variance": 198.5,
        "billed_cost": 4350.00,
        "ml_predicted_cost": 512.00,
        "discrepancy": 3838.00,
        "z_score": 7.82,
        "status": "ANOMALY",
    }
