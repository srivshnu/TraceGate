"""Entry point for the live transaction simulator daemon."""
from src.simulation.simulator import run

if __name__ == "__main__":
    run(interval_seconds=60)
