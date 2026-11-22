import json
from pathlib import Path

from google import genai
from google.genai import types

from src.config import GEMINI_API_KEY, setup_logger

logger = setup_logger(__name__)

# Tried in order; falls through on 404 / 429 / 503 etc.
MODEL_CHAIN = ["gemini-3.8-flash", "gemini-3.5-flash", "gemini-3.5-flash-lite"]


class LLMExplainer:
    """Service abstraction for integrating with Google Gemini (google-genai SDK)."""

    def __init__(self):
        self._validate_key()
        self.client = genai.Client(
            api_key=GEMINI_API_KEY,
            http_options=types.HttpOptions(timeout=20_000),  # ms, per request
        )
        self.prompts = self._load_prompts()

    def _validate_key(self):
        if not GEMINI_API_KEY or GEMINI_API_KEY == "your_api_key_here":
            raise ValueError(
                "GEMINI_API_KEY is missing or still set to the placeholder. "
                "Update the .env file in the project root with your actual API key."
            )
        logger.info("Gemini API key loaded successfully.")

    def _load_prompts(self) -> dict:
        """Loads the structured JSON prompt schema."""
        prompt_path = Path(__file__).parent / "prompts.json"
        with open(prompt_path, "r") as f:
            return json.load(f)

    def generate_explanation(
        self, billed_cost: float, predicted_cost: float, engine_temp: float
    ) -> str:
        """Builds the JSON prompt, calls Gemini, and returns plain-English explanation."""
        template = self.prompts["audit_explainer"]

        system_str = json.dumps(template["system_instruction"], indent=2)
        dynamic_obj = template["dynamic_template"]

        prompt_data = {
            "function": dynamic_obj["function"],
            "inputs": {
                "billed_amount_usd": round(billed_cost, 2),
                "ml_expected_amount_usd": round(predicted_cost, 2),
                "sensor_engine_temp_c": round(engine_temp, 2),
            },
            "constraints": dynamic_obj["constraints"],
        }
        prompt = json.dumps(prompt_data, indent=2)

        last_err = None
        for model_name in MODEL_CHAIN:
            try:
                logger.info(f"Calling Gemini ({model_name}) for anomaly explanation...")
                response = self.client.models.generate_content(
                    model=model_name,
                    contents=prompt,
                    config=types.GenerateContentConfig(system_instruction=system_str),
                )
                logger.info("Gemini explanation received successfully.")
                return response.text
            except Exception as e:
                last_err = e
                logger.error(f"Gemini call failed on {model_name}: {e}")
        return f"⚠️ AI explanation unavailable: {last_err}"
