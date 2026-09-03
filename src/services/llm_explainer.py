import json
import google.generativeai as genai
from pathlib import Path
from src.config import GEMINI_API_KEY, setup_logger

logger = setup_logger(__name__)


class LLMExplainer:
    """Service abstraction for integrating with Google Gemini LLM."""

    def __init__(self):
        self._validate_key()
        self.prompts = self._load_prompts()

    def _validate_key(self):
        if not GEMINI_API_KEY or GEMINI_API_KEY == "your_api_key_here":
            raise ValueError(
                "GEMINI_API_KEY is missing or still set to the placeholder. "
                "Update the .env file in the project root with your actual API key."
            )
        # Configure the SDK with the validated key on every instantiation
        genai.configure(api_key=GEMINI_API_KEY)
        logger.info("Gemini API key loaded and configured successfully.")

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

        try:
            logger.info("Calling Gemini API for anomaly explanation...")
            model = genai.GenerativeModel(
                "gemini-3.5-flash",
                system_instruction=system_str,
            )
            response = model.generate_content(prompt)
            logger.info("Gemini explanation received successfully.")
            return response.text
        except Exception as e:
            logger.error(f"Gemini API call failed: {e}")
            return f"⚠️ AI explanation unavailable: {e}"
