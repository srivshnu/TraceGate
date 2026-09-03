import json
import google.generativeai as genai
from pathlib import Path
from src.config import GEMINI_API_KEY, setup_logger

logger = setup_logger(__name__)

class LLMExplainer:
    """Service abstraction for integrating with Google's Gemini LLM for plain-text explanations."""
    
    def __init__(self):
        if not GEMINI_API_KEY or GEMINI_API_KEY == "your_api_key_here":
            logger.warning("GEMINI_API_KEY is missing or invalid. LLM features may fail.")
        else:
            genai.configure(api_key=GEMINI_API_KEY)
            
        self._load_prompts()
        
        # Serialize system instruction JSON object to a string for Gemini
        system_instruction_str = json.dumps(self.prompts["audit_explainer"]["system_instruction"], indent=2)
        
        self.model = genai.GenerativeModel(
            "gemini-1.5-flash",
            system_instruction=system_instruction_str
        )
        
    def _load_prompts(self):
        """Loads prompt templates from the external JSON file."""
        prompt_path = Path(__file__).parent / "prompts.json"
        with open(prompt_path, "r") as f:
            self.prompts = json.load(f)
            
    def generate_explanation(self, billed_cost: float, predicted_cost: float, engine_temp: float) -> str:
        """Populates dynamic JSON prompt values and serializes to JSON string."""
        template_obj = self.prompts["audit_explainer"]["dynamic_template"]
        
        # Inject dynamic values into structured dictionary
        prompt_data = {
          "function": template_obj["function"],
          "inputs": {
            "billed_amount_usd": round(billed_cost, 2),
            "ml_expected_amount_usd": round(predicted_cost, 2),
            "sensor_engine_temp_c": round(engine_temp, 2)
          },
          "constraints": template_obj["constraints"]
        }
        
        # Convert dictionary directly to JSON string for LLM execution
        prompt = json.dumps(prompt_data, indent=2)
        
        try:
            logger.info("Requesting explanation from Gemini LLM using JSON prompt...")
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            logger.error(f"Failed to generate LLM explanation: {e}")
            return "⚠️ Could not generate explanation. Ensure your GEMINI_API_KEY is correctly set in the .env file."
