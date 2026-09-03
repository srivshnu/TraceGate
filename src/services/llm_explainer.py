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
        
        # Initialize standard Flash model using the extracted System Instruction
        self.model = genai.GenerativeModel(
            "gemini-1.5-flash",
            system_instruction=self.prompts["audit_explainer"]["system_instruction"]
        )
        
    def _load_prompts(self):
        """Loads prompt templates from the external JSON file."""
        prompt_path = Path(__file__).parent / "prompts.json"
        with open(prompt_path, "r") as f:
            self.prompts = json.load(f)
            
    def generate_explanation(self, billed_cost: float, predicted_cost: float, engine_temp: float) -> str:
        """Injects dynamic variables into the JSON template and prompts the LLM."""
        template = self.prompts["audit_explainer"]["dynamic_template"]
        
        # Execute normal dynamic instruction via template formatting
        prompt = template.format(
            billed_cost=billed_cost,
            predicted_cost=predicted_cost,
            engine_temp=engine_temp
        )
        
        try:
            logger.info("Requesting explanation from Gemini LLM...")
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            logger.error(f"Failed to generate LLM explanation: {e}")
            return "⚠️ Could not generate explanation. Ensure your GEMINI_API_KEY is correctly set in the .env file."
