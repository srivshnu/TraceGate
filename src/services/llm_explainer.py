import google.generativeai as genai
from src.config import GEMINI_API_KEY, setup_logger

logger = setup_logger(__name__)

class LLMExplainer:
    """Service abstraction for integrating with Google's Gemini LLM for plain-text explanations."""
    
    def __init__(self):
        if not GEMINI_API_KEY or GEMINI_API_KEY == "your_api_key_here":
            logger.warning("GEMINI_API_KEY is missing or invalid. LLM features may fail.")
        else:
            genai.configure(api_key=GEMINI_API_KEY)
            
        # Initialize standard Flash model for fast generation
        self.model = genai.GenerativeModel("gemini-1.5-flash")
        
    def generate_explanation(self, billed_cost: float, predicted_cost: float, engine_temp: float) -> str:
        """Prompts the LLM to explain the financial discrepancy in non-technical terms."""
        
        prompt = f"""
        You are an AI assistant for an internal audit team. 
        A mechanic billed ${billed_cost:.2f}. However, our Machine Learning model 
        predicted the repair should only cost ${predicted_cost:.2f} based on the car's 
        healthy engine temperature of {engine_temp}C.
        Write exactly 3 simple bullet points explaining why this is suspicious to a non-technical manager.
        """
        
        try:
            logger.info("Requesting explanation from Gemini LLM...")
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            logger.error(f"Failed to generate LLM explanation: {e}")
            return "⚠️ Could not generate explanation. Ensure your GEMINI_API_KEY is correctly set in the .env file."
