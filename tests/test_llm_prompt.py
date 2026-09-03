import json
from pathlib import Path

def test_prompt_json_integrity():
    """Verify that the prompt JSON file exists and contains the correct schema structure."""
    prompt_path = Path(__file__).parent.parent / "src" / "services" / "prompts.json"
    
    assert prompt_path.exists(), "prompts.json file is missing"
    
    with open(prompt_path, "r") as f:
        prompts = json.load(f)
        
    assert "audit_explainer" in prompts
    
    explainer = prompts["audit_explainer"]
    assert "system_instruction" in explainer
    assert "dynamic_template" in explainer
    
    # Check that required variables exist in the dynamic inputs template
    inputs = explainer["dynamic_template"]["inputs"]
    assert "{billed_cost}" in inputs["billed_amount_usd"]
    assert "{predicted_cost}" in inputs["ml_expected_amount_usd"]
