import os
import torch
import requests
from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer

os.environ["TORCH_CUDA_ARCH_LIST"] = "12.1"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    
class BasicModelInterpreter():
    def __init__(self,model_name,url):
        self.url = url
        self.model_name = model_name
        self.role_description = 'Your role is to perform a "Full-Sentence-Extraction" to ensure no information loss for mathematical modeling.'
        
        self.task = """1. Identify all entities (Suppliers, Nodes, Products).
2. [CRITICAL] Parameterize All Constants: Convert every specific number (e.g., $120, 30 units) into a Parameter ID (e.g., cost_A, min_requirement).
3. Determine decision domains (Integer/Continuous/Binary).
4. Identify "Conditional Logic" using Symbolic IDs only.
5. Identify "Unit Consistency": Ensure you don't add 'Orders' to 'Tables'."""
        
        self.rules = """[STRICT SYMBOLIC MODELING RULES]
1. NO_HARDCODING:
   - Never use raw numbers (e.g., 120, 0.05, 1000) directly in the "Latex Model".
   - Every number in the 'User Query' must first be defined in the "Parameter" section with a unique 'id'.
   - The "Latex Model" must ONLY use these 'id's.

2. PARAMETER_MAPPING:
   - Example: "Cost is $120" -> Create Parameter { "id": "cost_A", "Latex Model": "c_A" }.
   - Sucessful Latex: "c_A * x_A", NOT "120 * x_A".

3. LOGIC_MAPPING (Big-M):
   - Use 'M' as a symbolic parameter for linking constraints. Define 'M' in the Parameter section if needed.

[LOGIC EXAMPLE (SYMBOLIC)]
- Rule: "If Supplier A is chosen, order at least 30 units."
- Correct Parameter: { "id": "min_qty_A", "Latex Model": "Q_{min,A}" }
- Correct Logic: x_A >= Q_{min,A} * y_A  (Symbolic IDs Only)
[Output Rules]
1. Response must be a single valid JSON object.
2. No conversational text or markdown code blocks.
3. Use double backslashes (\\\\) for all LaTeX commands to ensure JSON compatibility.
   """
        self.output_format = """
{
  "Set": [
    { 
      "id": "I", 
      "User Query": "The set of suppliers A, B, and C.", 
      "Latex Model": "I" 
    }
  ],
  "Parameter": [
    { 
      "id": "cost_A", 
      "User Query": "Cost of ordering from Supplier A is $120.", 
      "Latex Model": "c_{A}",
      "description": "Unit cost from supplier A" 
    }
  ],
  "Decision Variables": [
    { 
      "id": "x_A", 
      "User Query": "Number of orders from Supplier A.", 
      "Latex Model": "x_{A}", 
      "index": [], 
      "domain": "integer", 
      "lower_bound": 0 
    }
  ],
  "Objective function": [
    { 
      "id": "OBJ", 
      "sense": "min", 
      "User Query": "Minimize the total cost.", 
      "Latex Model": "\\\\min \\\\quad cost_A \\\\cdot x_A + cost_B \\\\cdot x_B" 
    }
  ],
  "Constraints": [
    { 
      "id": "C1", 
      "User Query": "Must order at least 150 tables.", 
      "Latex Model": "units_A \\\\cdot x_A + units_B \\\\cdot x_B \\\\geq min_total",
      "involves": {
        "variables": ["x_A", "x_B"],
        "parameters": ["units_A", "units_B", "min_total"]
      }
    }
  ]
}
"""
        
    def interpret(self, problem_description,feedback):
        problem = problem_description
        full_prompt = f"""[Role] {self.role_description}
[Problem] {problem}
[Feedback] {feedback}
[Task] {self.task}
[Rules] {self.rules}
[Format] {self.output_format}"""
        
        payload = {
            "model": self.model_name,
            "prompt": full_prompt,
            "stream": False,
            "options": {
                "temperature": 0.1,
                "num_predict": 2048   
            }
        }
        print("basic interpretation 요청중...")
        try:
            response = requests.post(self.url, json=payload)
            response.raise_for_status()
            print("basic interpretation 완료!")
            return response.json().get('response', '').strip()       
        
        except Exception as e:
            return f"basic interpretation 실패: {e}"
        
                
