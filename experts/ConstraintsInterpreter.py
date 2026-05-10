import os
import torch
import requests
from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer

# 1. Blackwell(sm_121) 하드웨어 호환성 강제 설정
os.environ["TORCH_CUDA_ARCH_LIST"] = "12.1"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    
class ConstraintsInterpreter():
    def __init__(self,model_name,url):
        self.url = url
        self.model_name = model_name
        self.role_description = 'Your task is to take a basic mathematical formulation and refine the constraints section with extreme precision.'
        self.task = """
Review the provided problem description and the initial basic interpretation. 
Your task is to exhaustively identify and refine all mathematical constraints. 
For each constraint, you must specify its 'type', the 'indices' it covers, and the 'variables' and 'parameters' it 'involves', as per the required JSON schema. 
Ensure the logic is complete and ready for code implementation.
"""
        self.rules = """[STRICT SYMBOLIC MODELING RULES]
1. NO_HARDCODING: 
   - Never use raw numbers (e.g., 150, 30, 0.1) in the "Latex Model". 
   - Use the 'id' of the Parameter defined in the previous step.
   - If a constant is missing an id, define a new Parameter id first.

2. SYMBOLIC INDICATOR ALGEBRA: 
   - "If A, then order at least Value": $x_A \ge \text{min\_qty\_A} \cdot y_A$ (Use parameter IDs, not numbers).
   - "If A, then B": $y_B \ge y_A$.

3. BIG-M HANDLING:
   - Do not use 10000. Use a parameter id 'M' and ensure it is defined in the "Parameter" section.

4. INDEX CONSISTENCY: 
   - Use clear indices (e.g., $i \in I$). Ensure all summations $\sum$ have explicit ranges using set IDs.

5. LINKING LOGIC:
   - Always involve the 'id's of both the Decision Variables and Parameters in the "involves" field.

[Output Rules]
1. Response must be a single valid JSON object.
2. For all LaTeX commands, use double backslashes (\\\\) to prevent JSON encoding errors.
3. Maintain the IDs (Sets, Parameters, Variables) from basic_interpretation.
4. No additional text, explanations, or markdown blocks."""
        
        self.output_format = """
{
  "Set": [
    { "id": "I", "User Query": "Suppliers A, B, C", "Latex Model": "I" }
  ],
  "Parameter": [
    { "id": "min_demand", "User Query": "At least 150 units", "Latex Model": "D_{min}" }
  ],
  "Decision Variables": [
    { "id": "x", "index": ["i"], "domain": "integer", "Latex Model": "x_{i}" }
  ],
  "Objective function": [
    { "id": "OBJ", "sense": "min", "Latex Model": "\\\\sum_{i \\\\in I} (cost_i \\\\cdot x_i)" }
  ],
  "Constraints": [
    { 
      "id": "C_demand", 
      "type": "demand_satisfaction", 
      "User Query": "The restaurant needs to order at least 150 tables.", 
      "Latex Model": "\\\\sum_{i \\\\in I} (units_i \\\\cdot x_i) \\\\geq min_demand", 
      "index": [], 
      "involves": { 
        "variables": ["x"], 
        "parameters": ["units", "min_demand"] 
      } 
    },
    {
      "id": "C_linking",
      "type": "indicator_linking",
      "User Query": "If supplier i is used, y_i is 1.",
      "Latex Model": "x_i \\\\leq M \\\\cdot y_i",
      "index": ["i"],
      "involves": {
        "variables": ["x", "y"],
        "parameters": ["M"]
      }
    }
  ]
}
"""
        
    def interpret(self, problem_description,basic_interpretation):
        problem = problem_description
        full_prompt = f"""[Role] {self.role_description}
[Problem] {problem}
[Basic Interpretation] {basic_interpretation}
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
        print("constraints interpretation 요청중...")
        try:
            response = requests.post(self.url, json=payload)
            response.raise_for_status()
            print("constraints interpretation 완료!")
            return response.json().get('response', '').strip()       
        
        except Exception as e:
            return f"constraints interpretation 실패: {e}"