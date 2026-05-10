import os
import torch
import requests
from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer

# 1. Blackwell(sm_121) 하드웨어 호환성 강제 설정
os.environ["TORCH_CUDA_ARCH_LIST"] = "12.1"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    
class Coder():
    def __init__(self,model_name,url):
        self.url = url
        self.model_name = model_name
        self.role_description = 'Your mission is to generate a Python script that separates mathematical logic from numerical data using external JSON input.'
        self.task = 'Translate the provided symbolic formulation into a Gurobi Python script that loads instance data from a JSON file.'
        
        self.rules = """[STRICT RULES]
1. SYMBOLIC CODING: Do NOT hard-code any numerical values (e.g., 120, 150) inside the model logic.
2. DATA LOADING: Use `argparse` to receive a `--data` argument pointing to a JSON file path. Load this file using the `json` library.
3. PARAMETER MAPPING: Map all LaTeX-defined parameters to their corresponding keys in the loaded JSON data.
   - Example: If LaTeX uses 'cost_A', the code should use 'data["parameters"]["cost_A"]'.
4. BIG-M HANDLING: Use the symbolic 'M' parameter from the JSON. Do not define it as a literal constant like 10000.
5. STANDALONE LOGIC: The script should focus on the mathematical structure. Assume the JSON file contains all necessary keys as defined in the 'Parameter' section of the formulation.
[Output Rules]
1. Respond ONLY with the raw Python code.
2. If there is any reference code provided, use it as a template and ensure all new code adheres to the same structure and style.
2. DO NOT use markdown code blocks (e.g., do not wrap with ```python or ```).
3. DO NOT include any introductory, explanatory, or concluding text. 
4. The response must start immediately with 'import gurobipy' and end with the last line of code.
"""
        
        self.output_format = """
import gurobipy as gp
from gurobipy import GRB
import json
import argparse

# 1. Argument Parsing
parser = argparse.ArgumentParser()
parser.add_argument('--data', type=str, required=True, help='Path to instance_data.json')
args = parser.parse_args()

# 2. Data Loading
with open(args.data, 'r', encoding='utf-8') as f:
    inst_data = json.load(f)

params = inst_data.get('parameters', {})
sets = inst_data.get('sets', {})

# 3. Model Initialization
m = gp.Model("optimization_model")

# 4. Sets & Variables
# Use sets['ID'] and map Decision Variables from JSON.

# 5. Objective & Constraints
# Use params['ID'] for all coefficients. 
# NO RAW NUMBERS should be used in m.setObjective() or m.addConstr().

# 6. Optimization & Output
m.optimize()
if m.status == GRB.OPTIMAL:
    print(f"Optimal Objective Value: {m.objVal}")
    for v in m.getVars():
        if v.x > 1e-6:
            print(f"{v.varName}: {v.x}")
else:
    print("Optimization was not successful.")
"""
        
    def generate(self,formulation):
        
        #if reference_code:
            #self.output_format = reference_code
        
        full_prompt = f"""[Role] {self.role_description}
[Whole Interpretation] {formulation}
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
        print("model generation 요청중...")
        try:
            response = requests.post(self.url, json=payload)
            response.raise_for_status()
            
            return response.json().get('response', '').strip()       
        
        except Exception as e:
            return f"model generation 실패: {e}"