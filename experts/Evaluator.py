import os
import torch
import requests
import re
from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer

# 1. Blackwell(sm_121) 하드웨어 호환성 강제 설정
os.environ["TORCH_CUDA_ARCH_LIST"] = "12.1"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    
class Evaluator():
    def __init__(self,model_name,url,threshold):
        self.url = url
        self.model_name = model_name
        self.threshold = threshold
        self.role_description = 'Mathematical Optimization Auditor (Triple-Verification)'
        self.task = 'Cross-verify (1) Problem Description, (2) Extracted Queries, and (3) LaTeX Formulations.'
        
        self.rules = """[AUDIT PRIORITIES & SYMBOLIC CHECKS]
1. PARAMETER MAPPING (CRITICAL): Does every number in the Problem Description have a corresponding Parameter 'id'? 
   - LaTeX formulations MUST use these 'id's, not the raw numbers.
   - Example: If the problem says "$120", is there a parameter 'cost_A' and is it used as 'cost_A' in the LaTeX?

2. SYMBOLIC UNIT CONSISTENCY: Ensure the logic $Parameter \times Variable$ is dimensionally sound.
   - Error Example: $x_A \ge min\_tables$ (Wrong if $x$ is orders and $min\_tables$ is a count of tables).
   - Correct Example: $units\_per\_order_A \cdot x_A \ge min\_tables$.

3. THE SYMBOLIC BRIDGE: Check linking logic $x \le M \cdot y$. 
   - 'M' must be defined as a Parameter 'id', not as a hardcoded 10000.

4. OBJECTIVE PURITY: Verify if the objective sense (min/max) and the summation of Symbolic Parameters match the goal.

5. NO HARDCODING: Any raw number (except 0, 1 for binary or simple coefficients) found in the "Latex Model" is a CRITICAL FAILURE.

[AUDIT PROCESS - STEP-BY-STEP]
- Step 1: List all numbers in the Problem. Cross-check if each has an assigned Parameter ID in the JSON.
- Step 2: Verify that "Latex Model" fields contain ONLY Parameter IDs and Variable IDs.
- Step 3: Check every 'if-then' sentence. Verify if the logic $y_B \ge y_A$ or $x_A \le M \cdot y_A$ is used with symbolic IDs."""
        
        self.output_format = """
{
    "CONFIDENCE_SCORE": 0.0,
    "CRITICAL_FAILURE_FOUND": "YES/NO",
    "HARDCODING_CHECK": {
        "found_raw_numbers_in_latex": ["List any numbers like 120, 150 found in LaTeX"],
        "status": "PASS/FAIL"
    },
    "SYMBOLIC_FIDELITY": {
        "missing_parameters": ["List numbers from text that don't have a Parameter ID"],
        "unused_parameters": ["List IDs defined but not used in any formula"]
    },
    "LOGICAL_LINKING_AUDIT": "Verify if Big-M and Binary logic use consistent symbolic IDs.",
    "CONSTRAINTS_FEEDBACK": [
        { "id": "C1", "status": "PASS/FAIL", "reason": "Detailed critique of symbolic logic" }
    ],
    "FINAL_VERDICT": "Instructions for Coder/InstanceGenerator to fix the logic or mapping."
}
"""
        
    def evaluate(self, problem_description, whole_interpretation):
        problem = problem_description
        full_prompt = f"""[Role] {self.role_description}
[Problem] {problem}
[Whole Interpretation] {whole_interpretation}
[Task] {self.task}
[Threshold] {self.threshold}
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
        print("model evaluation 요청중...") 
        try:
            response = requests.post(self.url, json=payload)
            response.raise_for_status()
            print("model evaluation 완료!")
            return response.json().get('response', '').strip()       
        
        except Exception as e:
            return f"model evaluation 실패: {e}"
        
    def _extract_json(self,text):
        match = re.search(r'\{.*\}', text, re.DOTALL)
        if match:
            return match.group(0)
        return text # 매칭 실패 시 원본 반환