import os
import torch
import requests
from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer

os.environ["TORCH_CUDA_ARCH_LIST"] = "12.1"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

class InstanceDataSetGenerator():
    def __init__(self, model_name, url):
        self.url = url
        self.model_name = model_name
        self.role_description = 'Expert Data Extractor for Mathematical Optimization.'
        
        self.task = """
1. Code Syntax Analysis: Analyze the 'Coder Output' to detect the exact indexing method for all 2D parameters (e.g., matrix[i, j] vs matrix[i][j]). [cite: 1, 7]
2. Key Discovery: Identify every unique key accessed under the `parameters` and `sets` dictionaries within the Python script. 
3. Numerical Extraction: Retrieve exact constants, coefficients, and set members from the 'Original Problem' description. [cite: 17, 36]
4. Type Alignment: Format the extracted data into JSON types (Scalar, List, or Map) that perfectly align with the loading logic in the 'Coder Output'. 
"""

        self.rules = """[UNIVERSAL DATA EXTRACTION RULES]
1. MANDATORY INDEXING SYNC: Your JSON structure must be a perfect 'key' to the code's 'lock'.
   - IF code uses 'var[i, j]', use a dict with stringified tuple keys: {"(0,1)": 10.0}. [cite: 1, 6]
   - IF code uses 'var[i][j]', use a nested list: [[0, 10], [10, 0]]. 
   - IF code uses 'var[i]', use a standard list: [10, 20]. 
2. KEY CONSISTENCY: JSON keys must be identical to the string literals used in the Python script's .get() or [] accessors. 
3. STRICT HIERARCHY: Wrap all data under exactly two top-level keys: "parameters" and "sets". [cite: 1, 13, 24]
4. RELATIONSHIP REPRESENTATION: Represent all pairs (precedence, edges, or mappings) as a list of lists: [[item1, item2], [item3, item4]]. [cite: 23, 36]
5. AUTOMATIC BIG-M: If the code requires a large constant 'M' not specified in the text, assign a value like 10000. [cite: 5, 13]
6. NO PROSE: Return ONLY the raw JSON object. Do not include markdown backticks (```json) or any explanation. 
"""

        self.output_format = """
{
    "parameters": {
        "scalar_var": 100.0,
        "matrix_var": { "(0,1)": 15.5 }, // Match this style to the code's specific indexing syntax
        "pair_list_var": [[1, 3], [2, 4]]
    },
    "sets": {
        "set_var": [0, 1, 2, 3, 4]
    }
}
"""
        
    
    def extract_instances_first(self, problem, coder_output):
        full_prompt = f"""[Role] {self.role_description}
[Original Problem] {problem}
[Coder Output] {coder_output}
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
        print("instance 추출중...")
        try:
            response = requests.post(self.url, json=payload)
            response.raise_for_status()
            
            return response.json().get('response', '').strip()       
        
        except Exception as e:
            return f"model generation 실패: {e}"
    
    
    def extract_instances_second(self,problem,code_path):
        coder_output = ""
        try:
            with open(code_path,'r',encoding='utf-8') as f:
                coder_output = f.read()
            
        except FileNotFoundError:
            print("코드 파일을 찾을 수 없습니다. 인스턴스 추출 실패.")
            return
        
        print("파일 읽기 성공! 인스턴스 추출중....")
        
        full_prompt = f"""[Role] {self.role_description}
[Original Problem] {problem}
[Coder Output] {coder_output}
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
        print("instance 추출중...")
        try:
            response = requests.post(self.url, json=payload)
            response.raise_for_status()
            
            return response.json().get('response', '').strip()       
        
        except Exception as e:
            return f"model generation 실패: {e}"
        
