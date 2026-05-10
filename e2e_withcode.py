import time
from experts.InstanceDataSetGenerator import InstanceDataSetGenerator
from code_runner import execute_code
from e2e import save_output
from utils import read_problem2

model_name = "qwen2.5:32b"
problem_name = "IndustryOR_21"
url = "http://localhost:11434/api/generate"

def nl2opt_withcode(problem_name, model_name, url):
    problem = read_problem2('newset', problem_name)
    
    instance_generator = InstanceDataSetGenerator(model_name=model_name, url=url)
    code_path = f"output/{problem_name}_gencode.py"
    
    print("인스턴스 생성 시작...")
    instances = instance_generator.extract_instances_second(problem=problem, code_path=code_path)
    save_output(instances, f"{problem_name}_instances_from_code", "json")
    print("인스턴스 생성 완료!")
    
    data_path = f"output/{problem_name}_instances_from_code.json"
    parsed_result = execute_code(code_path, data_path)
    print(f"실행 결과: {parsed_result}")
    

if __name__ == '__main__':
    start_time = time.time()
    nl2opt_withcode(problem_name, model_name, url)
    end_time = time.time()
    running_time = end_time - start_time
    print(f"총 소요시간: {running_time:.2f}s")
    