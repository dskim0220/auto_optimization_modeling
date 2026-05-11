# LLM 기반 최적화 모델 생성 멀티 에이전트 아키텍처 구현

>**기존 멀티 에이전트 모델(Chain-of-Experts)의 비효율적인 에이전트 반복 호출 구조를 개선하고, 컨피던스(Confidence) 평가 로직을 도입하여 파라미터를 증가시키지 않고, 제약조건 정답률을 2%에서 19%로 향상시킨 End-to-End 아키텍처 구현 프로젝트입니다.

### Tech Stack
**Language:** Python

**LLM:** Qwen2.5-3B-Instruct

**Optimization Solver:** Gurobi

## 1. e2e 모델 구조(E2E model architecture)
  <img src="./image/e2e_architecture.png" width="700">

## 2. 에이전트(Agents)
 * **BasicModelInterpreter:** 자연어 문제 쿼리에서 인스턴스, 목적함수, 제약조건 등 추출
  
 * **ConstraintsInterpreter:** BasicModelInterpreter에서 받은 정보를 기반으로 수리적 논리 구조를 구축하고, 이를 LaTeX 수식으로 변환
  
 * **Evaluator:** 이전 에이전트가 생성한 수식을 검증 및 피드백 제공
  
 * **Coder:** 완성된 모델링을 해결하는 Solver 코드 생성
  
 * **InstanceDataSetGenerator:** 문제 쿼리에서 인스턴스 정보를 추출

## 3. e2e 모델 수도코드(pseudo code)
  <img src="./image/e2e_pseudocode.png" width="700">
 * `Extractor`부터 `CodeGenerator`까지 이어지는 파이프라인과, `Confidence Threshold(임계값)`에 따른 조기 종료(Early termination) 및 피드백 루프 로직을 구현했습니다.

## 4. 실험 환경(Experiment setting)
  * **Model:** Qwen2.5-3B-Instruct(Local environment)
  
  * **Test Dataset:** newset

  * **Threshold:** 0.8

  * **Max trial:** 3

## 5. 성능 비교(Performance Benchmark)
  <img src="./image/benchmark_result.png" width="700">
  * 평가 결과, 제안하는 E2E 아키텍처가 기존 CoE 모델과 비교하여 높은 제약조건 도출 정확도(19%)를 기록하며 시스템 구조 개선의 유효성을 입증했습니다.
