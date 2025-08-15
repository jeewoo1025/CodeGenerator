# CODESIM 리뷰 (NAACL 2025 Findings)

## 📋 논문 컨텍스트

**CODESIM**은 NAACL 2025 Findings에서 제시된 혁신적인 multi-agent code generation 프레임워크입니다. 기존 MapCoder의 "multiple ungrounded exemplars" 접근법을 개선하여 **"single exemplar" 기반의 simulation-driven planning and debugging**을 구현했습니다.

### 🚀 핵심 혁신
- **3-Agent Architecture**: Planning Agent, Coding Agent, Debugging Agent의 협력적 구조
- **Simulation-Driven Verification**: Step-by-step 시뮬레이션을 통한 계획 검증
- **Internal Debugging**: 외부 도구 없이 시뮬레이션 기반 내부 디버깅
- **Human-like Perception**: 인간의 알고리즘 시각적 검증 방식 구현

### 📊 SOTA 성능 달성
- **HumanEval**: 95.1% (Pass@1)
- **MBPP**: 90.7% (Pass@1)  
- **APPS**: 22% (Pass@1)
- **CodeContests**: 29.1% (Pass@1)

## 🏗️ 아키텍처 다이어그램

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│  Planning Agent │───▶│  Coding Agent   │───▶│ Debugging Agent │
│                 │    │                 │    │                 │
│ • Problem       │    │ • Plan → Code   │    │ • Internal      │
│   Understanding │    │ • Execution     │    │   Simulation    │
│ • Exemplar      │    │ • Generation    │    │ • Bug Detection │
│   Recall        │    │                 │    │ • Code Fix      │
│ • Algorithm     │    │                 │    │                 │
│   Design        │    │                 │    │                 │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│ Plan Simulation │    │ Code Execution  │    │ Test Validation │
│ & Verification  │    │ & Evaluation    │    │ & Refinement    │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## 🔍 핵심 구현 분석

### 1. 3-Agent 구조 구현

#### Planning Agent (`CodeSIM.py` lines 200-280)
```python
# 핵심 계획 생성 로직
input_for_planning = [
    {
        "role": "user", 
        "content": prompt_for_planning.format(
            problem=problem,
            language=self.language,
        )
    },
]

# 계획 구조화
if "### Plan" not in response:
    plan = f"### Plan\n\n{response}"
else:
    plan = response[response.rfind("### Plan"):]
```

**구현 특징:**
- **Problem Understanding**: 문제 유형 및 제약사항 분석
- **Exemplar Recall**: 관련 예제 문제 회상 및 알고리즘 분석
- **Algorithm Design**: 최적 알고리즘 선택 및 튜토리얼 제공
- **Step-by-step Planning**: 상세한 실행 계획 수립

#### Coding Agent (`CodeSIM.py` lines 320-350)
```python
# 계획 기반 코드 생성
input_for_final_code_generation = [
    {
        "role": "user",
        "content": prompt_for_code_generation.format(
            problem_with_planning=problem_with_planning,
            language=self.language,
            std_input_prompt=std_input_prompt,
        )
    }
]

code = parse_response(response)
```

**구현 특징:**
- **Plan-to-Code Translation**: 검증된 계획을 실행 가능한 코드로 변환
- **Language-Specific Generation**: 프로그래밍 언어별 최적화된 코드 생성
- **Standard I/O Handling**: 경쟁 프로그래밍 환경에 최적화된 입출력 처리

#### Debugging Agent (`CodeSIM.py` lines 360-420)
```python
# 내부 디버깅 메커니즘
for debug_no in range(1, self.max_debug_try + 1):
    input_for_debugging = [
        {
            "role": "user",
            "content": prompt_for_debugging.format(
                problem_with_planning=problem_with_planning,
                code=code,
                language=self.language,
                test_log=test_log,
                std_input_prompt=std_input_prompt,
            )
        }
    ]
    
    code = parse_response(response)
    passed, test_log = self.check(data_row, additional_io, code)
    
    if passed:
        break
```

**구현 특징:**
- **Internal Simulation**: 외부 도구 없이 시뮬레이션 기반 버그 탐지
- **Step-by-step Analysis**: 실패한 테스트 케이스의 단계별 분석
- **Plan-Code Alignment**: 계획과 코드 간 불일치 점 검출
- **Iterative Refinement**: 최대 5회까지 반복적 코드 개선

### 2. Simulation-Driven 접근법 구현

#### Plan Verification (`CodeSIM.py` lines 290-320)
```python
# 계획 시뮬레이션 및 검증
input_for_simulation = [
    {
        "role": "user",
        "content": prompt_for_simulation.format(
            problem_with_planning=problem_with_planning,
            language=self.language,
        )
    },
]

# 계획 수정 필요성 판단
if "Plan Modification Needed" in response and \
    "No Plan Modification Needed" not in response:
    
    # 계획 정제 단계
    input_for_plan_refinement = [
        {
            "role": "user",
            "content": prompt_for_plan_refinement.format(
                problem_with_planning=problem_with_planning,
                language=self.language,
                critique=response
            )
        },
    ]
```

**구현 특징:**
- **Manual Simulation**: 코드 없이 수동으로 계획 단계별 실행
- **Output Comparison**: 예상 출력과 실제 출력 비교 검증
- **Plan Critique**: 시뮬레이션 결과 기반 계획 비판적 분석
- **Iterative Refinement**: 최대 5회까지 계획 개선 반복

#### Internal Debugging Simulation
```python
# 디버깅 프롬프트의 시뮬레이션 지시사항
prompt_for_debugging = """
### Simulation with failed test case
To detect where is the bug follow following steps:
    - Take a sample test case where it fails.
    - Take the input go through each step according to the plan
    - You will get a output that must be different from the expected output.

### Debugging Notes
- Based on this simulation detect any of the following cases:
    - Plan is wrong
    - Plan is correct but plan to code generation is wrong.
- Finally, discuss how to correct this code.
"""
```

**구현 특징:**
- **Step-by-step Execution**: 실패한 테스트 케이스의 단계별 실행 시뮬레이션
- **Plan-Code Mismatch Detection**: 계획과 코드 간 불일치 점 식별
- **Root Cause Analysis**: 버그의 근본 원인 분석
- **Corrective Action Planning**: 수정 방안 수립

### 3. Multi-Agent 간 통신 및 데이터 흐름

#### Agent 간 데이터 전달 구조
```python
# Planning → Coding → Debugging 데이터 흐름
problem_with_planning = f"## Problem:\n{problem}\n\n{plan}"

# 각 단계에서 이전 단계의 결과를 입력으로 활용
input_for_final_code_generation = [
    {
        "role": "user",
        "content": prompt_for_code_generation.format(
            problem_with_planning=problem_with_planning,  # 계획 포함
            language=self.language,
            std_input_prompt=std_input_prompt,
        )
    }
]

input_for_debugging = [
    {
        "role": "user", 
        "content": prompt_for_debugging.format(
            problem_with_planning=problem_with_planning,  # 계획 + 코드
            code=code,
            language=self.language,
            test_log=test_log,
            std_input_prompt=std_input_prompt,
        )
    }
]
```

#### 반복적 개선 메커니즘
```python
# Planning 반복 (최대 5회)
for plan_no in range(1, self.max_plan_try + 1):
    # ... planning logic ...
    if passed:
        break

# Debugging 반복 (최대 5회)  
for debug_no in range(1, self.max_debug_try + 1):
    # ... debugging logic ...
    if passed:
        break
```

## 🧪 벤치마크 및 평가 시스템

### ExecEval 연동 구현
```python
# 코드 실행 및 평가
def check(self, data_row: dict, additional_io: List[str], code: str) -> bool:
    passed_sample, test_log_sample = self.data.evaluate_sample_io(
        data_row, code, self.language
    )
    
    passed_additional, test_log_additional = self.data.evaluate_additional_io(
        data_row[self.data.id_key], additional_io, code, self.language
    )
    
    return passed_sample & passed_additional, test_log
```

### Pass@1 평가 메트릭
```python
# 결과 집계 및 요약
gen_summary(RESULTS_PATH, SUMMARY_PATH)

# ET/EP 데이터셋 생성
if "human" in DATASET.lower():
    generate_et_dataset_human(RESULTS_PATH, ET_RESULTS_PATH)
    gen_summary(ET_RESULTS_PATH, ET_SUMMARY_PATH)
```

## 🔧 실행 가이드

### 기본 실행 명령어
```bash
# HumanEval 데이터셋으로 CodeSIM 실행
python src/main.py --dataset HumanEval --strategy CodeSIM --model ChatGPT

# MBPP 데이터셋으로 CodeSIM 실행  
python src/main.py --dataset MBPP --strategy CodeSIM --model ChatGPT

# APPS 데이터셋으로 CodeSIM 실행
python src/main.py --dataset APPS --strategy CodeSIM --model ChatGPT

# LiveCodeBench 데이터셋으로 CodeSIM 실행
python src/main.py --dataset LiveCodeBench --strategy CodeSIM --model ChatGPT
```

### 고급 실행 옵션
```bash
# 계획 시도 횟수 및 디버깅 시도 횟수 조정
python src/main.py \
    --dataset HumanEval \
    --strategy CodeSIM \
    --model ChatGPT \
    --temperature 0 \
    --top_p 0.95 \
    --pass_at_k 1
```

### 모델별 실행
```bash
# Gemini 모델 사용
python src/main.py --dataset HumanEval --strategy CodeSIM --model gemini-pro --model_provider Gemini

# Groq 모델 사용
python src/main.py --dataset HumanEval --strategy CodeSIM --model mixtral-8x7b-32768 --model_provider Groq

# Anthropic 모델 사용
python src/main.py --dataset HumanEval --strategy CodeSIM --model claude-3-sonnet-20240229 --model_provider Anthropic
```

## 📊 코드 구조 상세 분석

### 핵심 클래스 구조
```
src/
├── promptings/
│   ├── CodeSIM.py              # 메인 CodeSIM 구현
│   ├── variations/
│   │   ├── CodeSIMWPVD.py     # With Plan Verification & Debugging
│   │   ├── CodeSIMWD.py        # With Debugging
│   │   ├── CodeSIMWPV.py       # With Plan Verification
│   │   ├── CodeSIMA.py         # Analogical variation
│   │   └── CodeSIMC.py         # Competitive programming
│   ├── Base.py                 # 기본 전략 클래스
│   └── PromptingFactory.py     # 전략 팩토리
├── models/
│   ├── ModelFactory.py         # 모델 팩토리
│   ├── OpenAI.py               # OpenAI 모델 구현
│   ├── Gemini.py               # Gemini 모델 구현
│   └── Anthropic.py            # Anthropic 모델 구현
├── datasets/
│   ├── DatasetFactory.py       # 데이터셋 팩토리
│   ├── HumanEvalDataset.py     # HumanEval 데이터셋
│   ├── MBPPDataset.py          # MBPP 데이터셋
│   ├── APPSDataset.py          # APPS 데이터셋
│   └── LiveCodeBenchDataset.py # LiveCodeBench 데이터셋
└── evaluations/
    ├── func_evaluate.py         # 함수 평가 엔진
    └── executor_utils.py        # 실행 유틸리티
```

### 주요 함수 분석

#### 1. `run_single_pass()` - 메인 실행 로직
```python
def run_single_pass(self, data_row: dict):
    # 1. 문제 분석 및 추가 I/O 수집
    problem = self.data.get_prompt(data_row)
    additional_io = []
    
    # 2. Planning Phase (최대 5회)
    for plan_no in range(1, self.max_plan_try + 1):
        # 계획 생성 → 시뮬레이션 검증 → 계획 정제
        # 코드 생성 → 테스트 실행
        
        # 3. Debugging Phase (최대 5회)
        for debug_no in range(1, self.max_debug_try + 1):
            # 내부 시뮬레이션 → 버그 탐지 → 코드 수정
            # 테스트 재실행
            
        if passed:
            break
```

#### 2. `check()` - 코드 검증 로직
```python
def check(self, data_row: dict, additional_io: List[str], code: str) -> bool:
    # 샘플 I/O 평가
    passed_sample, test_log_sample = self.data.evaluate_sample_io(
        data_row, code, self.language
    )
    
    # 추가 I/O 평가  
    passed_additional, test_log_additional = self.data.evaluate_additional_io(
        data_row[self.data.id_key], additional_io, code, self.language
    )
    
    # 통합 결과 반환
    return passed_sample & passed_additional, test_log
```

## 🚀 성능 최적화 구현

### 1. 경쟁 프로그래밍 최적화
```python
# APPS, CodeContest, XCode 데이터셋 최적화
self.is_competitive = type(self.data) == APPSDataset or \
    type(self.data) == CodeContestDataset or \
    type(self.data) == XCodeDataset

if self.is_competitive:
    std_input_prompt = """
    - Strictly follow the sample input and output format. 
    - The input should be taken from Standard input and output should be given to standard output.
    - For array input parse the array then pass it to the function.
    - Do not add extra print statement otherwise it will failed the test cases.
    """
```

### 2. LiveCodeBench 전용 최적화
```python
# LiveCodeBench 데이터셋 감지
def is_livecodebench(self) -> bool:
    return self.dataset_type == 'livecodebench'

# LiveCodeBench 전용 계획 프롬프트
if self.is_livecodebench():
    input_for_planning = [
        {
            "role": "user",
            "content": f"""You are a competitive programming expert tasked with generating an appropriate plan to solve a given LiveCodeBench problem using the **{self.language}** programming language.
            
            ## Problem
            {problem}
            
            **Expected Output:**
            Your response must be structured as follows:
            
            ### Problem Understanding
            - Think about the original problem. Develop an initial understanding about the problem.
            - Identify the problem type (array, string, graph, dynamic programming, etc.)
            - Note any constraints or edge cases
            
            ### Recall Example Problem
            Recall a relevant and distinct competitive programming problem (different from problem mentioned above) and
            - Describe it briefly
            - Identify the algorithm category (greedy, DP, graph, etc.)
            - Generate {self.language} code step by step to solve that problem
            - Discuss the algorithm to solve this problem
            - Finally generate a planning to solve that problem
            
            ### Algorithm to solve the original problem
            - Write down the algorithm that is well suited for the original problem
            - Give some tutorials about the algorithm for example:
                - How to approach this type of algorithm
                - Important things to consider
                - Time and space complexity analysis
            
            ### Plan
            - Write down a detailed, step-by-step plan to solve the **original problem**.
            - Include edge case handling
            - Consider optimization strategies
            
            --------
            **Important Instruction:**
            - Strictly follow the instructions.
            - Do not generate code.
            - Focus on competitive programming best practices."""
        },
    ]
```

## 🔄 MapCoder와의 차이점

### 1. Exemplar 접근법 차이
- **MapCoder**: "multiple ungrounded exemplars" 사용
- **CodeSIM**: "single exemplar" 기반 계획 생성

### 2. 계획 검증 단계 추가
```python
# CodeSIM의 계획 검증 단계 (MapCoder에는 없음)
input_for_simulation = [
    {
        "role": "user",
        "content": prompt_for_simulation.format(
            problem_with_planning=problem_with_planning,
            language=self.language,
        )
    },
]

# 시뮬레이션 결과에 따른 계획 수정
if "Plan Modification Needed" in response:
    input_for_plan_refinement = [
        {
            "role": "user",
            "content": prompt_for_plan_refinement.format(
                problem_with_planning=problem_with_planning,
                language=self.language,
                critique=response
            )
        },
    ]
```

### 3. 내부 디버깅 메커니즘
```python
# CodeSIM의 시뮬레이션 기반 내부 디버깅
prompt_for_debugging = """
### Simulation with failed test case
To detect where is the bug follow following steps:
    - Take a sample test case where it fails.
    - Take the input go through each step according to the plan
    - You will get a output that must be different from the expected output.

### Debugging Notes
- Based on this simulation detect any of the following cases:
    - Plan is wrong
    - Plan is correct but plan to code generation is wrong.
- Finally, discuss how to correct this code.
"""
```

## 🌟 확장성 및 모델 연동

### 1. 모델 팩토리 패턴
```python
class ModelFactory:
    @staticmethod
    def get_model_class(model_provider_name: str):
        model_provider_name = model_provider_name.lower()
        if model_provider_name == "gemini":
            return Gemini
        elif model_provider_name == "openai":
            return OpenAIV1Model
        elif model_provider_name == "openai-v2":
            return OpenAIV2Model
        elif model_provider_name == "groq":
            return GroqModel
        elif model_provider_name == "anthropic":
            return AnthropicModel
```

### 2. 전략 팩토리 패턴
```python
class PromptingFactory:
    @staticmethod
    def get_prompting_class(strategy_name: str):
        if strategy_name == "CodeSIM":
            return CodeSIM
        elif strategy_name == "CodeSIMWPVD":
            return CodeSIMWPVD
        elif strategy_name == "CodeSIMWD":
            return CodeSIMWD
        elif strategy_name == "CodeSIMWPV":
            return CodeSIMWPV
        elif strategy_name == "CodeSIMA":
            return CodeSIMA
        elif strategy_name == "CodeSIMC":
            return CodeSIMC
```

### 3. 데이터셋 팩토리 패턴
```python
class DatasetFactory:
    @staticmethod
    def create_dataset(dataset_name, **kwargs):
        dataset_class = DatasetFactory.get_dataset_class(dataset_name)
        
        if dataset_name.lower() in ["livecodebench", "lcb"] or dataset_name.startswith("lcb_"):
            if dataset_name.startswith("lcb_"):
                version = dataset_name.replace("lcb_", "")
            else:
                version = kwargs.get('release_version', 'release_v6')
            return dataset_class(release_version=version)
        else:
            return dataset_class(**kwargs)
```

## 📈 결과 분석 및 해석

### 1. 실행 결과 구조
```
results/
└── {DATASET}/
    └── {STRATEGY}/
        └── {MODEL_NAME}/
            └── {LANGUAGE}-{TEMPERATURE}-{TOP_P}-{PASS_AT_K}/
                ├── Run-{run_no}/
                │   ├── Results.jsonl          # 기본 결과
                │   ├── Summary.txt            # 요약 통계
                │   ├── Log.txt                # 실행 로그
                │   ├── Results-ET.jsonl       # ET 평가 결과
                │   ├── Summary-ET.txt         # ET 요약
                │   ├── Results-EP.jsonl       # EP 평가 결과
                │   ├── Summary-EP.txt         # EP 요약
                │   ├── Results-LCB.jsonl      # LiveCodeBench 결과
                │   ├── Summary-LCB.txt        # LiveCodeBench 요약
                │   └── Report-LCB.json        # LiveCodeBench 상세 리포트
```

### 2. 성능 지표 해석
- **Pass@1**: 첫 번째 시도에서 통과한 문제 비율
- **ET (Execution Time)**: 코드 실행 시간 분석
- **EP (Execution Pass)**: 실행 통과율 분석
- **LCB (LiveCodeBench)**: 경쟁 프로그래밍 특화 평가

## 🔮 향후 발전 방향

### 1. 모델 확장
- **o3-mini**: Ollama 기반 로컬 모델 연동
- **GPT-4o**: 최신 OpenAI 모델 지원
- **Claude 3.5 Sonnet**: Anthropic 최신 모델 지원

### 2. 전략 확장
- **CodeSIM+**: 강화학습 기반 에이전트 협력 최적화
- **CodeSIM-Multi**: 다중 언어 동시 생성 지원
- **CodeSIM-Adaptive**: 문제 유형별 자동 전략 선택

### 3. 평가 시스템 확장
- **Code Quality Metrics**: 코드 품질 지표 추가
- **Runtime Performance**: 실행 시간 성능 분석
- **Memory Usage**: 메모리 사용량 분석

## 📚 참고 자료

- **논문**: CODESIM: Multi-Agent Code Generation with Simulation-Driven Planning and Debugging (NAACL 2025 Findings)
- **코드베이스**: [GitHub Repository](https://github.com/your-repo/codesim)
- **데이터셋**: HumanEval, MBPP, APPS, CodeContests, LiveCodeBench
- **평가 프레임워크**: ExecEval, Pass@k metrics

---

**Note**: 이 README는 CODESIM 프레임워크의 실제 구현 코드를 기반으로 작성되었으며, 논문에서 제시한 이론적 개념들이 어떻게 실제로 구현되었는지 상세히 분석하고 있습니다.
