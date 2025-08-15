# CodeGenerator

CodeGenerator는 다양한 대규모 언어 모델(LLM)을 사용하여 코드 생성 성능을 평가하는 프레임워크입니다. OpenAI, Anthropic, Google, vLLM 등 다양한 모델을 지원하며, Direct, CoT, CodeSIM, MapCoder 등 다양한 프롬프팅 전략을 통해 코드 생성 성능을 평가할 수 있습니다.

## 🚀 주요 기능

- **다양한 모델 지원**: OpenAI, Anthropic, Google, vLLM 등
- **Qwen3 모델 지원**: Qwen3.5, Qwen3, Qwen3-Coder 계열 모델 (총 43개 모델)
- **다양한 전략**: Direct, CoT, CodeSIM, MapCoder, SelfPlanning, Analogical 등
- **다양한 데이터셋**: HumanEval, MBPP, LiveCodeBench, APPS, xCodeEval 등
- **실시간 평가**: 코드 실행 및 테스트 자동화
- **크로스 플랫폼**: Windows, Linux, macOS 지원

## 📦 설치

### 1. 기본 의존성 설치

```bash
# 의존성 설치
pip install -r requirements.txt
```

### 2. vLLM 설치 확인

vLLM이 제대로 설치되었는지 확인:

```bash
python -c "import vllm; print('vLLM 설치 완료')"
```

### 3. CUDA 설정 확인

```bash
# CUDA 버전 확인
nvidia-smi
nvcc --version

# PyTorch CUDA 지원 확인
python -c "import torch; print(f'CUDA 사용 가능: {torch.cuda.is_available()}')"
```

## 🎯 사용법

### 1. Qwen3 모델 평가 (vLLM) - 권장

#### 기본 사용법
```bash
# Direct 전략으로 Qwen3-0.6B 평가
python run_qwen_evaluation.py \
    --model Qwen3-0.6B \
    --dataset HumanEval \
    --strategy Direct
```

#### CodeSIM 전략 사용
```bash
# CodeSIM 전략으로 평가 (코드 전용 모델 권장)
python run_qwen_evaluation.py \
    --model Qwen3-0.6B \
    --dataset HumanEval \
    --strategy CodeSIM \
    --max_plan_try 5 \
    --max_debug_try 5 \
    --additional_info_run 0
```

#### 다양한 전략 사용
```bash
# MapCoder 전략
python run_qwen_evaluation.py \
    --model Qwen3-0.6B \
    --dataset HumanEval \
    --strategy MapCoder

    # CoT (Chain of Thought) 전략
    python run_qwen_evaluation.py \
        --model Qwen3-0.6B \
        --dataset HumanEval \
        --strategy CoT

    # SelfPlanning 전략
    python run_qwen_evaluation.py \
        --model Qwen3-0.6B \
        --dataset HumanEval \
        --strategy SelfPlanning
```

#### LiveCodeBench 데이터셋
```bash
# LiveCodeBench 데이터셋으로 평가
python run_qwen_evaluation.py \
    --model Qwen3-0.6B \
    --dataset LiveCodeBench \
    --strategy CodeSIM \
    --temperature 0.1 \
    --max_tokens 4096 \
    --tensor_parallel_size 2
```

### 2. 기존 main.py 사용

```bash
# vLLM으로 Qwen3-0.6B 평가
python src/main.py \
    --model Qwen3-0.6B \
    --model_provider vllm \
    --dataset HumanEval \
    --strategy Direct \
    --temperature 0 \
    --top_p 0.95

# CodeSIM 전략 사용
python src/main.py \
    --model Qwen3-0.6B \
    --model_provider vllm \
    --dataset HumanEval \
    --strategy CodeSIM \
    --temperature 0 \
    --top_p 0.95
```

### 3. 다른 모델들

```bash
# OpenAI GPT-4
python src/main.py \
    --model gpt-4 \
    --model_provider OpenAI \
    --dataset HumanEval \
    --strategy Direct

# Anthropic Claude-3-Sonnet
python src/main.py \
    --model claude-3-sonnet \
    --model_provider anthropic \
    --dataset HumanEval \
    --strategy Direct

# Google Gemini Pro
python src/main.py \
    --model gemini-pro \
    --model_provider Google \
    --dataset HumanEval \
    --strategy Direct
```

## 🏗️ 지원 모델

### Qwen3 계열 (vLLM)
아래 모델명은 실제 지원되는 모델명과 일치해야 하며, `src/constants/qwen_models.py` 기준입니다.

#### Qwen3 시리즈
- Qwen3-0.6B
- Qwen3-1.7B
- Qwen3-4B
- Qwen3-8B
- Qwen3-14B
- Qwen3-32B
- Qwen3-30B-A3B (MoE)
- Qwen3-235B-A22B (MoE)

#### Qwen3-Coder 시리즈
- Qwen3-Coder-30B-A3B-Instruct (MoE, 코드 특화)
- Qwen3-Coder-480B-A35B-Instruct (MoE, 코드 특화)

### 기타 모델
- **OpenAI**: GPT-3.5-turbo, GPT-4, GPT-4o, GPT-4o-mini 등
- **Anthropic**: Claude-3-Haiku, Claude-3-Sonnet, Claude-3-Opus 등
- **Google**: Gemini Pro, Gemini Flash, Gemini 1.5 Pro 등
- **Groq**: Llama3-8B, Llama3-70B, Mixtral-8x7B 등

## 🎭 지원 전략

### 지원 프롬프팅 전략
- **Direct**: 직접 코드 생성 (가장 빠르고 효율적)
- **CoT**: Chain of Thought (단계별 사고 과정)
- **SelfPlanning**: 자체 계획 수립 및 실행
- **CodeSIM**: 코드 시뮬레이션, 계획 수립, 디버깅 (가장 정확함)
- **MapCoder**: 맵핑 기반 코드 생성
- **Analogical**: 유사 사례 기반 생성

#### CodeSIM 변형 전략
- **CodeSIMWD**: CodeSIM with Debugging
- **CodeSIMWPV**: CodeSIM with Planning and Validation
- **CodeSIMWPVD**: CodeSIM with Planning, Validation and Debugging
- **CodeSIMA**: CodeSIM Advanced
- **CodeSIMC**: CodeSIM Compact

## 📊 지원 데이터셋


### 1. Qwen3 모델 평가 (vLLM) - 권장

#### 기본 사용법
```bash
# Direct 전략으로 Qwen3-0.6B 평가
python run_qwen_evaluation.py \
    --model Qwen3-0.6B \
    --dataset HumanEval \
    --strategy Direct
```

#### CodeSIM 전략 사용
```bash
# CodeSIM 전략으로 평가 
python run_qwen_evaluation.py \
    --model Qwen3-0.6B \
    --dataset HumanEval \
    --strategy CodeSIM \
    --max_plan_try 5 \
    --max_debug_try 5 \
    --additional_info_run 0
```

#### 다양한 전략 사용
```bash
# MapCoder 전략
python run_qwen_evaluation.py \
    --model Qwen3-0.6B \
    --dataset HumanEval \
    --strategy MapCoder

# CoT (Chain of Thought) 전략
python run_qwen_evaluation.py \
    --model Qwen3-0.6B \
    --dataset HumanEval \
    --strategy CoT

# SelfPlanning 전략
python run_qwen_evaluation.py \
    --model Qwen3-0.6B \
    --dataset HumanEval \
    --strategy SelfPlanning
```
│   │   ├── OpenAI.py            # OpenAI 모델
│   │   ├── Anthropic.py         # Anthropic 모델
│   │   ├── VLLMModel.py         # vLLM 모델 (Qwen3 지원)
│   │   ├── Gemini.py            # Google Gemini 모델
│   │   └── ModelFactory.py      # 모델 팩토리
│   ├── promptings/               # 프롬프팅 전략
│   │   ├── Base.py              # 기본 전략 클래스
│   │   ├── Direct.py            # Direct 전략
│   │   ├── CodeSIM.py           # CodeSIM 전략
│   │   ├── MapCoder.py          # MapCoder 전략
│   │   └── PromptingFactory.py  # 전략 팩토리
│   ├── datasets/                 # 데이터셋 로더
│   │   ├── HumanEvalDataset.py  # HumanEval 데이터셋
│   │   ├── LiveCodeBenchDataset.py # LiveCodeBench 데이터셋
│   │   └── DatasetFactory.py    # 데이터셋 팩토리
│   ├── evaluations/              # 평가 로직
│   │   ├── func_evaluate.py     # 함수 평가
│   │   └── resource_limit.py    # 리소스 제한
│   ├── constants/                # 상수 정의
│   │   ├── qwen_models.py       # Qwen3 모델 설정
│   │   └── paths.py             # 경로 상수
│   ├── utils/                    # 유틸리티 함수
│   │   ├── summary.py           # 결과 요약
│   │   └── parse.py             # 파싱 유틸리티
│   ├── results/                  # 결과 처리
│   │   └── Results.py           # 결과 클래스
│   └── main.py                  # 메인 실행 스크립트
├── data/                         # 데이터셋 파일들
│   ├── HumanEval/               # HumanEval 데이터
│   ├── MBPP/                    # MBPP 데이터
│   ├── LiveCodeBench/           # LiveCodeBench 데이터
│   └── APPS/                    # APPS 데이터
├── results/                      # 평가 결과 (자동 생성)
├── run_qwen_evaluation.py       # Qwen3 모델 평가 통합 스크립트
├── test_setup.py                # 설정 테스트 스크립트
├── requirements.txt              # Python 의존성
└── README.md                     # 이 파일
```

## 🚀 빠른 시작

### 1. 설치 및 설정

```bash
# 저장소 클론
git clone https://github.com/your-username/CodeGenerator.git
cd CodeGenerator

# 의존성 설치
pip install -r requirements.txt

# 설정 테스트
python test_setup.py
```

### 2. 첫 번째 평가 실행

```bash
# Qwen3-0.6B로 HumanEval 평가
python run_qwen_evaluation.py \
    --model Qwen3-0.6B \
    --dataset HumanEval \
    --strategy Direct
```

### 3. 결과 확인

```bash
# 결과 디렉토리 확인
ls results/

# 요약 파일 확인
cat results/Qwen3-0.6B_HumanEval_Direct_*/Summary.txt
```

## 🎯 고급 사용법

### 배치 평가

여러 모델을 순차적으로 평가:

```bash
# 여러 모델 평가
for model in "Qwen3-8B" "Qwen3-14B" "Qwen3-32B"; do
    python run_qwen_evaluation.py \
        --model $model \
        --dataset HumanEval \
        --strategy Direct
done
```

### 다양한 전략 비교

```bash
# 같은 모델로 다른 전략 비교
for strategy in "Direct" "CoT" "CodeSIM" "MapCoder"; do
    python run_qwen_evaluation.py \
        --model Qwen3-Coder-7B \
        --dataset HumanEval \
        --strategy $strategy
done
```

### 성능 최적화

```bash
# GPU 메모리 최적화
python run_qwen_evaluation.py \
    --model Qwen3-Coder-7B \
    --dataset HumanEval \
    --strategy Direct \
    --gpu_memory_utilization 0.8 \
    --max_tokens 1024

# 다중 GPU 사용
python run_qwen_evaluation.py \
    --model Qwen3-Coder-14B \
    --dataset HumanEval \
    --strategy Direct \
    --tensor_parallel_size 2
```

## 🔍 문제 해결

### 일반적인 문제들

#### 1. GPU 메모리 부족
```bash
# GPU 메모리 사용률 조정
--gpu_memory_utilization 0.7

    # 더 작은 모델 사용
    --model Qwen3-0.6B

# 최대 토큰 수 줄이기
--max_tokens 1024
```

#### 2. 모델 다운로드 실패
```bash
# Hugging Face 토큰 설정
export HF_TOKEN=your_token_here

# 네트워크 타임아웃 증가
export HF_HUB_DOWNLOAD_TIMEOUT=1000
```

#### 3. vLLM 초기화 실패
```bash
# CUDA 버전 확인
nvidia-smi
nvcc --version

# vLLM 재설치
pip uninstall vllm
pip install vllm

# PyTorch 재설치
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

#### 4. CodeSIM 실행 오류
```bash
# 더 작은 배치 크기
--max_tokens 1024

# 메모리 사용량 줄이기
--gpu_memory_utilization 0.7
```

## 📊 결과 분석

### 결과 파일 구조

```
results/
└── Qwen_Qwen3-Coder-7B_HumanEval_Direct_20241201_143022/
    ├── Results.jsonl          # 상세 평가 결과
    ├── Summary.txt            # 결과 요약
    ├── Log.txt               # 실행 로그
    ├── Results-ET.jsonl      # Execution Time 결과
    └── Summary-ET.txt        # Execution Time 요약
```