#!/usr/bin/env python3
"""
Qwen3 계열 모델들을 vLLM을 사용하여 평가하는 통합 스크립트입니다.
모든 프롬프팅 전략(Direct, CodeSIM, MapCoder 등)을 지원합니다.
"""

import argparse
import sys
import os
from datetime import datetime

# src 디렉토리를 Python 경로에 추가
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from constants.qwen_models import list_available_qwen_models, get_qwen_model_config, list_coder_models
from models.ModelFactory import ModelFactory
from promptings.PromptingFactory import PromptingFactory
from datasets.DatasetFactory import DatasetFactory
from results.Results import Results
from utils.summary import gen_summary
from utils.livecodebench_utils import evaluate_livecodebench_results, generate_livecodebench_report

def main():
    parser = argparse.ArgumentParser(description="Qwen3 모델들을 vLLM으로 평가 (모든 전략 지원)")
    
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=list_available_qwen_models(),
        help="사용할 Qwen 모델"
    )
    
    parser.add_argument(
        "--dataset",
        type=str,
        default="HumanEval",
        choices=[
            "HumanEval",
            "MBPP", 
            "APPS",
            "xCodeEval",
            "LiveCodeBench"
        ],
        help="평가할 데이터셋"
    )
    
    parser.add_argument(
        "--strategy",
        type=str,
        default="Direct",
        choices=[
            "Direct",
            "CoT",
            "SelfPlanning",
            "Analogical",
            "MapCoder",
            "CodeSIM",
            "CodeSIMWD",
            "CodeSIMWPV",
            "CodeSIMWPVD",
            "CodeSIMA",
            "CodeSIMC",
        ],
        help="사용할 프롬프팅 전략"
    )
    
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="생성 온도"
    )
    
    parser.add_argument(
        "--top_p",
        type=float,
        default=0.95,
        help="Top-p 샘플링"
    )
    
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=2048,
        help="최대 토큰 수"
    )
    
    parser.add_argument(
        "--tensor_parallel_size",
        type=int,
        default=1,
        help="텐서 병렬 크기 (GPU 수)"
    )
    
    parser.add_argument(
        "--gpu_memory_utilization",
        type=float,
        default=0.9,
        help="GPU 메모리 사용률"
    )
    
    parser.add_argument(
        "--language",
        type=str,
        default="python",
        help="프로그래밍 언어"
    )
    
    parser.add_argument(
        "--pass_at_k",
        type=int,
        default=1,
        help="Pass@k 값"
    )
    
    parser.add_argument(
        "--verbose",
        type=int,
        default=1,
        help="상세 출력 레벨"
    )
    
    # CodeSIM 특별 파라미터들
    parser.add_argument(
        "--max_plan_try",
        type=int,
        default=5,
        help="최대 계획 시도 횟수 (CodeSIM 전략용)"
    )
    
    parser.add_argument(
        "--max_debug_try",
        type=int,
        default=5,
        help="최대 디버깅 시도 횟수 (CodeSIM 전략용)"
    )
    
    parser.add_argument(
        "--additional_info_run",
        type=int,
        default=0,
        help="추가 정보 실행 횟수 (CodeSIM 전략용)"
    )
    
    args = parser.parse_args()
    
    # 모델 설정 가져오기
    model_config = get_qwen_model_config(args.model)
    if not model_config:
        print(f"알 수 없는 모델: {args.model}")
        sys.exit(1)
    
    # 전략별 권장사항
    if args.strategy.startswith("CodeSIM") and not model_config.get('is_coder', False):
        print(f"⚠️  경고: {args.model}은 일반 모델입니다.")
        print(f"   CodeSIM 전략에는 코드 전용 모델을 권장합니다:")
        coder_models = list_coder_models()
        print(f"   권장 모델: {', '.join(coder_models[:5])}")
        print()
    
    # 실험 이름 생성
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"Qwen_{args.model}_{args.dataset}_{args.strategy}_{timestamp}"
    
    # 결과 경로 설정
    results_path = f"results/{run_name}/Results.jsonl"
    summary_path = f"results/{run_name}/Summary.txt"
    logs_path = f"results/{run_name}/logs.txt"
    
    # 디렉토리 생성
    os.makedirs(f"results/{run_name}", exist_ok=True)
    
    print(f"=== Qwen3 모델 평가 시작 ===")
    print(f"모델: {args.model}")
    print(f"데이터셋: {args.dataset}")
    print(f"전략: {args.strategy}")
    print(f"실험 이름: {run_name}")
    print(f"결과 경로: {results_path}")
    print()
    
    try:
        # 모델 초기화
        print("모델 초기화 중...")
        model = ModelFactory.get_model_class("vllm")(
            model_name=model_config["model_name"],
            temperature=args.temperature,
            top_p=args.top_p,
            max_tokens=args.max_tokens,
            tensor_parallel_size=args.tensor_parallel_size,
            gpu_memory_utilization=args.gpu_memory_utilization,
            max_model_len=model_config["context_length"]
        )
        print("모델 초기화 완료!")
        
        # 데이터셋 생성
        print("데이터셋 로딩 중...")
        if args.dataset.lower() == "livecodebench":
            data = DatasetFactory.create_dataset(args.dataset, release_version="release_v6")
        else:
            data = DatasetFactory.create_dataset(args.dataset)
        print("데이터셋 로딩 완료!")
        
        # 결과 객체 생성
        results = Results(results_path)
        
        # 전략별 파라미터 설정
        strategy_kwargs = {
            'model': model,
            'data': data,
            'language': args.language,
            'pass_at_k': args.pass_at_k,
            'results': results,
            'verbose': args.verbose
        }
        
        # CodeSIM 전략인 경우 추가 파라미터 설정
        if args.strategy.startswith("CodeSIM"):
            strategy_kwargs.update({
                'additional_info_run': args.additional_info_run,
                'max_plan_try': args.max_plan_try,
                'max_debug_try': args.max_debug_try
            })
        
        # 전략 생성 및 실행
        print(f"{args.strategy} 전략으로 평가 시작...")
        strategy = PromptingFactory.get_prompting_class(args.strategy)(**strategy_kwargs)
        
        strategy.run(result_log_mode='full')
        print("평가 완료!")
        
        # 요약 생성
        print("결과 요약 생성 중...")
        gen_summary(results_path, summary_path)
        print("요약 생성 완료!")
        
        # LiveCodeBench 특별 처리
        if args.dataset.lower() == "livecodebench":
            print("LiveCodeBench 상세 평가 중...")
            lcb_results_path = f"results/{run_name}/Results-LCB.jsonl"
            lcb_summary_path = f"results/{run_name}/Summary-LCB.txt"
            lcb_report_path = f"results/{run_name}/Report-LCB.json"
            
            evaluate_livecodebench_results(results_path, lcb_summary_path)
            generate_livecodebench_report(results_path, lcb_report_path)
            print("LiveCodeBench 평가 완료!")
        
        print(f"\n=== 평가 완료 ===")
        print(f"결과 파일: {results_path}")
        print(f"요약 파일: {summary_path}")
        print(f"로그 파일: {logs_path}")
        
    except Exception as e:
        print(f"오류 발생: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
