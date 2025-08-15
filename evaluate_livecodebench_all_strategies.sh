#!/bin/bash

# LiveCodeBench 데이터셋을 모든 prompting 기법으로 평가하는 스크립트
# Model: GPT-4.1 (OpenAI)

echo "=========================================="
echo "LiveCodeBench 전체 Prompting 기법 평가 시작"
echo "Model: GPT-4.1 (OpenAI)"
echo "Dataset: LiveCodeBench release_v6"
echo "=========================================="

# 기본 설정
DATASET="lcb_release_v6"
MODEL="gpt-4o"
MODEL_PROVIDER="OpenAI"
LANGUAGE="Python3"
TEMPERATURE=0
TOP_P=0.95
PASS_AT_K=1
VERBOSE="2"
CONTINUE="yes"
RESULT_LOG="partial"
STORE_LOG_IN_FILE="yes"

# 모든 prompting 기법 리스트
STRATEGIES=(
    "Direct"
    "CoT"
    "SelfPlanning"
    "Analogical"
    "MapCoder"
    "CodeSIM"
    "CodeSIMWD"
    "CodeSIMWPV"
    "CodeSIMWPVD"
    "CodeSIMA"
    "CodeSIMC"
)

# 결과 요약 파일
SUMMARY_FILE="livecodebench_evaluation_summary.txt"
echo "LiveCodeBench 전체 Prompting 기법 평가 결과 요약" > $SUMMARY_FILE
echo "평가 시작 시간: $(date)" >> $SUMMARY_FILE
echo "Model: $MODEL ($MODEL_PROVIDER)" >> $SUMMARY_FILE
echo "Dataset: $DATASET" >> $SUMMARY_FILE
echo "==========================================" >> $SUMMARY_FILE

# 각 prompting 기법에 대해 평가 실행
for strategy in "${STRATEGIES[@]}"; do
    echo ""
    echo "=========================================="
    echo "평가 중: $strategy"
    echo "시작 시간: $(date)"
    echo "=========================================="
    
    # 평가 명령어 실행
    python src/main.py \
        --dataset $DATASET \
        --strategy $strategy \
        --model $MODEL \
        --model_provider $MODEL_PROVIDER \
        --language $LANGUAGE \
        --temperature $TEMPERATURE \
        --top_p $TOP_P \
        --pass_at_k $PASS_AT_K \
        --verbose $VERBOSE \
        --cont $CONTINUE \
        --result_log $RESULT_LOG \
        --store_log_in_file $STORE_LOG_IN_FILE
    
    # 실행 결과 확인
    if [ $? -eq 0 ]; then
        echo "✅ $strategy 평가 완료"
        echo "✅ $strategy 평가 완료 - $(date)" >> $SUMMARY_FILE
        
        # 결과 파일 경로 찾기
        RESULT_DIR="results/$DATASET/$strategy/$MODEL/$LANGUAGE-$TEMPERATURE-$TOP_P-$PASS_AT_K"
        if [ -d "$RESULT_DIR" ]; then
            # 가장 최근 Run 폴더 찾기
            LATEST_RUN=$(ls -td "$RESULT_DIR"/Run-* 2>/dev/null | head -1)
            if [ -n "$LATEST_RUN" ]; then
                echo "   결과 저장 위치: $LATEST_RUN"
                echo "   결과 저장 위치: $LATEST_RUN" >> $SUMMARY_FILE
                
                # Summary.txt 파일이 있다면 내용 확인
                if [ -f "$LATEST_RUN/Summary.txt" ]; then
                    echo "   요약 정보:"
                    echo "   요약 정보:" >> $SUMMARY_FILE
                    tail -20 "$LATEST_RUN/Summary.txt" | while read line; do
                        echo "     $line"
                        echo "     $line" >> $SUMMARY_FILE
                    done
                fi
            fi
        fi
    else
        echo "❌ $strategy 평가 실패"
        echo "❌ $strategy 평가 실패 - $(date)" >> $SUMMARY_FILE
    fi
    
    echo "=========================================="
    echo ""
    
    # API 호출 제한을 고려한 대기 (필요시)
    if [ "$strategy" != "${STRATEGIES[-1]}" ]; then
        echo "다음 평가를 위해 10초 대기..."
        sleep 10
    fi
done

echo ""
echo "=========================================="
echo "모든 Prompting 기법 평가 완료!"
echo "완료 시간: $(date)"
echo "=========================================="

echo "" >> $SUMMARY_FILE
echo "==========================================" >> $SUMMARY_FILE
echo "모든 Prompting 기법 평가 완료!" >> $SUMMARY_FILE
echo "완료 시간: $(date)" >> $SUMMARY_FILE
echo "==========================================" >> $SUMMARY_FILE

# 최종 요약 출력
echo ""
echo "평가 결과 요약이 $SUMMARY_FILE 파일에 저장되었습니다."
echo ""
echo "개별 결과는 다음 경로에서 확인할 수 있습니다:"
echo "  results/$DATASET/[STRATEGY]/$MODEL/$LANGUAGE-$TEMPERATURE-$TOP_P-$PASS_AT_K/Run-[N]/"

# 전체 결과 통합 요약 생성
echo ""
echo "전체 결과 통합 요약 생성 중..."
python -c "
import os
import json
from datetime import datetime

def generate_overall_summary():
    base_path = 'results/$DATASET'
    summary = {
        'evaluation_time': datetime.now().isoformat(),
        'model': '$MODEL',
        'model_provider': '$MODEL_PROVIDER',
        'dataset': '$DATASET',
        'strategies': {}
    }
    
    strategies = ['Direct', 'CoT', 'SelfPlanning', 'Analogical', 'MapCoder', 
                 'CodeSIM', 'CodeSIMWD', 'CodeSIMWPV', 'CodeSIMWPVD', 'CodeSIMA', 'CodeSIMC']
    
    for strategy in strategies:
        strategy_path = f'{base_path}/{strategy}/$MODEL/$LANGUAGE-$TEMPERATURE-$TOP_P-$PASS_AT_K'
        if os.path.exists(strategy_path):
            # 가장 최근 Run 폴더 찾기
            run_dirs = [d for d in os.listdir(strategy_path) if d.startswith('Run-')]
            if run_dirs:
                latest_run = sorted(run_dirs)[-1]
                run_path = f'{strategy_path}/{latest_run}'
                
                # Results.jsonl 파일에서 통계 추출
                results_file = f'{run_path}/Results.jsonl'
                if os.path.exists(results_file):
                    try:
                        with open(results_file, 'r', encoding='utf-8') as f:
                            results = [json.loads(line) for line in f if line.strip()]
                        
                        total_problems = len(results)
                        passed_problems = sum(1 for r in results if r.get('passed', False))
                        pass_rate = (passed_problems / total_problems * 100) if total_problems > 0 else 0
                        
                        summary['strategies'][strategy] = {
                            'total_problems': total_problems,
                            'passed_problems': passed_problems,
                            'pass_rate': round(pass_rate, 2),
                            'result_path': run_path
                        }
                    except Exception as e:
                        summary['strategies'][strategy] = {'error': str(e)}
                else:
                    summary['strategies'][strategy] = {'status': 'No results file'}
            else:
                summary['strategies'][strategy] = {'status': 'No run directory'}
        else:
            summary['strategies'][strategy] = {'status': 'Not evaluated'}
    
    # 전체 요약 파일 저장
    with open('livecodebench_overall_summary.json', 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    # 콘솔 출력
    print('\\n=== LiveCodeBench 전체 평가 결과 요약 ===')
    print(f'평가 시간: {summary[\"evaluation_time\"]}')
    print(f'모델: {summary[\"model\"]} ({summary[\"model_provider\"]})')
    print(f'데이터셋: {summary[\"dataset\"]}')
    print('\\n각 전략별 결과:')
    print('-' * 80)
    
    for strategy, data in summary['strategies'].items():
        if 'pass_rate' in data:
            print(f'{strategy:15} | 총 문제: {data[\"total_problems\"]:3d} | 통과: {data[\"passed_problems\"]:3d} | 통과율: {data[\"pass_rate\"]:5.2f}%')
        else:
            print(f'{strategy:15} | 상태: {data.get(\"status\", \"Unknown\")}')
    
    print('-' * 80)
    print(f'\\n전체 요약이 livecodebench_overall_summary.json 파일에 저장되었습니다.')

generate_overall_summary()
"
