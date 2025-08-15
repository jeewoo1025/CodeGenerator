# LiveCodeBench Dataset

이 디렉토리는 LiveCodeBench 데이터셋을 저장하는 곳입니다.

## 데이터 다운로드

LiveCodeBench 데이터를 다운로드하려면 다음 명령어를 실행하세요:

```bash
# LiveCodeBench 공식 저장소에서 데이터 다운로드
git clone https://github.com/LiveCodeBench/LiveCodeBench.git
cd LiveCodeBench

# 또는 Xolver 저장소의 lcv 폴더에서 데이터 가져오기
git clone https://github.com/kagnlp/Xolver.git
cd Xolver/lcv
```

## 데이터 형식

LiveCodeBench 데이터는 다음과 같은 형식이어야 합니다:

```json
{
  "id": "문제_고유_ID",
  "title": "문제_제목",
  "difficulty": "난이도",
  "language": "프로그래밍_언어",
  "description": "문제_설명",
  "input_format": "입력_형식_설명",
  "output_format": "출력_형식_설명",
  "constraints": "제약_조건",
  "examples": "예시_입출력",
  "test_cases": [
    {
      "input": "테스트_입력",
      "output": "예상_출력"
    }
  ],
  "solution": "참고_솔루션_코드"
}
```

## 사용법

main.py에서 LiveCodeBench 데이터셋을 사용하려면:

```bash
python src/main.py --dataset LiveCodeBench --language Python3 --strategy Direct
```

또는 약어 사용:

```bash
python src/main.py --dataset lcb --language Python3 --strategy Direct
```
