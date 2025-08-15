from dataclasses import dataclass, field

import requests
from .exec_outcome import ExecOutcome

# 확장된 유닛 테스트 결과를 저장하는 데이터 클래스
@dataclass
class ExtendedUnittest:
    input: str                                   # 테스트 입력값
    output: list[str] = field(default_factory=list)  # 테스트 실행 결과 출력값
    result: str | None = None                    # 테스트 결과 상태 (성공/실패 등)
    exec_outcome: ExecOutcome | None = None      # 실행 결과 상태 (ExecOutcome Enum)

    # 객체를 JSON 형식(딕셔너리)으로 변환
    def json(self):
        _json = self.__dict__                    # dataclass의 속성을 딕셔너리로 변환
        if self.exec_outcome is not None:
            # Enum 값이면 문자열 이름으로 변환
            _json["exec_outcome"] = self.exec_outcome.name

        return _json

    # JSON(딕셔너리)로부터 ExtendedUnittest 객체 생성
    @classmethod
    def from_json(cls, _json):
        return cls(
            input=_json.get("input", ""),
            output=_json.get("output", list()),
            result=_json.get("result", None),
            exec_outcome=_json.get("exec_outcome", None),
        )


# 빈 값 관련 예외 클래스
class EmptyValueError(Exception):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


# 구체적인 빈 값 예외들
class EmptyUnittestError(EmptyValueError):   # 유닛 테스트가 없는 경우
    pass

class EmptyLanguageError(EmptyValueError):   # 언어가 없는 경우
    pass

class EmptySourceCodeError(EmptyValueError): # 소스 코드가 없는 경우
    pass


# API 서버와 통신을 담당하는 클래스
class APICommunication:
    _session: requests.Session

    def __init__(self, server_url: str = "http://localhost:5000"):
        self._session = requests.Session()                   # 세션 객체 생성
        self.execute_code_url = f"{server_url}/api/execute_code"  # 코드 실행 API 엔드포인트
        self.get_runtimes_url = f"{server_url}/api/all_runtimes"  # 런타임 정보 API 엔드포인트

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self._session.close()  # 세션 종료

    # 서버에서 지원하는 런타임 목록 가져오기
    def get_runtimes(self):
        return self._session.get(self.get_runtimes_url).json()

    # 서버에 코드 실행 요청
    def execute_code(
        self,
        language: str,                        # 사용 언어 (예: python, java 등)
        source_code: str,                     # 실행할 소스 코드
        unittests: list[dict],                 # 유닛 테스트 목록
        limits: dict | None,                   # 실행 리소스 제한
        block_network: bool = True,            # 네트워크 차단 여부
        stop_on_first_fail: bool = True,       # 첫 실패 시 중단 여부
        use_sanitizer: bool = False,           # 메모리 에러 검사 도구 사용 여부
        compiler_program_name: str | None = None,  # 컴파일러 프로그램 이름
        compiler_flags: str | None = None,         # 컴파일러 옵션
        interpreter_cmd: str | None = None,        # 인터프리터 명령어
        interpreter_flags: str | None = None,      # 인터프리터 옵션
        sample_id: int | None = None,              # 샘플 ID
        task_id: str | int | None = None,          # 태스크 ID
    ) -> tuple[list[ExtendedUnittest], int | None, str | int | None]:
        
        # 필수 값 체크
        if language is None:
            raise EmptyLanguageError
        if source_code is None:
            raise EmptySourceCodeError
        if unittests is None or len(unittests) == 0:
            raise EmptyUnittestError

        # 요청 바디 구성
        request_body = dict(
            language=language,
            source_code=source_code,
            unittests=unittests,
            limits=limits if isinstance(limits, dict) else dict(),
            compile_cmd=compiler_program_name,
            compile_flags=compiler_flags,
            execute_cmd=interpreter_cmd,
            execute_flags=interpreter_flags,
            block_network=block_network,
            stop_on_first_fail=stop_on_first_fail,
            use_sanitizer=use_sanitizer,
        )

        # API 서버에 POST 요청
        json_response = self._session.post(
            self.execute_code_url,
            json=request_body,
            headers={"Content-Type": "application/json"},
        ).json()

        # 에러 응답 처리
        if "error" in json_response:
            return "error", json_response["error"], task_id
        if "data" not in json_response:
            return "error", str(json_response), task_id

        # 정상 응답
        return (
            json_response["data"],
            None,
            task_id,
        )
