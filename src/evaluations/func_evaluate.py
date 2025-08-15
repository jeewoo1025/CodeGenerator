from typing import *
import contextlib
import signal

from .executor_utils import function_with_timeout


def evaluate_io(
    sample_io: list[str],
    completion: str,
    timeout: int = 5,
    stop_early: bool = False,
):
    """
    주어진 코드(completion)와 입출력 예제(sample_io)를 실행하여
    테스트를 통과하는지 확인하는 함수.

    Args:
        sample_io (list[str]): 테스트에 사용할 입력/출력 코드 조각 목록
        completion (str): 테스트할 코드 (함수 구현 등)
        timeout (int): 각 테스트 실행 제한 시간(초)
        stop_early (bool): True이면 첫 실패 시 바로 종료

    Returns:
        tuple[bool, str]: (전체 테스트 통과 여부, 테스트 로그 문자열)
    """
    if len(sample_io) == 0:
        # 테스트 케이스가 없는 경우 무조건 통과
        return True, ""

    test_log = ""  # 테스트 결과 로그 저장
    passed = True  # 모든 테스트 통과 여부
    for io in sample_io:
        try:
            # 코드에 "from typing import *"가 없으면 추가
            code = ("from typing import *\n" if "from typing import *" not in completion else "") \
                + completion + "\n" + io + "\n"
            
            # 제한 시간 내 코드 실행
            function_with_timeout(
                exec,  # 파이썬 코드 실행
                (code, globals()),  # exec 인자
                timeout
            )
            test_log += f"Passed in test case: {io}\n"
        except Exception as e:
            if stop_early:
                # 첫 실패 시 즉시 종료
                return False, f"Failed in test case: {io}\n"
            passed = False
            test_log += f"Failed in test case: {io}\n"

    return passed, test_log


def evaluate_io_et(
    sample_io: list[str],
    completion: str,
    timeout: int = 5,
    prompt: str = "",
):
    """
    sample_io 목록을 하나로 합쳐서 코드(completion)와 함께 실행하는 함수.

    Args:
        sample_io (list[str]): 테스트 케이스 코드 조각 목록
        completion (str): 테스트할 코드
        timeout (int): 실행 제한 시간(초)
        prompt (str): 코드 앞부분에 추가할 프롬프트/코드

    Returns:
        bool: 모든 실행이 성공하면 True, 실패 시 False
    """
    io = "\n".join(sample_io)  # 모든 테스트 케이스를 하나로 합침
    try:
        # typing 임포트 여부 확인 후 코드 구성
        code = ("from typing import *\n" if "from typing import *" not in completion else "") \
            + prompt + completion + "\n" + io + "\n"
        
        function_with_timeout(
            exec,
            (code, globals()),
            timeout
        )
        return True
    except Exception as e:
        return False


def evaluate_functional_correctness(
    test: str,
    entry_point: str,
    completion: str,
    timeout: int = 5,
):
    """
    함수 동작의 정합성을 검증하는 함수.

    Args:
        test (str): 테스트 코드
        entry_point (str): 검증할 함수 이름
        completion (str): 함수 구현 코드
        timeout (int): 실행 제한 시간(초)

    Returns:
        str: "passed" 또는 "failed: 에러메시지"
    """
    try:
        # check(entry_point)로 함수의 동작 검증
        code = ("from typing import *\n" if "from typing import *" not in completion else "") \
            + completion + "\n" + test + "\n" + f"check({entry_point})"

        function_with_timeout(
            exec,
            (code, globals()),
            timeout
        )
        return "passed"
    except Exception as e:
        return f"failed: {e}"


class TimeoutException(Exception):
    """코드 실행 제한 시간 초과 시 사용되는 예외"""
    pass
