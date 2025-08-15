import json
import os
import numpy as np
import tqdm
from yaml import safe_load
from typing import List, Dict, Any
import tempfile
import subprocess
import sys

from .api_comm import APICommunication
from .exec_outcome import ExecOutcome
from constants.lang_mappings import LANGUAGE_MAPPING

limits_by_lang_cfg_file = "./src/evaluations/limits_by_lang.yaml"

assert os.path.exists(
    limits_by_lang_cfg_file), "Need resource limit defaults for all runtimes, provide the path to default 'limits_by_lang.yaml' or to the modified one."

with open(limits_by_lang_cfg_file) as limit_cfg_rp:
    limits_by_lang = safe_load(limit_cfg_rp)

unittest_file = "./data/xCodeEval/unittest_db.json"
assert os.path.exists(unittest_file), "Unittest file not found."

with open(unittest_file) as ut_rp:
    unittest_db = json.load(ut_rp)


api_comm = APICommunication()


def xcode_evaluate(
    generated_code: str,
    src_uid: str,
    lang: str
):

    assert src_uid in unittest_db, "Can not find the task id or source id"

    assert lang in LANGUAGE_MAPPING, f"language must be inside the supported language list: {LANGUAGE_MAPPING.keys()}"

    results, _, _ = api_comm.execute_code(
        language=LANGUAGE_MAPPING[lang],
        source_code=generated_code,
        unittests=unittest_db[src_uid],
        limits=limits_by_lang[LANGUAGE_MAPPING[lang]],
        task_id=src_uid,
    )

    if results == "error":
        return False

    passed = True
    for result in results:
        if result['exec_outcome'] != ExecOutcome.PASSED.value:
            passed = False
            break

    return passed


def xcode_execute_internal_test(
    generated_code: str,
    tests: List[dict],
    src_uid: str,
    lang: str
):
    results, _, _ = api_comm.execute_code(
        language=LANGUAGE_MAPPING[lang],
        source_code=generated_code,
        unittests=tests,
        limits=limits_by_lang[LANGUAGE_MAPPING[lang]],
        task_id=src_uid,
        stop_on_first_fail=False
    )

    passed = True
    passed_feedback = []
    failed_feedback = []

    idx = 0
    try:
        for idx, result in enumerate(results):
            if result['exec_outcome'] == ExecOutcome.PASSED.value:
                passed_feedback.append(tests[idx])
            if result['exec_outcome'] != ExecOutcome.PASSED.value:
                failed_feedback.append(tests[idx])
                passed = False
    except:
        passed = False
        failed_feedback.extend(tests[idx:])

    feedback = f'Tested passed: \n{json.dumps(passed_feedback)}\n\nTests failed: \n{json.dumps(failed_feedback)}'

    return passed, feedback


def normalize_output(output: str) -> List[str]:
    """
    Normalize output for comparison:
    - strip trailing spaces
    - split into lines
    """
    return [line.rstrip() for line in output.strip().splitlines()]


def livecodebench_evaluate(code: str, test_cases: List[Dict[str, Any]], timeout: int = 5) -> tuple[bool, str]:
    """
    Evaluate Python code against LiveCodeBench-style test cases (local execution).
    Returns:
        passed (bool): 모든 테스트 통과 여부
        feedback (str): 실패한 경우 상세 정보, 통과 시 빈 문자열
    """
    code_path = None
    try:
        # Save the code to a temp file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as tmp:
            tmp.write(code)
            code_path = tmp.name

        for idx, tc in enumerate(test_cases):
            input_data = tc.get("input", "")
            expected_output = tc.get("output", "")

            # handle if expected_output is list or str
            if isinstance(expected_output, list):
                expected_lines = normalize_output("\n".join(expected_output))
            else:
                expected_lines = normalize_output(str(expected_output))

            try:
                result = subprocess.run(
                    [sys.executable, code_path],
                    input=input_data.encode(),
                    capture_output=True,
                    text=True,
                    timeout=tc.get("timeout", timeout)
                )

                actual_lines = normalize_output(result.stdout)

                if actual_lines != expected_lines:
                    feedback = (
                        f"[Fail] Test case {idx+1}:\n"
                        f"Input:\n{input_data}\n"
                        f"Expected:\n{expected_lines}\n"
                        f"Got:\n{actual_lines}"
                    )
                    return False, feedback

            except subprocess.TimeoutExpired:
                feedback = f"[Timeout] Test case {idx+1} exceeded {timeout} seconds."
                return False, feedback

            except Exception as e:
                feedback = f"[Error] Test case {idx+1} execution failed: {e}"
                return False, feedback

        return True, ""  # All test cases passed

    finally:
        if code_path and os.path.exists(code_path):
            try:
                os.remove(code_path)
            except Exception as e:
                print(f"[Warning] Could not remove temp file: {e}")