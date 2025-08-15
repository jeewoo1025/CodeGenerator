import logging
import sys
import traceback
from typing import Optional, List, Dict, Any
import time
import json

from vllm import LLM, SamplingParams
from models.Base import BaseModel


class VLLMModel(BaseModel):
    def __init__(self, model_name: str, **kwargs):
        super().__init__(**kwargs)
        self.model_name = model_name
        self.llm = None
        self.max_tokens = kwargs.get('max_tokens', 2048)
        self.temperature = kwargs.get('temperature', 0.0)
        self.top_p = kwargs.get('top_p', 0.95)
        self.frequency_penalty = kwargs.get('frequency_penalty', 0.0)
        self.presence_penalty = kwargs.get('presence_penalty', 0.0)
        
        # vLLM 모델 초기화
        try:
            self.llm = LLM(
                model=model_name,
                trust_remote_code=True,
                tensor_parallel_size=kwargs.get('tensor_parallel_size', 1),
                gpu_memory_utilization=kwargs.get('gpu_memory_utilization', 0.9),
                max_model_len=kwargs.get('max_model_len', 8192),
                dtype=kwargs.get('dtype', 'auto')
            )
            logging.info(f"vLLM 모델 {model_name} 초기화 완료")
        except Exception as e:
            logging.error(f"vLLM 모델 초기화 실패: {e}")
            raise e

    def prompt(self, processed_input, frequency_penalty: float = 0, presence_penalty: float = 0) -> tuple:
        """
        vLLM을 사용하여 프롬프트를 처리합니다.
        
        Args:
            processed_input: 입력 프롬프트 (문자열 또는 List[dict])
            frequency_penalty: 빈도 페널티 (vLLM에서는 지원하지 않음)
            presence_penalty: 존재 페널티 (vLLM에서는 지원하지 않음)
            
        Returns:
            (응답 텍스트, 실행 세부사항) 튜플
        """
        if self.llm is None:
            raise Exception("vLLM 모델이 초기화되지 않았습니다.")
        
        try:
            # 입력 형식 처리
            if isinstance(processed_input, list) and len(processed_input) > 0:
                # Chat 형식인 경우 첫 번째 메시지의 content 사용
                if isinstance(processed_input[0], dict) and 'content' in processed_input[0]:
                    input_text = processed_input[0]['content']
                else:
                    input_text = str(processed_input[0])
            else:
                input_text = str(processed_input)
            
            # SamplingParams 설정
            sampling_params = SamplingParams(
                temperature=self.temperature,
                top_p=self.top_p,
                max_tokens=self.max_tokens,
                stop=None
            )
            
            # 모델 추론 실행
            outputs = self.llm.generate([input_text], sampling_params)
            
            if outputs and len(outputs) > 0:
                response = outputs[0].outputs[0].text
                # vLLM 실행 세부사항
                run_details = {
                    'model_name': self.model_name,
                    'temperature': self.temperature,
                    'top_p': self.top_p,
                    'max_tokens': self.max_tokens,
                    'provider': 'vllm'
                }
                return response.strip(), run_details
            else:
                run_details = {
                    'model_name': self.model_name,
                    'error': 'No output generated',
                    'provider': 'vllm'
                }
                return "", run_details
                
        except Exception as e:
            logging.error(f"vLLM 추론 중 오류 발생: {e}")
            logging.error(traceback.format_exc())
            run_details = {
                'model_name': self.model_name,
                'error': str(e),
                'provider': 'vllm'
            }
            raise Exception(f"vLLM 오류: {e}") from e

    def __del__(self):
        """소멸자에서 리소스 정리"""
        if hasattr(self, 'llm') and self.llm is not None:
            try:
                del self.llm
            except:
                pass
