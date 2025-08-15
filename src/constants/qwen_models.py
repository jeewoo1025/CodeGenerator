"""
Qwen3 계열 모델들의 설정을 정의합니다.
vLLM을 사용하여 로컬에서 실행할 수 있는 모델들입니다.
"""

QWEN3_MODELS = {
    # Qwen3 계열
    "Qwen3-0.6B": {
        "model_name": "Qwen/Qwen3-0.6B",
        "provider": "vllm",
        "description": "Qwen3 0.6B 파라미터 모델",
        "max_tokens": 32768,
        "context_length": 32768
    },
    "Qwen3-1.7B": {
        "model_name": "Qwen/Qwen3-1.7B",
        "provider": "vllm",
        "description": "Qwen3 1.7B 파라미터 모델",
        "max_tokens": 32768,
        "context_length": 32768
    },
    "Qwen3-4B": {
        "model_name": "Qwen/Qwen3-4B",
        "provider": "vllm",
        "description": "Qwen3 4B 파라미터 모델",
        "max_tokens": 32768,
        "context_length": 32768
    },
    "Qwen3-8B": {
        "model_name": "Qwen/Qwen3-8B",
        "provider": "vllm",
        "description": "Qwen3 8B 파라미터 모델",
        "max_tokens": 32768,
        "context_length": 128000
    },
    "Qwen3-14B": {
        "model_name": "Qwen/Qwen3-14B",
        "provider": "vllm",
        "description": "Qwen3 14B 파라미터 모델",
        "max_tokens": 32768,
        "context_length": 128000
    },
    "Qwen3-32B": {
        "model_name": "Qwen/Qwen3-32B",
        "provider": "vllm",
        "description": "Qwen3 32B 파라미터 모델",
        "max_tokens": 32768,
        "context_length": 128000
    },
    "Qwen3-30B-A3B": {
        "model_name": "Qwen/Qwen3-30B-A3B",
        "provider": "vllm",
        "description": "Qwen3 30B 파라미터 MoE 모델 (3B 활성 파라미터)",
        "max_tokens": 32768,
        "context_length": 128000
    },
    "Qwen3-235B-A22B": {
        "model_name": "Qwen/Qwen3-235B-A22B",
        "provider": "vllm",
        "description": "Qwen3 235B 파라미터 MoE 모델 (22B 활성 파라미터)",
        "max_tokens": 32768,
        "context_length": 128000
    },
    # Qwen3-Coder 계열
    "Qwen3-Coder-30B-A3B-Instruct": {
        "model_name": "Qwen/Qwen3-Coder-30B-A3B-Instruct",
        "provider": "vllm",
        "description": "Qwen3-Coder 30B 파라미터 코드 전용 MoE 모델 (3B 활성 파라미터)",
        "max_tokens": 65536,
        "context_length": 256000,
        "is_coder": True
    },
    "Qwen3-Coder-480B-A35B-Instruct": {
        "model_name": "Qwen/Qwen3-Coder-480B-A35B-Instruct",
        "provider": "vllm",
        "description": "Qwen3-Coder 480B 파라미터 코드 전용 MoE 모델 (35B 활성 파라미터)",
        "max_tokens": 65536,
        "context_length": 256000,
        "is_coder": True
    }
}

def get_qwen_model_config(model_name: str) -> dict:
    """
    Qwen 모델 이름에 해당하는 설정을 반환합니다.
    
    Args:
        model_name: 모델 이름 (예: "Qwen3-4B")
        
    Returns:
        모델 설정 딕셔너리
    """
    return QWEN3_MODELS.get(model_name, {})

def list_available_qwen_models() -> list:
    """
    사용 가능한 모든 Qwen 모델 이름을 반환합니다.
    
    Returns:
        모델 이름 리스트
    """
    return list(QWEN3_MODELS.keys())

def list_coder_models() -> list:
    """
    코드 전용 모델들의 이름을 반환합니다.
    
    Returns:
        코드 전용 모델 이름 리스트
    """
    return [name for name, config in QWEN3_MODELS.items() if config.get('is_coder', False)]

def is_qwen_model(model_name: str) -> bool:
    """
    주어진 모델 이름이 Qwen 모델인지 확인합니다.
    
    Args:
        model_name: 모델 이름
        
    Returns:
        Qwen 모델이면 True, 아니면 False
    """
    return model_name in QWEN3_MODELS

def is_coder_model(model_name: str) -> bool:
    """
    주어진 모델이 코드 전용 모델인지 확인합니다.
    
    Args:
        model_name: 모델 이름
        
    Returns:
        코드 전용 모델이면 True, 아니면 False
    """
    config = get_qwen_model_config(model_name)
    return config.get('is_coder', False)