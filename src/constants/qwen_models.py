"""
Qwen3 계열 모델들의 설정을 정의합니다.
vLLM을 사용하여 로컬에서 실행할 수 있는 모델들입니다.
"""

QWEN3_MODELS = {
    # Qwen3.5 계열
    "Qwen3.5-0.5B": {
        "model_name": "Qwen/Qwen3.5-0.5B",
        "provider": "vllm",
        "description": "Qwen3.5 0.5B 파라미터 모델",
        "max_tokens": 2048,
        "context_length": 8192
    },
    "Qwen3.5-1.8B": {
        "model_name": "Qwen/Qwen3.5-1.8B", 
        "provider": "vllm",
        "description": "Qwen3.5 1.8B 파라미터 모델",
        "max_tokens": 2048,
        "context_length": 8192
    },
    "Qwen3.5-4B": {
        "model_name": "Qwen/Qwen3.5-4B",
        "provider": "vllm", 
        "description": "Qwen3.5 4B 파라미터 모델",
        "max_tokens": 2048,
        "context_length": 8192
    },
    "Qwen3.5-7B": {
        "model_name": "Qwen/Qwen3.5-7B",
        "provider": "vllm",
        "description": "Qwen3.5 7B 파라미터 모델", 
        "max_tokens": 2048,
        "context_length": 8192
    },
    "Qwen3.5-14B": {
        "model_name": "Qwen/Qwen3.5-14B",
        "provider": "vllm",
        "description": "Qwen3.5 14B 파라미터 모델",
        "max_tokens": 2048,
        "context_length": 8192
    },
    "Qwen3.5-32B": {
        "model_name": "Qwen/Qwen3.5-32B",
        "provider": "vllm",
        "description": "Qwen3.5 32B 파라미터 모델",
        "max_tokens": 2048,
        "context_length": 8192
    },
    "Qwen3.5-72B": {
        "model_name": "Qwen/Qwen3.5-72B",
        "provider": "vllm",
        "description": "Qwen3.5 72B 파라미터 모델",
        "max_tokens": 2048,
        "context_length": 8192
    },
    
    # Qwen3 계열 (정확한 버전)
    "Qwen3-0.5B": {
        "model_name": "Qwen/Qwen3-0.5B",
        "provider": "vllm",
        "description": "Qwen3 0.5B 파라미터 모델",
        "max_tokens": 2048,
        "context_length": 8192
    },
    "Qwen3-1.5B": {
        "model_name": "Qwen/Qwen3-1.5B",
        "provider": "vllm",
        "description": "Qwen3 1.5B 파라미터 모델",
        "max_tokens": 2048,
        "context_length": 8192
    },
    "Qwen3-3B": {
        "model_name": "Qwen/Qwen3-3B",
        "provider": "vllm",
        "description": "Qwen3 3B 파라미터 모델",
        "max_tokens": 2048,
        "context_length": 8192
    },
    "Qwen3-7B": {
        "model_name": "Qwen/Qwen3-7B",
        "provider": "vllm",
        "description": "Qwen3 7B 파라미터 모델",
        "max_tokens": 2048,
        "context_length": 8192
    },
    "Qwen3-14B": {
        "model_name": "Qwen/Qwen3-14B",
        "provider": "vllm",
        "description": "Qwen3 14B 파라미터 모델",
        "max_tokens": 2048,
        "context_length": 8192
    },
    "Qwen3-32B": {
        "model_name": "Qwen/Qwen3-32B",
        "provider": "vllm",
        "description": "Qwen3 32B 파라미터 모델",
        "max_tokens": 2048,
        "context_length": 8192
    },
    "Qwen3-72B": {
        "model_name": "Qwen/Qwen3-72B",
        "provider": "vllm",
        "description": "Qwen3 72B 파라미터 모델",
        "max_tokens": 2048,
        "context_length": 8192
    },
    
    # Qwen3.5-MoE 계열
    "Qwen3.5-MoE-2.7B": {
        "model_name": "Qwen/Qwen3.5-MoE-2.7B",
        "provider": "vllm",
        "description": "Qwen3.5 MoE 2.7B 파라미터 모델",
        "max_tokens": 2048,
        "context_length": 8192
    },
    "Qwen3.5-MoE-3.5B": {
        "model_name": "Qwen/Qwen3.5-MoE-3.5B",
        "provider": "vllm",
        "description": "Qwen3.5 MoE 3.5B 파라미터 모델",
        "max_tokens": 2048,
        "context_length": 8192
    },
    "Qwen3.5-MoE-6.5B": {
        "model_name": "Qwen/Qwen3.5-MoE-6.5B",
        "provider": "vllm",
        "description": "Qwen3.5 MoE 6.5B 파라미터 모델",
        "max_tokens": 2048,
        "context_length": 8192
    },
    "Qwen3.5-MoE-12B": {
        "model_name": "Qwen/Qwen3.5-MoE-12B",
        "provider": "vllm",
        "description": "Qwen3.5 MoE 12B 파라미터 모델",
        "max_tokens": 2048,
        "context_length": 8192
    },
    "Qwen3.5-MoE-20B": {
        "model_name": "Qwen/Qwen3.5-MoE-20B",
        "provider": "vllm",
        "description": "Qwen3.5 MoE 20B 파라미터 모델",
        "max_tokens": 2048,
        "context_length": 8192
    },
    "Qwen3.5-MoE-32B": {
        "model_name": "Qwen/Qwen3.5-MoE-32B",
        "provider": "vllm",
        "description": "Qwen3.5 MoE 32B 파라미터 모델",
        "max_tokens": 2048,
        "context_length": 8192
    },
    
    # Qwen3-Coder 계열 (정확한 버전)
    "Qwen3-Coder-0.5B": {
        "model_name": "Qwen/Qwen3-Coder-0.5B",
        "provider": "vllm",
        "description": "Qwen3-Coder 0.5B 파라미터 코드 전용 모델",
        "max_tokens": 2048,
        "context_length": 8192,
        "is_coder": True
    },
    "Qwen3-Coder-1.5B": {
        "model_name": "Qwen/Qwen3-Coder-1.5B",
        "provider": "vllm",
        "description": "Qwen3-Coder 1.5B 파라미터 코드 전용 모델",
        "max_tokens": 2048,
        "context_length": 8192,
        "is_coder": True
    },
    "Qwen3-Coder-3B": {
        "model_name": "Qwen/Qwen3-Coder-3B",
        "provider": "vllm",
        "description": "Qwen3-Coder 3B 파라미터 코드 전용 모델",
        "max_tokens": 2048,
        "context_length": 8192,
        "is_coder": True
    },
    "Qwen3-Coder-7B": {
        "model_name": "Qwen/Qwen3-Coder-7B",
        "provider": "vllm",
        "description": "Qwen3-Coder 7B 파라미터 코드 전용 모델",
        "max_tokens": 2048,
        "context_length": 8192,
        "is_coder": True
    },
    "Qwen3-Coder-14B": {
        "model_name": "Qwen/Qwen3-Coder-14B",
        "provider": "vllm",
        "description": "Qwen3-Coder 14B 파라미터 코드 전용 모델",
        "max_tokens": 2048,
        "context_length": 8192,
        "is_coder": True
    },
    "Qwen3-Coder-32B": {
        "model_name": "Qwen/Qwen3-Coder-32B",
        "provider": "vllm",
        "description": "Qwen3-Coder 32B 파라미터 코드 전용 모델",
        "max_tokens": 2048,
        "context_length": 8192,
        "is_coder": True
    },
    "Qwen3-Coder-72B": {
        "model_name": "Qwen/Qwen3-Coder-72B",
        "provider": "vllm",
        "description": "Qwen3-Coder 72B 파라미터 코드 전용 모델",
        "max_tokens": 2048,
        "context_length": 8192,
        "is_coder": True
    },
    
    # Qwen3-Coder-MoE 계열
    "Qwen3-Coder-MoE-2.7B": {
        "model_name": "Qwen/Qwen3-Coder-MoE-2.7B",
        "provider": "vllm",
        "description": "Qwen3-Coder MoE 2.7B 파라미터 코드 전용 모델",
        "max_tokens": 2048,
        "context_length": 8192,
        "is_coder": True
    },
    "Qwen3-Coder-MoE-3.5B": {
        "model_name": "Qwen/Qwen3-Coder-MoE-3.5B",
        "provider": "vllm",
        "description": "Qwen3-Coder MoE 3.5B 파라미터 코드 전용 모델",
        "max_tokens": 2048,
        "context_length": 8192,
        "is_coder": True
    },
    "Qwen3-Coder-MoE-6.5B": {
        "model_name": "Qwen/Qwen3-Coder-MoE-6.5B",
        "provider": "vllm",
        "description": "Qwen3-Coder MoE 6.5B 파라미터 코드 전용 모델",
        "max_tokens": 2048,
        "context_length": 8192,
        "is_coder": True
    },
    "Qwen3-Coder-MoE-12B": {
        "model_name": "Qwen/Qwen3-Coder-MoE-12B",
        "provider": "vllm",
        "description": "Qwen3-Coder MoE 12B 파라미터 코드 전용 모델",
        "max_tokens": 2048,
        "context_length": 8192,
        "is_coder": True
    },
    "Qwen3-Coder-MoE-20B": {
        "model_name": "Qwen/Qwen3-Coder-MoE-20B",
        "provider": "vllm",
        "description": "Qwen3-Coder MoE 20B 파라미터 코드 전용 모델",
        "max_tokens": 2048,
        "context_length": 8192,
        "is_coder": True
    },
    "Qwen3-Coder-MoE-32B": {
        "model_name": "Qwen/Qwen3-Coder-MoE-32B",
        "provider": "vllm",
        "description": "Qwen3-Coder MoE 32B 파라미터 코드 전용 모델",
        "max_tokens": 2048,
        "context_length": 8192,
        "is_coder": True
    }
}

def get_qwen_model_config(model_name: str) -> dict:
    """
    Qwen 모델 이름에 해당하는 설정을 반환합니다.
    
    Args:
        model_name: 모델 이름 (예: "Qwen3.5-7B")
        
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
