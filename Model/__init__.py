from .model import BBOB, BBOBConfig
from .build import build_BBOB
from transformers import AutoConfig, AutoModel

# Register so that Auto classes can resolve our implementation
AutoConfig.register(BBOBConfig.model_type, BBOBConfig)
AutoModel.register(BBOBConfig, BBOB)

__all__ = [
    "BBOBConfig",
    "BBOB",
    "build_BBOB",
] 