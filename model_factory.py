from enum import Enum
from typing import Union, Dict, Type
from ..logit_lens.model_helper.llama_2_helper import Llama7BHelper
from ..logit_lens.model_helper.llama_3_1_helper import Llama3_1_8BHelper
from ..logit_lens.model_helper.qwen_helper import QwenHelper

class ModelType(Enum):
    """Enumeration of supported model types"""
    LLAMA_7B = "llama_7b"
    LLAMA_3_1_8B = "llama_3_1_8b"
    QWEN_7B = "qwen_7b"
    
    @classmethod
    def from_string(cls, model_name: str) -> 'ModelType':
        """Create ModelType enum value from string"""
        try:
            return cls(model_name.lower())
        except ValueError:
            raise ValueError(f"Unsupported model type: {model_name}")

class ModelFactory:
    """Factory class for creating and managing different types of model instances"""
    
    _model_registry: Dict[ModelType, Type] = {
        ModelType.LLAMA_7B: Llama7BHelper,
        ModelType.LLAMA_3_1_8B: Llama3_1_8BHelper,
        ModelType.QWEN_7B: QwenHelper
    }
    
    @classmethod
    def register_model(cls, model_type: ModelType, model_class: Type) -> None:
        """Register a new model type
        
        Args:
            model_type: Model type enum value
            model_class: Model class
        """
        cls._model_registry[model_type] = model_class
    
    @classmethod
    def create_model(
        cls,
        model_type: ModelType,
        use_local: bool = True,
        local_path: str = "./explanation/models_hf",
        token: str = None,
        **kwargs
    ) -> Union[Llama7BHelper, Llama3_1_8BHelper]:
        """Create a model instance
        
        Args:
            model_type: Type of model to create
            use_local: Whether to use locally cached model
            local_path: Local model path
            token: HuggingFace token (only needed when use_local=False)
            **kwargs: Additional parameters passed to model constructor
            
        Returns:
            Created model instance
            
        Raises:
            ValueError: If model type is not supported
        """
        if model_type not in cls._model_registry:
            raise ValueError(f"Unsupported model type: {model_type}")
            
        model_class = cls._model_registry[model_type]
        return model_class(use_local=use_local, local_path=local_path, token=token, **kwargs)
    
    @classmethod
    def get_supported_models(cls) -> list[str]:
        """Get names of all supported model types"""
        return [model_type.value for model_type in cls._model_registry.keys()] 