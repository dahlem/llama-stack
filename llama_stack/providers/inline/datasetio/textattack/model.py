"""
Inline provider implementation for TextAttack model operations.
"""

from typing import Dict, Any, List, Optional, Union

from llama_stack.providers.base.datasetio.textattack.model import ModelProvider


class InlineModelProvider(ModelProvider):
    """Provider that loads and uses TextAttack models directly."""
    
    def load_model(self, model_config: Any) -> Any:
        """Load a model based on configuration."""
        return model_config.create()
    
    def predict(self, model: Any, inputs: List[str]) -> List[Any]:
        """Make predictions with a loaded model."""
        return [model.predict(input_text) for input_text in inputs]