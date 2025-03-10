"""
Base provider interface for TextAttack model operations.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Union


class ModelProvider(ABC):
    """Base provider interface for TextAttack model operations."""
    
    @abstractmethod
    def load_model(self, model_config: Any) -> Any:
        """Load a model based on configuration.
        
        Args:
            model_config: Model configuration
            
        Returns:
            Loaded model instance
        """
        pass
    
    @abstractmethod
    def predict(self, model: Any, inputs: List[str]) -> List[Any]:
        """Make predictions with a loaded model.
        
        Args:
            model: Loaded model instance
            inputs: List of input texts
            
        Returns:
            List of predictions
        """
        pass