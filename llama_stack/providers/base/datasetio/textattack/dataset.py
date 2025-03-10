"""
Base provider interface for TextAttack dataset operations.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Union


class DatasetProvider(ABC):
    """Base provider interface for TextAttack dataset operations."""
    
    @abstractmethod
    def load_dataset(self, dataset_config: Any) -> Any:
        """Load a dataset based on configuration.
        
        Args:
            dataset_config: Dataset configuration
            
        Returns:
            Loaded dataset instance
        """
        pass
    
    @abstractmethod
    def export_dataset(self, dataset: Any, export_config: Any) -> str:
        """Export a dataset based on configuration.
        
        Args:
            dataset: Dataset to export
            export_config: Export configuration
            
        Returns:
            Path or identifier of the exported dataset
        """
        pass