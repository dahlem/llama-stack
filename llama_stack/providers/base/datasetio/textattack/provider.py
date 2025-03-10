"""
Base provider interface for TextAttack operations.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Union

from llama_stack.datasetio.textattack.config import TextAttackConfig


class TextAttackProvider(ABC):
    """Base provider interface for TextAttack operations."""
    
    @abstractmethod
    def execute_attack(self, config: TextAttackConfig) -> Dict[str, Any]:
        """Execute an attack operation.
        
        Args:
            config: Validated TextAttackConfig for an attack operation
            
        Returns:
            Dictionary with attack results
        """
        pass
    
    @abstractmethod
    def execute_augmentation(self, config: TextAttackConfig) -> Dict[str, Any]:
        """Execute a data augmentation operation.
        
        Args:
            config: Validated TextAttackConfig for an augmentation operation
            
        Returns:
            Dictionary with augmentation results
        """
        pass
    
    def execute(self, config: TextAttackConfig) -> Dict[str, Any]:
        """Execute the appropriate operation based on the config.
        
        Args:
            config: Validated TextAttackConfig
            
        Returns:
            Dictionary with operation results
        """
        if config.operation == 'attack':
            return self.execute_attack(config)
        elif config.operation == 'augmentation':
            return self.execute_augmentation(config)
        else:
            raise ValueError(f"Unknown operation: {config.operation}")