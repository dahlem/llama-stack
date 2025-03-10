"""
TextAttack integration for llama-stack.

This module provides integration between TextAttack and llama-stack,
allowing for configuration-driven NLP attacks and augmentation.
"""

from .config import TextAttackConfig, load_config_from_yaml
from .adapters import DatasetAdapter, ResultsAdapter

# Convenience function that uses the default provider
def execute_textattack(config: TextAttackConfig):
    """Execute a TextAttack operation using the default provider.
    
    Args:
        config: Validated TextAttackConfig
        
    Returns:
        Results from the operation
    """
    # Import here to avoid circular imports
    from llama_stack.providers import default_textattack_provider
    return default_textattack_provider.execute(config)


def get_textattack_provider(provider_type: str = "inline", **kwargs):
    """Get a TextAttack provider based on type.
    
    Args:
        provider_type: Type of provider ('inline' or 'remote')
        **kwargs: Additional arguments for the provider
    
    Returns:
        TextAttackProvider instance
    """
    # Import here to avoid circular imports
    from llama_stack.providers import get_textattack_provider as provider_factory
    return provider_factory(provider_type, **kwargs)

__all__ = [
    # Configuration
    "TextAttackConfig",
    "load_config_from_yaml",
    
    # Adapters
    "DatasetAdapter",
    "ResultsAdapter",
    
    # Functions
    "execute_textattack",
    "get_textattack_provider",
]