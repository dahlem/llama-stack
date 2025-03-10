# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""
Provider factory methods for llama-stack.
"""

import os
from typing import Optional, Dict, Any, Union

# Import base providers
from llama_stack.providers.base.datasetio.textattack.provider import TextAttackProvider
from llama_stack.providers.base.datasetio.textattack.model import ModelProvider
from llama_stack.providers.base.datasetio.textattack.dataset import DatasetProvider

# Import inline providers
from llama_stack.providers.inline.datasetio.textattack.provider import InlineTextAttackProvider
from llama_stack.providers.inline.datasetio.textattack.model import InlineModelProvider
from llama_stack.providers.inline.datasetio.textattack.dataset import InlineDatasetProvider

# Import remote providers
from llama_stack.providers.remote.datasetio.textattack.provider import RemoteTextAttackProvider
from llama_stack.providers.remote.datasetio.textattack.model import RemoteModelProvider
from llama_stack.providers.remote.datasetio.textattack.dataset import RemoteDatasetProvider


def get_textattack_provider(provider_type: str = "inline", **kwargs) -> TextAttackProvider:
    """Get a TextAttack provider based on type.
    
    Args:
        provider_type: Type of provider ('inline' or 'remote')
        **kwargs: Additional arguments for the provider
    
    Returns:
        TextAttackProvider instance
    
    Raises:
        ValueError: If provider_type is not supported
    """
    # Check environment for default provider
    provider_type = os.environ.get("LLAMA_TEXTATTACK_PROVIDER", provider_type)
    
    if provider_type.lower() == "inline":
        return InlineTextAttackProvider()
    elif provider_type.lower() == "remote":
        api_url = kwargs.get("api_url", os.environ.get("LLAMA_TEXTATTACK_API_URL"))
        api_key = kwargs.get("api_key", os.environ.get("LLAMA_TEXTATTACK_API_KEY"))
        
        if not api_url:
            raise ValueError("API URL must be provided for remote provider")
        
        return RemoteTextAttackProvider(api_url=api_url, api_key=api_key)
    else:
        raise ValueError(f"Unsupported provider type: {provider_type}")


def get_textattack_model_provider(provider_type: str = "inline", **kwargs) -> ModelProvider:
    """Get a TextAttack model provider based on type.
    
    Args:
        provider_type: Type of provider ('inline' or 'remote')
        **kwargs: Additional arguments for the provider
    
    Returns:
        ModelProvider instance
    
    Raises:
        ValueError: If provider_type is not supported
    """
    # Check environment for default provider
    provider_type = os.environ.get("LLAMA_TEXTATTACK_MODEL_PROVIDER", provider_type)
    
    if provider_type.lower() == "inline":
        return InlineModelProvider()
    elif provider_type.lower() == "remote":
        api_url = kwargs.get("api_url", os.environ.get("LLAMA_TEXTATTACK_MODEL_API_URL"))
        api_key = kwargs.get("api_key", os.environ.get("LLAMA_TEXTATTACK_MODEL_API_KEY"))
        
        if not api_url:
            raise ValueError("API URL must be provided for remote provider")
        
        return RemoteModelProvider(api_url=api_url, api_key=api_key)
    else:
        raise ValueError(f"Unsupported provider type: {provider_type}")


def get_textattack_dataset_provider(provider_type: str = "inline", **kwargs) -> DatasetProvider:
    """Get a TextAttack dataset provider based on type.
    
    Args:
        provider_type: Type of provider ('inline' or 'remote')
        **kwargs: Additional arguments for the provider
    
    Returns:
        DatasetProvider instance
    
    Raises:
        ValueError: If provider_type is not supported
    """
    # Check environment for default provider
    provider_type = os.environ.get("LLAMA_TEXTATTACK_DATASET_PROVIDER", provider_type)
    
    if provider_type.lower() == "inline":
        return InlineDatasetProvider()
    elif provider_type.lower() == "remote":
        api_url = kwargs.get("api_url", os.environ.get("LLAMA_TEXTATTACK_DATASET_API_URL"))
        api_key = kwargs.get("api_key", os.environ.get("LLAMA_TEXTATTACK_DATASET_API_KEY"))
        
        if not api_url:
            raise ValueError("API URL must be provided for remote provider")
        
        return RemoteDatasetProvider(api_url=api_url, api_key=api_key)
    else:
        raise ValueError(f"Unsupported provider type: {provider_type}")


# Define singleton instances for convenience
default_textattack_provider = get_textattack_provider()
default_model_provider = get_textattack_model_provider()
default_dataset_provider = get_textattack_dataset_provider()

__all__ = [
    # Base interfaces
    "TextAttackProvider",
    "ModelProvider",
    "DatasetProvider",
    
    # Inline providers
    "InlineTextAttackProvider",
    "InlineModelProvider",
    "InlineDatasetProvider",
    
    # Remote providers
    "RemoteTextAttackProvider",
    "RemoteModelProvider",
    "RemoteDatasetProvider",
    
    # Factory methods
    "get_textattack_provider",
    "get_textattack_model_provider",
    "get_textattack_dataset_provider",
    
    # Defaults
    "default_textattack_provider",
    "default_model_provider",
    "default_dataset_provider",
]