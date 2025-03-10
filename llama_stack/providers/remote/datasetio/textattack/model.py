"""
Remote provider implementation for TextAttack model operations.
"""

import requests
from typing import Dict, Any, List, Optional, Union

from llama_stack.providers.base.datasetio.textattack.model import ModelProvider


class RemoteModelProvider(ModelProvider):
    """Provider that accesses TextAttack models via a remote API."""
    
    def __init__(self, api_url: str, api_key: Optional[str] = None):
        """Initialize the remote model provider.
        
        Args:
            api_url: URL of the model API
            api_key: Optional API key for authentication
        """
        self.api_url = api_url.rstrip('/')
        self.api_key = api_key
        self.headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    
    def load_model(self, model_config: Any) -> str:
        """Load a model via the remote API.
        
        Note: This doesn't actually load the model locally, but returns
        an identifier that can be used with the predict method.
        """
        endpoint = f"{self.api_url}/models/load"
        
        # Convert config to dict for API request
        config_dict = model_config.dict() if hasattr(model_config, 'dict') else model_config
        
        # Send request to API
        response = requests.post(
            endpoint,
            json=config_dict,
            headers=self.headers
        )
        
        # Check for errors
        response.raise_for_status()
        
        # Return model identifier
        return response.json().get('model_id')
    
    def predict(self, model_id: str, inputs: List[str]) -> List[Any]:
        """Make predictions with a model via the remote API."""
        endpoint = f"{self.api_url}/models/{model_id}/predict"
        
        # Send request to API
        response = requests.post(
            endpoint,
            json={"inputs": inputs},
            headers=self.headers
        )
        
        # Check for errors
        response.raise_for_status()
        
        # Return predictions
        return response.json().get('predictions', [])