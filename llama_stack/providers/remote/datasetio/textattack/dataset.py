"""
Remote provider implementation for TextAttack dataset operations.
"""

import os
import requests
from typing import Dict, Any, List, Optional, Union

from llama_stack.providers.base.datasetio.textattack.dataset import DatasetProvider
from llama_stack.datasetio.textattack.config import ExportConfig


class RemoteDatasetProvider(DatasetProvider):
    """Provider that accesses TextAttack datasets via a remote API."""
    
    def __init__(self, api_url: str, api_key: Optional[str] = None):
        """Initialize the remote dataset provider.
        
        Args:
            api_url: URL of the dataset API
            api_key: Optional API key for authentication
        """
        self.api_url = api_url.rstrip('/')
        self.api_key = api_key
        self.headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    
    def load_dataset(self, dataset_config: Any) -> str:
        """Load a dataset via the remote API.
        
        Note: This doesn't actually load the dataset locally, but returns
        an identifier that can be used with the remote API.
        """
        endpoint = f"{self.api_url}/datasets/load"
        
        # Convert config to dict for API request
        config_dict = dataset_config.dict() if hasattr(dataset_config, 'dict') else dataset_config
        
        # Send request to API
        response = requests.post(
            endpoint,
            json=config_dict,
            headers=self.headers
        )
        
        # Check for errors
        response.raise_for_status()
        
        # Return dataset identifier
        return response.json().get('dataset_id')
    
    def export_dataset(self, dataset_id: str, export_config: ExportConfig) -> str:
        """Export a dataset via the remote API."""
        endpoint = f"{self.api_url}/datasets/{dataset_id}/export"
        
        # Convert config to dict for API request
        config_dict = export_config.dict() if hasattr(export_config, 'dict') else export_config
        
        # Send request to API
        response = requests.post(
            endpoint,
            json=config_dict,
            headers=self.headers
        )
        
        # Check for errors
        response.raise_for_status()
        
        # Get download URL from response
        download_url = response.json().get('download_url')
        
        if download_url and export_config.path:
            # Download the exported dataset
            file_response = requests.get(download_url, headers=self.headers)
            file_response.raise_for_status()
            
            # Ensure directory exists
            directory = os.path.dirname(export_config.path)
            if directory and not os.path.exists(directory):
                os.makedirs(directory, exist_ok=True)
            
            # Save file
            with open(export_config.path, 'wb') as f:
                f.write(file_response.content)
            
            return export_config.path
        
        return download_url or ""