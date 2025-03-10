"""
Remote provider implementation for TextAttack operations.
"""

import os
import json
import requests
from typing import Dict, Any, List, Optional, Union
import tempfile

from llama_stack.datasetio.textattack.config import TextAttackConfig
from llama_stack.providers.base.datasetio.textattack.provider import TextAttackProvider


class RemoteTextAttackProvider(TextAttackProvider):
    """Provider that executes TextAttack operations via a remote API."""
    
    def __init__(self, api_url: str, api_key: Optional[str] = None):
        """Initialize the remote provider.
        
        Args:
            api_url: URL of the TextAttack API
            api_key: Optional API key for authentication
        """
        self.api_url = api_url.rstrip('/')
        self.api_key = api_key
        self.headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    
    def execute_attack(self, config: TextAttackConfig) -> Dict[str, Any]:
        """Execute an attack operation via the remote API.
        
        Args:
            config: Validated TextAttackConfig for an attack operation
            
        Returns:
            Dictionary with attack results
        """
        endpoint = f"{self.api_url}/attack"
        
        # Convert config to dict for API request
        config_dict = config.dict()
        
        # Send request to API
        response = requests.post(
            endpoint,
            json=config_dict,
            headers=self.headers
        )
        
        # Check for errors
        response.raise_for_status()
        
        # Return results
        return response.json()
    
    def execute_augmentation(self, config: TextAttackConfig) -> Dict[str, Any]:
        """Execute a data augmentation operation via the remote API.
        
        Args:
            config: Validated TextAttackConfig for an augmentation operation
            
        Returns:
            Dictionary with augmentation results
        """
        endpoint = f"{self.api_url}/augment"
        
        # Convert config to dict for API request
        config_dict = config.dict()
        
        # Send request to API
        response = requests.post(
            endpoint,
            json=config_dict,
            headers=self.headers
        )
        
        # Check for errors
        response.raise_for_status()
        
        # Get results
        results = response.json()
        
        # If the result includes a dataset to download
        if 'dataset_download_url' in results:
            download_url = results['dataset_download_url']
            
            # Download the dataset
            if config.augmentation.export:
                export_path = config.augmentation.export.path or f"{config.augmentation.export.name}.csv"
                
                # Ensure directory exists
                directory = os.path.dirname(export_path)
                if directory and not os.path.exists(directory):
                    os.makedirs(directory, exist_ok=True)
                
                # Download file
                file_response = requests.get(download_url, headers=self.headers)
                file_response.raise_for_status()
                
                with open(export_path, 'wb') as f:
                    f.write(file_response.content)
                
                results['export_path'] = export_path
                del results['dataset_download_url']
        
        return results