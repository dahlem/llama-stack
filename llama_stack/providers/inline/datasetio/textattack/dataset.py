"""
Inline provider implementation for TextAttack dataset operations.
"""

import os
from typing import Dict, Any, List, Optional, Union
import textattack
from textattack.datasets import Dataset as TextAttackDataset

from llama_stack.providers.base.datasetio.textattack.dataset import DatasetProvider
from llama_stack.datasetio.textattack.config import ExportConfig
from llama_stack.datasetio.textattack.adapters import DatasetAdapter


class InlineDatasetProvider(DatasetProvider):
    """Provider that loads and exports TextAttack datasets directly."""
    
    def __init__(self):
        """Initialize the dataset provider."""
        self.adapter = DatasetAdapter()
    
    def load_dataset(self, dataset_config: Any) -> TextAttackDataset:
        """Load a dataset based on configuration."""
        return dataset_config.create()
    
    def export_dataset(self, dataset: TextAttackDataset, export_config: ExportConfig) -> str:
        """Export a dataset based on configuration."""
        if export_config.format == 'csv':
            path = export_config.path or f"{export_config.name}.csv"
            
            # Ensure directory exists
            directory = os.path.dirname(path)
            if directory and not os.path.exists(directory):
                os.makedirs(directory, exist_ok=True)
                
            with open(path, 'w') as f:
                f.write('text,label\n')
                for text, label in dataset:
                    # Escape quotes in text
                    escaped_text = text.replace('"', '""')
                    f.write(f'"{escaped_text}",{label}\n')
            
            return path
        
        elif export_config.format == 'llama_stack':
            # Convert to llama-stack format and save
            llama_dataset = self.adapter.textattack_to_llama(dataset, export_config.name)
            
            from llama_stack.datasetio import DatasetManager
            dataset_manager = DatasetManager()
            dataset_manager.save(llama_dataset)
            
            return export_config.name
        
        return ""