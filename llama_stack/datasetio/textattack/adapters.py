"""
Adapter classes for converting between llama-stack and TextAttack formats.
"""

from typing import Any, Dict, List, Optional, Tuple, Union

class DatasetAdapter:
    """Adapter for converting between llama-stack and TextAttack datasets."""
    
    def llama_to_textattack(self, 
                           llama_dataset: Any, 
                           text_field: str = "text", 
                           label_field: str = "label") -> Any:
        """Convert a llama-stack dataset to TextAttack format.
        
        Args:
            llama_dataset: llama-stack Dataset instance
            text_field: Field name containing text data
            label_field: Field name containing label data
            
        Returns:
            TextAttack Dataset object
        """
        import textattack
        
        examples = []
        for item in llama_dataset:
            if text_field in item and label_field in item:
                examples.append((item[text_field], item[label_field]))
        
        return textattack.datasets.Dataset(examples)
    
    def textattack_to_llama(self, 
                           textattack_dataset: Any, 
                           name: str = "converted_dataset",
                           text_field: str = "text", 
                           label_field: str = "label") -> Any:
        """Convert a TextAttack dataset to llama-stack format.
        
        Args:
            textattack_dataset: TextAttack Dataset instance
            name: Name for the new llama-stack dataset
            text_field: Field name to use for text data
            label_field: Field name to use for label data
            
        Returns:
            llama-stack Dataset object
        """
        from llama_stack.datasetio import Dataset as LlamaDataset
        
        data = []
        for text, label in textattack_dataset:
            item = {text_field: text, label_field: label}
            data.append(item)
        
        return LlamaDataset(data, name=name)


class ResultsAdapter:
    """Adapter for converting TextAttack results to llama-stack format."""
    
    def attack_results_to_df(self, results: List[Any]) -> Any:
        """Convert TextAttack attack results to pandas DataFrame.
        
        Args:
            results: List of AttackResult objects
            
        Returns:
            pandas DataFrame
        """
        import pandas as pd
        
        data = []
        for result in results:
            data.append({
                "original_text": result.original_text(),
                "perturbed_text": result.perturbed_text() if result.perturbed_text() else None,
                "original_output": result.original_output,
                "perturbed_output": result.perturbed_output if hasattr(result, "perturbed_output") else None,
                "ground_truth_output": result.ground_truth_output,
                "success": result.successful,
                "num_queries": result.num_queries,
            })
        
        return pd.DataFrame(data)
    
    def augmentation_results_to_df(self, original_dataset: Any, 
                                  augmented_dataset: Any) -> Any:
        """Convert augmentation results to pandas DataFrame.
        
        Args:
            original_dataset: Original TextAttack dataset
            augmented_dataset: Augmented TextAttack dataset
            
        Returns:
            pandas DataFrame
        """
        import pandas as pd
        
        original_data = [{"text": text, "label": label, "source": "original"} 
                         for text, label in original_dataset]
        
        augmented_data = [{"text": text, "label": label, "source": "augmented"} 
                          for text, label in augmented_dataset]
        
        combined_data = original_data + augmented_data
        return pd.DataFrame(combined_data)