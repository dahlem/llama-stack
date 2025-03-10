"""
Inline provider implementation for TextAttack operations.
"""

import os
from typing import Dict, Any, List, Optional, Union
import textattack
from textattack.datasets import Dataset as TextAttackDataset

from llama_stack.datasetio.textattack.config import TextAttackConfig, ExportConfig
from llama_stack.datasetio.textattack.adapters import DatasetAdapter
from llama_stack.providers.base.datasetio.textattack.provider import TextAttackProvider


class InlineTextAttackProvider(TextAttackProvider):
    """Provider that executes TextAttack operations directly."""
    
    def __init__(self):
        """Initialize the inline provider."""
        self.adapter = DatasetAdapter()
        # Import here to avoid circular imports
        from llama_stack.providers.inline.datasetio.textattack.model import InlineModelProvider
        from llama_stack.providers.inline.datasetio.textattack.dataset import InlineDatasetProvider
        
        self.model_provider = InlineModelProvider()
        self.dataset_provider = InlineDatasetProvider()
    
    def execute_attack(self, config: TextAttackConfig) -> Dict[str, Any]:
        """Execute an attack operation directly using TextAttack.
        
        Args:
            config: Validated TextAttackConfig for an attack operation
            
        Returns:
            Dictionary with attack results
        """
        # Build the attack
        attack = self._build_attack(config)
        
        # Load dataset
        dataset = self.dataset_provider.load_dataset(config.dataset)
        
        # Create loggers
        loggers = self._create_loggers(config)
        
        # Create attacker
        attacker = textattack.Attacker(
            attack=attack,
            dataset=dataset,
            attack_args=textattack.AttackArgs(
                num_examples=config.attack_params.num_examples,
                shuffle=config.attack_params.shuffle,
                attack_n=config.attack_params.attack_n
            )
        )
        
        # Add loggers
        for logger in loggers:
            attacker.attack_args.loggers.append(logger)
        
        # Run the attack
        results = attacker.attack_dataset()
        
        # Return results summary
        return {
            "total_examples": len(results),
            "successful_attacks": sum(1 for r in results if r.successful),
            "success_rate": sum(1 for r in results if r.successful) / len(results) if results else 0,
            "average_perturbed_word_percentage": sum(r.perturbed_word_percentage for r in results if r.successful) / sum(1 for r in results if r.successful) if any(r.successful for r in results) else 0,
        }
    
    def execute_augmentation(self, config: TextAttackConfig) -> Dict[str, Any]:
        """Execute a data augmentation operation directly using TextAttack.
        
        Args:
            config: Validated TextAttackConfig for an augmentation operation
            
        Returns:
            Dictionary with augmentation results
        """
        transformation = config.transformation.create()
        constraints = [c.create() for c in config.constraints]
        dataset = self.dataset_provider.load_dataset(config.dataset)
        
        # Create augmenter
        augmenter = textattack.augmentation.Augmenter(
            transformation=transformation,
            constraints=constraints,
            pct_words_to_swap=config.augmentation.pct_words_to_swap,
            transformations_per_example=config.augmentation.transformations_per_example
        )
        
        # Augment dataset
        augmented_texts = []
        for text, label in dataset:
            augmented_texts.extend(
                [(augmented_text, label) for augmented_text in augmenter.augment(text)]
            )
        
        # Create augmented dataset
        augmented_dataset = textattack.datasets.Dataset(augmented_texts)
        
        # Export if specified
        export_path = None
        if config.augmentation.export:
            export_path = self.dataset_provider.export_dataset(
                augmented_dataset, 
                config.augmentation.export
            )
        
        return {
            "original_examples": len(dataset),
            "augmented_examples": len(augmented_dataset),
            "expansion_factor": len(augmented_dataset) / len(dataset) if len(dataset) > 0 else 0,
            "export_path": export_path
        }
    
    def _build_attack(self, config: TextAttackConfig) -> textattack.Attack:
        """Build a TextAttack attack based on the configuration."""
        if config.attack_type == 'recipe':
            recipe = config.attack_recipe.create()
            model = self.model_provider.load_model(config.model)
            return recipe.build(model)
        
        # Custom attack
        goal_function = config.goal_function.create()
        if hasattr(goal_function, 'model') and not goal_function.model:
            # If the goal function has a model attribute but it's not set,
            # we need to load the model and set it
            model = self.model_provider.load_model(config.model)
            goal_function.model = model
        
        return textattack.Attack(
            goal_function=goal_function,
            constraints=[c.create() for c in config.constraints],
            transformation=config.transformation.create(),
            search_method=config.search_method.create(),
            pre_transformation_constraints=[c.create() for c in config.pre_transformation_constraints]
        )
    
    def _create_loggers(self, config: TextAttackConfig) -> List[textattack.loggers.Logger]:
        """Create logger instances."""
        # Ensure output directories exist
        for logger_config in config.loggers:
            if 'filename' in logger_config.params:
                output_dir = os.path.dirname(logger_config.params['filename'])
                if output_dir and not os.path.exists(output_dir):
                    os.makedirs(output_dir, exist_ok=True)
        
        return [logger_config.create() for logger_config in config.loggers]