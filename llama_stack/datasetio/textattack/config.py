"""
Pydantic configuration models for TextAttack operations.
"""

import yaml
import importlib
from typing import Dict, Any, List, Optional, Union, Literal, Type, TypeVar
from pydantic import BaseModel, Field, model_validator, field_validator

T = TypeVar('T')

class DynamicComponent(BaseModel):
    """Base model for dynamically loaded components."""
    module: str
    name: str
    params: Optional[Dict[str, Any]] = Field(default_factory=dict)
    
    def create(self):
        """Instantiate the component."""
        # Import the module and get the class
        module = importlib.import_module(self.module)
        cls = getattr(module, self.name)
        
        # Process params to instantiate any nested components
        processed_params = {}
        for key, value in self.params.items():
            if isinstance(value, DynamicComponent):
                processed_params[key] = value.create()
            elif isinstance(value, list):
                # Handle lists of components
                processed_params[key] = [
                    item.create() if isinstance(item, DynamicComponent) else item
                    for item in value
                ]
            else:
                processed_params[key] = value
        
        # Instantiate the class with processed parameters
        return cls(**processed_params)

class TransformationConfig(DynamicComponent):
    """Configuration for a TextAttack transformation."""
    pass

class ConstraintConfig(DynamicComponent):
    """Configuration for a TextAttack constraint."""
    pass

class PreTransformationConstraintConfig(DynamicComponent):
    """Configuration for a TextAttack pre-transformation constraint."""
    pass

class GoalFunctionConfig(DynamicComponent):
    """Configuration for a TextAttack goal function."""
    pass

class SearchMethodConfig(DynamicComponent):
    """Configuration for a TextAttack search method."""
    pass

class ModelConfig(DynamicComponent):
    """Configuration for a TextAttack model wrapper."""
    pass

class AttackRecipeConfig(DynamicComponent):
    """Configuration for a TextAttack attack recipe."""
    pass

class LoggerConfig(DynamicComponent):
    """Configuration for a TextAttack logger."""
    pass

class DatasetFieldMapping(BaseModel):
    """Field mapping for dataset conversion."""
    text_field: str = "text"
    label_field: str = "label"

class StandardDatasetConfig(DynamicComponent):
    """Configuration for a standard TextAttack dataset."""
    pass

class LlamaStackDatasetConfig(BaseModel):
    """Configuration for a dataset from llama-stack."""
    use_llama_stack: bool = True
    name: str
    split: str = "train"
    text_field: str = "text"
    label_field: str = "label"
    
    def create(self):
        """Load a dataset from llama-stack and convert to TextAttack format."""
        try:
            from llama_stack.datasetio import DatasetManager
            from llama_stack.datasetio.loaders import HuggingFaceLoader
            from llama_stack.datasetio.textattack.adapters import DatasetAdapter
        except ImportError:
            raise ImportError("llama-stack is required for this dataset type. "
                             "Install it with: pip install llama-stack")
        
        # Load dataset using llama-stack
        dataset_manager = DatasetManager()
        loader = HuggingFaceLoader(dataset_name=self.name)
        llama_dataset = loader.load(split=self.split)
        
        # Convert to TextAttack format
        adapter = DatasetAdapter()
        return adapter.llama_to_textattack(
            llama_dataset, 
            text_field=self.text_field, 
            label_field=self.label_field
        )

class DatasetConfig(BaseModel):
    """Configuration for a dataset, which can be standard or llama-stack."""
    standard: Optional[StandardDatasetConfig] = None
    llama_stack: Optional[LlamaStackDatasetConfig] = None
    
    @model_validator(mode='before')
    @classmethod
    def check_one_dataset_type(cls, data):
        """Ensure only one dataset type is specified."""
        if isinstance(data, dict):
            if "use_llama_stack" in data and data.get("use_llama_stack"):
                # This is a llama-stack dataset specified in flat format
                llama_config = {
                    "name": data.get("name"),
                    "split": data.get("split", "train"),
                    "text_field": data.get("text_field", "text"),
                    "label_field": data.get("label_field", "label"),
                    "use_llama_stack": True
                }
                return {"llama_stack": llama_config}
            
            if not (("standard" in data) ^ ("llama_stack" in data)):
                raise ValueError("Exactly one of 'standard' or 'llama_stack' must be specified")
        return data
    
    def create(self):
        """Create the appropriate dataset."""
        if self.standard:
            return self.standard.create()
        elif self.llama_stack:
            return self.llama_stack.create()
        else:
            raise ValueError("No dataset configuration provided")

class ExportConfig(BaseModel):
    """Configuration for exporting results."""
    format: Literal["csv", "llama_stack"] = "csv"
    path: Optional[str] = None
    name: Optional[str] = "augmented_dataset"

class AugmentationConfig(BaseModel):
    """Configuration for data augmentation."""
    pct_words_to_swap: float = 0.1
    transformations_per_example: int = 1
    export: Optional[ExportConfig] = None

class AttackParams(BaseModel):
    """Parameters for attack execution."""
    num_examples: int = 10
    shuffle: bool = False
    attack_n: bool = False

class TextAttackConfig(BaseModel):
    """Main configuration for TextAttack operations."""
    operation: Literal["attack", "augmentation"] = "attack"
    attack_type: Optional[Literal["recipe", "custom"]] = None
    transformation: Optional[TransformationConfig] = None
    constraints: List[ConstraintConfig] = Field(default_factory=list)
    pre_transformation_constraints: List[PreTransformationConstraintConfig] = Field(default_factory=list)
    goal_function: Optional[GoalFunctionConfig] = None
    search_method: Optional[SearchMethodConfig] = None
    model: Optional[ModelConfig] = None
    dataset: DatasetConfig
    attack_recipe: Optional[AttackRecipeConfig] = None
    loggers: List[LoggerConfig] = Field(default_factory=list)
    attack_params: AttackParams = Field(default_factory=AttackParams)
    augmentation: Optional[AugmentationConfig] = None
    
    @field_validator('attack_type')
    def validate_attack_type(cls, v, info):
        """Validate attack_type based on operation."""
        values = info.data
        if values.get('operation') == 'attack' and not v:
            return 'custom'  # Default to custom if not specified for attack
        return v
    
    @field_validator('augmentation')
    def validate_augmentation(cls, v, info):
        """Validate that augmentation is present for augmentation operation."""
        values = info.data
        if values.get('operation') == 'augmentation' and not v:
            return AugmentationConfig()  # Default augmentation settings
        return v
    
    @model_validator(mode='after')
    def validate_attack_config(self) -> 'TextAttackConfig':
        """Validate that the configuration is complete based on operation and attack type."""
        operation = self.operation
        attack_type = self.attack_type
        
        if operation == 'attack':
            if attack_type == 'recipe':
                if not self.attack_recipe:
                    raise ValueError("Attack recipe must be specified for attack_type 'recipe'")
                if not self.model:
                    raise ValueError("Model must be specified for attack_type 'recipe'")
            elif attack_type == 'custom':
                required_fields = ['transformation', 'goal_function', 'search_method']
                for field in required_fields:
                    if not getattr(self, field):
                        raise ValueError(f"{field} must be specified for custom attack")
        
        return self


def load_config_from_yaml(config_path: str) -> TextAttackConfig:
    """Load a TextAttackConfig from a YAML file.
    
    Args:
        config_path: Path to YAML configuration file
        
    Returns:
        Validated TextAttackConfig instance
        
    Raises:
        ValueError: If the configuration is invalid
    """
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    
    return TextAttackConfig.parse_obj(config_dict)