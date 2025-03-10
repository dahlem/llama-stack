#!/usr/bin/env python3
"""

Fixed example script for using TextAttack with llama-stack,
with proper model initialization and configuration handling.
"""

import os
import argparse
import yaml
import textattack
import nltk
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from textattack.datasets import HuggingFaceDataset
from llama_stack.datasetio.textattack import load_config_from_yaml


def download_nltk_resources():
    """Download NLTK resources needed for TextAttack"""
    resources = [
        'punkt',
        'averaged_perceptron_tagger',
        'universal_tagset',
        'stopwords',
        'wordnet',
        'omw-1.4'
    ]
    for resource in resources:
        try:
            nltk.download(resource, quiet=True)
            print(f"NLTK resource '{resource}' is available")
        except Exception as e:
            print(f"Error downloading NLTK resource '{resource}': {e}")


def get_config_value(config_obj, key, default=None):
    """Helper function to get values from config objects, handling both attribute and dict access."""
    if hasattr(config_obj, key):
        return getattr(config_obj, key)
    elif isinstance(config_obj, dict) and key in config_obj:
        return config_obj[key]
    return default

def initialize_model_wrapper(model_config):
    """Initialize a HuggingFace model wrapper from configuration."""
    # Try to get model params in different ways
    params = get_config_value(model_config, 'params', {})
    
    # Get model_name from params
    if isinstance(params, dict):
        model_name = params.get('model_name')
    else:
        model_name = get_config_value(params, 'model_name')
    
    if not model_name:
        # Fallback to direct access in case the structure is different
        model_name = 'distilbert-base-uncased-finetuned-sst-2-english'
        
    print(f"Loading model: {model_name}")
    
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    return textattack.models.wrappers.HuggingFaceModelWrapper(model, tokenizer)


def create_component(component_config, model_wrapper=None):
    """Create a TextAttack component from configuration."""
    if not component_config:
        return None
    
    # Get module and name, handling different config formats
    module_name = get_config_value(component_config, 'module')
    class_name = get_config_value(component_config, 'name')
    
    if not module_name or not class_name:
        print(f"Warning: Invalid component config: {component_config}")
        return None
        
    # Import the module and get the class
    module = __import__(module_name, fromlist=[class_name])
    component_class = getattr(module, class_name)
    
    # Process params - convert to dictionary if it's an object
    params = get_config_value(component_config, 'params', {})
    
    if params and not isinstance(params, dict):
        # Try to convert object to dictionary
        param_dict = {}
        try:
            # Try vars() for object attributes
            for key, value in vars(params).items():
                if not key.startswith('_'):
                    param_dict[key] = value
            params = param_dict
        except (TypeError, AttributeError):
            # If that fails, try direct attribute access for common params
            params = {}
    
    # Special handling for HuggingFaceModelWrapper with model_name parameter
    if module_name == 'textattack.models.wrappers' and class_name == 'HuggingFaceModelWrapper':
        # If model_name is provided, load the actual model and tokenizer
        model_name = get_config_value(params, 'model_name')
        if model_name:
            print(f"Loading HuggingFace model: {model_name}")
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
            model = AutoModelForSequenceClassification.from_pretrained(model_name)
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            # Replace the model_name parameter with actual model and tokenizer
            processed_params = {}
            for k, v in params.items():
                if k != 'model_name':
                    processed_params[k] = v
            processed_params['model'] = model
            processed_params['tokenizer'] = tokenizer
            return component_class(**processed_params)
    
    # Print info about embedding parameters if they exist
    if ((module_name == 'textattack.transformations' and class_name == 'WordSwapEmbedding') or
        (module_name == 'textattack.constraints.semantics' and class_name == 'WordEmbeddingDistance')):
        embedding_type = get_config_value(params, 'embedding_type')
        if embedding_type:
            print(f"Note: embedding_type '{embedding_type}' parameter will be ignored for {class_name} - TextAttack uses default embeddings")
        
    # Create a copy of params, filtering out known unsupported parameters
    filtered_params = {}
    for k, v in params.items():
        # Skip embedding_type parameter for word swap and embedding constraint components
        if k == 'embedding_type' and (
            (module_name == 'textattack.transformations' and class_name == 'WordSwapEmbedding') or
            (module_name == 'textattack.constraints.semantics' and class_name == 'WordEmbeddingDistance')
        ):
            continue
        filtered_params[k] = v
        
    # Handle nested component configurations in params
    processed_params = filtered_params.copy()
    for param_name, param_value in filtered_params.items():
        # Check if this parameter is itself a component configuration
        if isinstance(param_value, dict) and 'module' in param_value and 'name' in param_value:
            # This is a nested component configuration, create it recursively
            print(f"Creating nested component for parameter '{param_name}'")
            nested_component = create_component(param_value, model_wrapper)
            processed_params[param_name] = nested_component
    
    # Special handling for goal function which needs the model_wrapper
    if module_name == 'textattack.goal_functions' and model_wrapper:
        # For goal functions, always use model_wrapper parameter and remove any 'model' parameter
        # Check if 'model' is in params but is actually a model_wrapper
        if 'model' in processed_params:
            print("Converting 'model' parameter to 'model_wrapper' for goal function")
            # If model is a component like HuggingFaceModelWrapper, use it as model_wrapper
            processed_params['model_wrapper'] = processed_params.pop('model')
        else:
            # Use the provided model_wrapper
            processed_params['model_wrapper'] = model_wrapper
    
    # No special handling needed here anymore - handled by the general filter above
            
    # Instantiate the component
    return component_class(**processed_params)


def create_component_list(component_list, model_wrapper=None):
    """Create a list of TextAttack components from configuration."""
    if not component_list:
        return []
    
    # Handle both list and object formats
    if not isinstance(component_list, list):
        try:
            # Try to convert to list if it's an iterable
            component_list = list(component_list)
        except (TypeError, ValueError):
            print(f"Warning: component_list is not a list: {type(component_list)}")
            return []
        
    components = []
    for component in component_list:
        result = create_component(component, model_wrapper)
        if result:
            components.append(result)
            
    return components


def create_dataset(config):
    """Create a TextAttack dataset from configuration."""
    dataset_config = get_config_value(config, 'dataset', {})
    
    # Check for standard dataset configuration
    standard_dataset = get_config_value(dataset_config, 'standard')
    if standard_dataset:
        # Get dataset parameters
        params = get_config_value(standard_dataset, 'params', {})
        
        # Extract required parameters
        name = get_config_value(params, 'name_or_dataset', 'glue')
        subset = get_config_value(params, 'subset', None)
        split = get_config_value(params, 'split', 'train')
        columns = get_config_value(params, 'dataset_columns', None)
        
        print(f"Loading dataset: {name}, subset: {subset}, split: {split}")
        
        try:
            # Import from textattack.datasets
            from textattack.datasets import HuggingFaceDataset
            dataset = HuggingFaceDataset(name, subset, split, dataset_columns=columns)
            print(f"Dataset loaded with {len(dataset)} examples")
            if len(dataset) > 0:
                print(f"Sample data format: {type(dataset[0])}")
            return dataset
        except ValueError as e:
            # Check if the error is related to dataset configuration
            error_str = str(e)
            if "BuilderConfig" in error_str and subset:
                print(f"Error loading dataset with subset '{subset}': {error_str}")
                print("Trying without subset...")
                # Try loading without the subset
                return HuggingFaceDataset(name, None, split, dataset_columns=columns)
            raise
    
    # Check for llama_stack dataset configuration
    llama_stack_dataset = get_config_value(dataset_config, 'llama_stack')
    if llama_stack_dataset:
        # Convert llama_stack dataset to HuggingFaceDataset
        print("Converting llama_stack dataset to standard HuggingFaceDataset")
        
        # Extract required parameters
        name = get_config_value(llama_stack_dataset, 'name', 'sst2')
        split = get_config_value(llama_stack_dataset, 'split', 'train')
        text_field = get_config_value(llama_stack_dataset, 'text_field', 'sentence')
        label_field = get_config_value(llama_stack_dataset, 'label_field', 'label')
        
        # Map common dataset names to their HuggingFace equivalents
        dataset_mapping = {
            'sst2': ('glue', 'sst2'),
            'imdb': ('imdb', None),
            'yelp': ('yelp_polarity', None),
            'agnews': ('ag_news', None),
            'snli': ('snli', None),
            'mnli': ('glue', 'mnli'),
            'cola': ('glue', 'cola'),
            'rte': ('glue', 'rte')
        }
        
        # Get the appropriate dataset name and subset
        if name.lower() in dataset_mapping:
            hf_name, hf_subset = dataset_mapping[name.lower()]
        else:
            # If not in mapping, just use the name directly
            hf_name = name
            hf_subset = None
        
        print(f"Loading dataset: {hf_name}, subset: {hf_subset}, split: {split}")
        
        # Create dataset columns based on text and label fields
        if isinstance(text_field, list):
            columns = [text_field, label_field]
        else:
            columns = [[text_field], label_field]
        
        try:
            # Import from textattack.datasets
            from textattack.datasets import HuggingFaceDataset
            dataset = HuggingFaceDataset(hf_name, hf_subset, split, dataset_columns=columns)
            print(f"Dataset loaded with {len(dataset)} examples")
            if len(dataset) > 0:
                print(f"Sample data format: {type(dataset[0])}")
            return dataset
        except ValueError as e:
            print(f"Error loading dataset: {e}")
            print("Falling back to default SST-2 dataset")
            return HuggingFaceDataset('glue', 'sst2', 'train', 
                                     dataset_columns=[[text_field], label_field])
    
    # Default to SST-2 if no valid configuration found
    print("Warning: No valid dataset configuration found, defaulting to SST-2")
    from textattack.datasets import HuggingFaceDataset
    return HuggingFaceDataset('glue', 'sst2', 'train', 
                             dataset_columns=[['sentence'], 'label'])


def create_loggers(config):
    """Create TextAttack loggers from configuration."""
    loggers = []
    logger_configs = get_config_value(config, 'loggers', [])
    
    if not logger_configs:
        # Default to a CSV logger if none specified
        print("No loggers configured, defaulting to CSV logger")
        os.makedirs("results/textattack", exist_ok=True)
        return [textattack.loggers.CSVLogger(filename="results/textattack/attack_results.csv")]
    
    for logger_config in logger_configs:
        # Get module and name
        module_name = get_config_value(logger_config, 'module')
        class_name = get_config_value(logger_config, 'name')
        
        if not module_name or not class_name:
            continue
            
        # Get params
        params = get_config_value(logger_config, 'params', {})
        
        # Create directory for logger if filename is specified
        filename = get_config_value(params, 'filename')
        if filename:
            os.makedirs(os.path.dirname(filename), exist_ok=True)
        
        # Convert params to dictionary if it's not already
        if params and not isinstance(params, dict):
            param_dict = {}
            try:
                for key, value in vars(params).items():
                    if not key.startswith('_'):
                        param_dict[key] = value
                params = param_dict
            except (TypeError, AttributeError):
                params = {}
        
        # Create logger
        try:
            module = __import__(module_name, fromlist=[class_name])
            logger_class = getattr(module, class_name)
            loggers.append(logger_class(**params))
        except (ImportError, AttributeError) as e:
            print(f"Error creating logger {module_name}.{class_name}: {e}")
        
    return loggers


def run_attack(config_path, verbose=True):
    """Run a TextAttack attack with proper model initialization.
    
    Args:
        config_path: Path to the YAML attack configuration
        verbose: Whether to print verbose output
    """
    # Create results directory if needed
    os.makedirs("results/textattack", exist_ok=True)
    
    # Load configuration
    print(f"Loading attack configuration from {config_path}")
    config = load_config_from_yaml(config_path)
    
    # Initialize model wrapper
    model_wrapper = initialize_model_wrapper(config)
    
    # Handle different attack types
    attack_type = get_config_value(config, 'attack_type')
    if attack_type == 'recipe':
        # Recipe-based attack
        attack_recipe = get_config_value(config, 'attack_recipe')
        if attack_recipe:
            module_name = get_config_value(attack_recipe, 'module')
            class_name = get_config_value(attack_recipe, 'name')
            
            print(f"Initializing recipe: {module_name}.{class_name}")
            
            # Special handling for TextFoolerJin2019 which has NLTK issues
            if class_name == 'TextFoolerJin2019':
                try:
                    # Import required modules
                    from textattack.goal_functions import UntargetedClassification
                    from textattack.constraints.semantics import WordEmbeddingDistance
                    from textattack.constraints.semantics.sentence_encoders import UniversalSentenceEncoder
                    from textattack.constraints.pre_transformation import RepeatModification, StopwordModification
                    from textattack.transformations import WordSwapEmbedding
                    from textattack.search_methods import GreedyWordSwapWIR
                    
                    # Create the goal function
                    goal_function = UntargetedClassification(model_wrapper)
                    
                    # Create constraints - skip the problematic PartOfSpeech constraint
                    constraints = [
                        RepeatModification(),  # No repeating words
                        StopwordModification(),  # No swapping stopwords
                        WordEmbeddingDistance(min_cos_sim=0.5),  # Embedding constraint
                        UniversalSentenceEncoder(threshold=0.840845057),  # USE constraint
                    ]
                    
                    # Create transformation
                    transformation = WordSwapEmbedding(max_candidates=50)
                    
                    # Create search method
                    search_method = GreedyWordSwapWIR(wir_method="delete")
                    
                    # Build attack
                    from textattack import Attack
                    attack = Attack(goal_function, constraints, transformation, search_method)
                    print("Created custom TextFooler attack without problematic NLTK components")
                except Exception as e:
                    print(f"Error creating custom TextFooler attack: {e}")
                    # Continue with the normal recipe loading
                    try:
                        recipe_module = __import__(module_name, fromlist=[class_name])
                        recipe_class = getattr(recipe_module, class_name)
                        
                        # Check if this is a recipe class or an Attack subclass
                        if hasattr(recipe_class, 'build'):
                            # This is a recipe with a build method
                            recipe = recipe_class.build(model_wrapper)
                            attack = recipe
                        else:
                            # This might be a direct Attack subclass
                            # Try instantiating it directly with required components
                            from textattack.attack_recipes import AttackRecipe
                            if issubclass(recipe_class, AttackRecipe):
                                # It's a recipe class, so instantiate and build
                                recipe = recipe_class()
                                attack = recipe.build(model_wrapper)
                            else:
                                raise TypeError(f"{class_name} is not a valid AttackRecipe")
                    except Exception as e2:
                        raise ValueError(f"Failed to build TextFooler recipe: {e} -> {e2}")
            else:
                # For other recipes, use standard approach
                try:
                    recipe_module = __import__(module_name, fromlist=[class_name])
                    recipe_class = getattr(recipe_module, class_name)
                    
                    # Check if this is a recipe class or an Attack subclass
                    if hasattr(recipe_class, 'build'):
                        # This is a recipe with a build method
                        recipe = recipe_class.build(model_wrapper)
                        attack = recipe
                    else:
                        # This might be a direct Attack subclass
                        # Try instantiating it directly with required components
                        from textattack.attack_recipes import AttackRecipe
                        if issubclass(recipe_class, AttackRecipe):
                            # It's a recipe class, so instantiate and build
                            recipe = recipe_class()
                            attack = recipe.build(model_wrapper)
                        else:
                            raise TypeError(f"{class_name} is not a valid AttackRecipe")
                except Exception as e:
                    # Try using a pre-built recipe constructor
                    print(f"Error loading recipe: {e}")
                    print("Attempting alternative recipe construction method...")
                    try:
                        from textattack.attack_recipes import TextFoolerJin2019, PWWSRen2019, HotFlipEbrahimi2017
                        # Map common recipe names to their classes
                        recipe_map = {
                            'TextFoolerJin2019': TextFoolerJin2019,
                            'PWWSRen2019': PWWSRen2019,
                            'HotFlipEbrahimi2017': HotFlipEbrahimi2017
                        }
                        if class_name in recipe_map:
                            recipe_class = recipe_map[class_name]
                            attack = recipe_class.build(model_wrapper)
                            print(f"Successfully built {class_name} recipe")
                        else:
                            raise ValueError(f"Recipe {class_name} not found in built-in recipes")
                    except Exception as e2:
                        raise ValueError(f"Failed to build recipe: {e2} (after original error: {e})")
        else:
            raise ValueError("Recipe attack specified but no attack_recipe found in config")
    else:
        # Custom attack
        print("Building custom attack...")
        
        # Create goal function with model wrapper
        goal_function = create_component(get_config_value(config, 'goal_function'), model_wrapper)
        if not goal_function:
            raise ValueError("Failed to create goal function")
        
        # Create transformation
        transformation = create_component(get_config_value(config, 'transformation'))
        if not transformation:
            raise ValueError("Failed to create transformation")
        
        # Create constraints
        constraints = create_component_list(get_config_value(config, 'constraints', []), model_wrapper)
        
        # Add pre-transformation constraints if they exist
        pre_constraints = get_config_value(config, 'pre_transformation_constraints', [])
        if pre_constraints:
            constraints.extend(create_component_list(pre_constraints, model_wrapper))
        
        # Create search method
        search_method = create_component(get_config_value(config, 'search_method'))
        if not search_method:
            raise ValueError("Failed to create search method")
        
        # Create attack
        attack = textattack.Attack(
            goal_function=goal_function,
            constraints=constraints,
            transformation=transformation,
            search_method=search_method
        )
    
    # Create dataset
    dataset = create_dataset(config)
    
    # Create attack args
    attack_params = get_config_value(config, 'attack_params', {})
    
    # Extract parameters
    if isinstance(attack_params, dict):
        num_examples = attack_params.get('num_examples', 10)
        shuffle = attack_params.get('shuffle', True)
    else:
        num_examples = get_config_value(attack_params, 'num_examples', 10)
        shuffle = get_config_value(attack_params, 'shuffle', True)
    
    attack_args = textattack.AttackArgs(
        num_examples=num_examples,
        shuffle=shuffle
    )
    
    # Create loggers
    loggers = create_loggers(config)
    attack_args.loggers = loggers
    
    # Create attacker
    attacker = textattack.Attacker(attack, dataset, attack_args)
    
    # Run attack
    print("Running attack...")
    results = attacker.attack_dataset()
    
    # Print results
    if verbose and results:
        print("\nAttack Results (from fixed_textattack_example.py):")
        # The results might be a list of individual results or a single results object
        if isinstance(results, list):
            # Count successful attacks from the list
            total = len(results)
            successful = sum(1 for r in results if hasattr(r, 'perturbed_result') and r.perturbed_result is not None)
            success_rate = successful / total if total > 0 else 0
            print(f"Total examples: {total}")
            print(f"Successful attacks: {successful}")
            print(f"Success rate: {success_rate:.2f}")
        else:
            # Handle the case where results is a single object
            print(f"Total examples: {getattr(results, 'total_attacked', 0)}")
            print(f"Successful attacks: {getattr(results, 'successful_attacks', 0)}")
            print(f"Success rate: {getattr(results, 'success_rate', 0):.2f}")
            print(f"Average perturbed word %: {getattr(results, 'avg_word_perturb_percent', 0):.2f}")
    
    return results


def run_augmentation(config_path, verbose=True):
    """Run a TextAttack augmentation with proper model initialization.
    
    Args:
        config_path: Path to the YAML augmentation configuration
        verbose: Whether to print verbose output
    """
    # Create results directory if needed
    os.makedirs("results/textattack", exist_ok=True)
    
    # Load configuration
    print(f"Loading augmentation configuration from {config_path}")
    config = load_config_from_yaml(config_path)
    
    # Create transformation
    transformation_config = get_config_value(config, 'transformation')
    if not transformation_config:
        raise ValueError("No transformation specified in the augmentation configuration")
    
    transformation = create_component(transformation_config)
    if not transformation:
        raise ValueError("Failed to create transformation")
    
    # Create dataset
    dataset = create_dataset(config)
    
    # Get augmentation parameters
    augmentation_config = get_config_value(config, 'augmentation', {})
    transformations_per_example = get_config_value(augmentation_config, 'transformations_per_example', 1)
    
    # Import augmentation modules
    from textattack.augmentation import Augmenter
    
    # Create augmenter
    augmenter = Augmenter(
        transformation=transformation,
        transformations_per_example=transformations_per_example
    )
    
    # Run augmentation
    print("Running augmentation...")
    
    # Inspect and convert dataset to format expected by augmenter
    texts = []
    labels = []
    
    # Get a sample to understand the structure
    if len(dataset) > 0:
        sample_item = dataset[0]
        print(f"Dataset item structure: {type(sample_item)}")
        if isinstance(sample_item, tuple):
            print(f"Item[0] type: {type(sample_item[0])}")
    
    # Extract text and labels
    for item in dataset:
        if isinstance(item, tuple) and len(item) >= 2:
            text_item, label = item[0], item[1]
            
            # The text might be an OrderedDict with a key like 'sentence' or 'text'
            if hasattr(text_item, 'get'):
                # Try common text field names
                for field in ['sentence', 'text', 'premise', 'hypothesis']:
                    if field in text_item:
                        text = text_item[field]
                        texts.append(text)
                        labels.append(label)
                        break
                else:
                    print(f"Warning: Could not find text field in {text_item.keys()}")
            elif isinstance(text_item, str):
                # If it's already a string, just use it
                texts.append(text_item)
                labels.append(label)
            else:
                print(f"Warning: Unsupported text item type: {type(text_item)}")
        else:
            print(f"Warning: Skipping unexpected dataset item format: {item}")
            
    print(f"Extracted {len(texts)} text examples from dataset")
    
    # Only run augmentation if we have texts
    if not texts:
        raise ValueError("No valid text examples found in dataset")
        
    # Limit the number of examples to process during testing
    max_examples = 50  # Set a lower number for testing
    if len(texts) > max_examples:
        print(f"Limiting to {max_examples} examples for testing")
        texts = texts[:max_examples]
        labels = labels[:max_examples]
    
    # Run augmentation on each text individually then combine results
    print(f"Augmenting {len(texts)} examples with {transformations_per_example} transformations each...")
    augmented_texts = []
    
    # Process in smaller batches to show progress
    batch_size = 10
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        print(f"Processing batch {i//batch_size + 1}/{(len(texts) + batch_size - 1)//batch_size}...")
        
        for text in batch:
            # Make sure text is a string
            if not isinstance(text, str):
                print(f"Warning: Skipping non-string input: {text}")
                continue
                
            # Augment this example
            try:
                augmented = augmenter.augment(text)
                augmented_texts.extend(augmented)
            except Exception as e:
                print(f"Error augmenting text: {e}")
    
    # Process results
    if verbose:
        print(f"\nAugmentation Results:")
        print(f"Original examples: {len(texts)}")
        print(f"Augmented examples: {len(augmented_texts)}")
        print(f"Expansion factor: {len(augmented_texts) / len(texts):.2f}")
    
    # Handle export if configured
    export_config = get_config_value(augmentation_config, 'export', {})
    export_path = get_config_value(export_config, 'path')
    export_format = get_config_value(export_config, 'format', 'csv')
    
    if export_path:
        output_dir = os.path.dirname(export_path)
        os.makedirs(output_dir, exist_ok=True)
        
        if export_format.lower() == 'csv':
            import csv
            with open(export_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['text', 'label'])
                
                # Write original examples
                for text, label in zip(texts, labels):
                    writer.writerow([text, label])
                
                # Write augmented examples
                for i, text in enumerate(augmented_texts):
                    # Use same label as the original example it was derived from
                    original_idx = i // transformations_per_example
                    if original_idx < len(labels):
                        writer.writerow([text, labels[original_idx]])
            
            print(f"Augmented dataset exported to: {export_path}")
        else:
            print(f"Unsupported export format: {export_format}")
    
    return {
        "original_examples": len(texts),
        "augmented_examples": len(augmented_texts),
        "expansion_factor": len(augmented_texts) / len(texts) if texts else 0,
        "export_path": export_path if export_path else None
    }


def main():
    """Main function to parse arguments and run commands."""
    # Create results directory if needed
    os.makedirs("results/textattack", exist_ok=True)
    
    # Ensure NLTK resources are downloaded
    download_nltk_resources()
    
    parser = argparse.ArgumentParser(description="Run TextAttack operations with fixed implementation")
    parser.add_argument("--operation", choices=["attack", "augmentation"], required=False,
                      help="Type of operation to perform (optional, will be detected from config file if not provided)")
    parser.add_argument("--config", required=True,
                      help="Path to YAML configuration file")
    
    args = parser.parse_args()
    
    # Create results directory if needed
    os.makedirs("results/textattack", exist_ok=True)
    
    # If operation not specified, detect it from the configuration
    if not args.operation:
        # Load the config to get the operation type
        config = load_config_from_yaml(args.config)
        operation = get_config_value(config, 'operation', 'attack').lower()
        
        if operation not in ['attack', 'augmentation']:
            print(f"Warning: Unknown operation '{operation}' in config, defaulting to 'attack'")
            operation = 'attack'
        print(f"Detected operation from config file: {operation}")
    else:
        operation = args.operation
    
    # Run the appropriate operation
    if operation == "attack":
        run_attack(args.config)
    elif operation == "augmentation":
        run_augmentation(args.config)


if __name__ == "__main__":
    main()
