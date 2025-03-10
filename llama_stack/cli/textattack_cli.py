"""
Command-line interface for TextAttack integration with llama-stack.
"""

import argparse
import sys
import json
import yaml
from typing import Dict, Any
import os

from .config import TextAttackConfig, load_config_from_yaml
from .executor import TextAttackExecutor


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="llama-stack TextAttack integration",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Run command
    run_parser = subparsers.add_parser(
        "run", help="Run a TextAttack operation from a YAML config"
    )
    run_parser.add_argument(
        "--config", type=str, required=True, help="Path to YAML configuration file"
    )
    run_parser.add_argument(
        "--output", type=str, help="Path to save results as JSON (optional)"
    )
    
    # Generate config template command
    template_parser = subparsers.add_parser(
        "generate-template", help="Generate a configuration template"
    )
    template_parser.add_argument(
        "--type", choices=["attack", "augmentation"], default="attack",
        help="Type of template to generate"
    )
    template_parser.add_argument(
        "--output", type=str, required=True, help="Path to save template"
    )
    
    # Validate config command
    validate_parser = subparsers.add_parser(
        "validate", help="Validate a configuration file"
    )
    validate_parser.add_argument(
        "--config", type=str, required=True, help="Path to YAML configuration file"
    )
    
    # List components command
    list_parser = subparsers.add_parser(
        "list", help="List available TextAttack components"
    )
    list_parser.add_argument(
        "--component", choices=[
            "transformations", "constraints", "goal-functions", 
            "search-methods", "attack-recipes"
        ], required=True, help="Component type to list"
    )
    
    return parser.parse_args()


def run_operation(args):
    """Run a TextAttack operation from a YAML config."""
    try:
        # Load and validate config
        config = load_config_from_yaml(args.config)
        
        # Execute the operation
        executor = TextAttackExecutor(config)
        results = executor.execute()
        
        # Print results to console
        print(json.dumps(results, indent=2))
        
        # Save results to file if specified
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(results, f, indent=2)
        
        return 0
    except Exception as e:
        print(f"Error: {str(e)}", file=sys.stderr)
        return 1


def main():
    args = parse_args()
    try:
        if args.command == "run":
            return run_operation(args)
        elif args.command == "generate-template":
            return generate_template(args)
        elif args.command == "validate":
            return validate_config(args)
        elif args.command == "list":
            return list_components(args)
        else:
            print("No command specified. Use --help for available commands.")
            return 1
    except Exception as e:
        print(f"Error: {str(e)}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())


def generate_template(args):
    """Generate a configuration template."""
    if args.type == "attack":
        template = {
            "operation": "attack",
            "attack_type": "recipe",
            "attack_recipe": {
                "module": "textattack.attack_recipes",
                "name": "TextFoolerJin2019"
            },
            "model": {
                "module": "textattack.models.wrappers",
                "name": "HuggingFaceModelWrapper",
                "params": {
                    "model_name": "textattack/bert-base-uncased-imdb"
                }
            },
            "dataset": {
                "llama_stack": {
                    "name": "imdb",
                    "split": "test",
                    "text_field": "text",
                    "label_field": "label"
                }
            },
            "attack_params": {
                "num_examples": 10,
                "shuffle": True
            },
            "loggers": [
                {
                    "module": "textattack.loggers",
                    "name": "CSVLogger",
                    "params": {
                        "filename": "results/attack_results.csv"
                    }
                }
            ]
        }
    else:  # augmentation
        template = {
            "operation": "augmentation",
            "transformation": {
                "module": "textattack.transformations",
                "name": "WordSwapEmbedding"
            },
            "constraints": [
                {
                    "module": "textattack.constraints.semantics",
                    "name": "WordEmbeddingDistance",
                    "params": {
                        "min_cos_sim": 0.8
                    }
                }
            ],
            "dataset": {
                "llama_stack": {
                    "name": "sst2",
                    "split": "train"
                }
            },
            "augmentation": {
                "pct_words_to_swap": 0.1,
                "transformations_per_example": 2,
                "export": {
                    "format": "llama_stack",
                    "name": "augmented_dataset"
                }
            }
        }
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    
    # Write to file
    with open(args.output, 'w') as f:
        yaml.dump(template, f, default_flow_style=False, sort_keys=False)
    
    print(f"Template saved to {args.output}")
    return 0


def validate_config(args):
    """Validate a configuration file."""
    try:
        config = load_config_from_yaml(args.config)
        print(f"Configuration is valid: {args.config}")
        print("Configuration details:")
        print(f"  Operation: {config.operation}")
        if config.operation == "attack":
            print(f"  Attack type: {config.attack_type}")
        print(f"  Dataset: {config.dataset.llama_stack.name if config.dataset.llama_stack else config.dataset.standard.params.get('dataset_name')}")
        return 0
    except Exception as e:
        print(f"Configuration is invalid: {str(e)}", file=sys.stderr)
        return 1


def list_components(args):
    """List available TextAttack components."""
    import inspect
    import textattack

    component_type = args.component.replace('-', '_')
    
    mapping = {
        "transformations": textattack.transformations,
        "constraints": textattack.constraints,
        "goal_functions": textattack.goal_functions,
        "search_methods": textattack.search_methods,
        "attack_recipes": textattack.attack_recipes
    }
    
    module = mapping.get(component_type)
    if not module:
        print(f"Unknown component type: {args.component}", file=sys.stderr)
        return 1
    
    # Get all classes from the module that aren't imported
    module_name = module.__name__
    components = []
    
    for name, obj in inspect.getmembers(module):
        if inspect.isclass(obj) and obj.__module__.startswith(module_name):
            components.append((name, obj))
    
    if not components:
        print(f"No {args.component} found in TextAttack")
        return 0
    
    print(f"Available {args.component}:")
    for name, cls in sorted(components, key=lambda x: x[0]):
        doc = inspect.getdoc(cls)
        first_line = doc.split('\n')[0] if doc else "No description available"
        print(f"  {name}: {first_line}")
    
    return 0