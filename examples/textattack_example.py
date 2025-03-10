#!/usr/bin/env python3
"""
Example script demonstrating the use of TextAttack with llama-stack.
"""

import os
import argparse
from llama_stack.datasetio.textattack import (
    load_config_from_yaml,
    get_textattack_provider,
    execute_textattack
)


def run_attack(config_path, provider_type="inline"):
    """Run a TextAttack attack using the specified provider.
    
    Args:
        config_path: Path to the YAML attack configuration
        provider_type: Type of provider to use ("inline" or "remote")
    """
    print(f"Loading attack configuration from {config_path}")
    config = load_config_from_yaml(config_path)
    
    print(f"Running attack using {provider_type} provider")
    if provider_type == "default":
        results = execute_textattack(config)
    else:
        provider = get_textattack_provider(provider_type)
        results = provider.execute(config)
    
    print("\nAttack Results:")
    print(f"Total examples: {results['total_examples']}")
    print(f"Successful attacks: {results['successful_attacks']}")
    print(f"Success rate: {results['success_rate']:.2f}")
    print(f"Average perturbed word percentage: {results['average_perturbed_word_percentage']:.2f}")


def run_augmentation(config_path, provider_type="inline"):
    """Run a TextAttack augmentation using the specified provider.
    
    Args:
        config_path: Path to the YAML augmentation configuration
        provider_type: Type of provider to use ("inline" or "remote")
    """
    print(f"Loading augmentation configuration from {config_path}")
    config = load_config_from_yaml(config_path)
    
    print(f"Running augmentation using {provider_type} provider")
    if provider_type == "default":
        results = execute_textattack(config)
    else:
        provider = get_textattack_provider(provider_type)
        results = provider.execute(config)
    
    print("\nAugmentation Results:")
    print(f"Original examples: {results['original_examples']}")
    print(f"Augmented examples: {results['augmented_examples']}")
    print(f"Expansion factor: {results['expansion_factor']:.2f}")
    
    if 'export_path' in results and results['export_path']:
        print(f"Augmented dataset exported to: {results['export_path']}")


def main():
    """Main function to parse arguments and run commands."""
    parser = argparse.ArgumentParser(description="Run TextAttack operations")
    parser.add_argument("--operation", choices=["attack", "augmentation"], required=True,
                      help="Type of operation to perform")
    parser.add_argument("--config", required=True,
                      help="Path to YAML configuration file")
    parser.add_argument("--provider", choices=["inline", "remote", "default"], default="default",
                      help="Type of provider to use")
    
    args = parser.parse_args()
    
    # Create results directory if needed
    os.makedirs("results", exist_ok=True)
    
    if args.operation == "attack":
        run_attack(args.config, args.provider)
    elif args.operation == "augmentation":
        run_augmentation(args.config, args.provider)


if __name__ == "__main__":
    main()