#!/usr/bin/env python3
"""
LiveCodeBench Dedicated Execution Script (Improved Version)

Supports the latest LiveCodeBench version (release_v6) and allows easy execution with various configurations.
Improved based on Xolver's lcb.py.
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path
from typing import Dict, Any


class LiveCodeBenchRunner:
    """LiveCodeBench runner"""
    
    def __init__(self):
        self.supported_versions = [
            "release_v1", "release_v2", "release_v3",
            "release_v4", "release_v5", "release_v6"
        ]
        self.default_config = {
            "model": "ChatGPT",
            "model_provider": "OpenAI",
            "strategy": "Direct",
            "language": "Python3",
            "temperature": 0,
            "top_p": 0.95,
            "pass_at_k": 1,
            "verbose": "2",
            "version": "release_v6"
        }
    
    def check_data_availability(self, version: str) -> bool:
        """Check data file availability"""
        data_path = f"data/LiveCodeBench/{version}/livecodebench.jsonl"
        if not os.path.exists(data_path):
            print(f"LiveCodeBench {version} data file not found: {data_path}")
            print("Please download and convert the data first.")
            print("Examples:")
            print(f"  python src/datasets/convert-livecodebench.py --download --version {version}")
            print(f"  python src/datasets/convert-livecodebench.py --input ./LiveCodeBench --version {version}")
            return False
        return True
    
    def run_evaluation(self, config: Dict[str, Any]) -> bool:
        """Run LiveCodeBench evaluation"""
        version = config.get('version', 'release_v6')
        
        # Check data file availability
        if not self.check_data_availability(version):
            return False
        
        # Execute main.py
        cmd = [
            "python", "src/main.py",
            "--dataset", f"lcb_{version}",
            "--model", config['model'],
            "--model_provider", config['model_provider'],
            "--strategy", config['strategy'],
            "--language", config['language'],
            "--temperature", str(config['temperature']),
            "--top_p", str(config['top_p']),
            "--pass_at_k", str(config['pass_at_k']),
            "--verbose", config['verbose']
        ]
        
        print(f"Execution command: {' '.join(cmd)}")
        print(f"Starting LiveCodeBench {version} evaluation...")
        print(f"Model: {config['model']} ({config['model_provider']})")
        print(f"Strategy: {config['strategy']}")
        print(f"Language: {config['language']}")
        print(f"Temperature: {config['temperature']}")
        print(f"Top-p: {config['top_p']}")
        print(f"Pass@k: {config['pass_at_k']}")
        print("-" * 50)
        
        try:
            result = subprocess.run(cmd, check=True, capture_output=False)
            print("-" * 50)
            print(f"LiveCodeBench {version} evaluation completed!")
            return True
        except subprocess.CalledProcessError as e:
            print(f"Error occurred: {e}")
            return False
    
    def interactive_config(self) -> Dict[str, Any]:
        """Interactive configuration input"""
        print("LiveCodeBench Evaluation Configuration")
        print("=" * 50)
        
        config = self.default_config.copy()
        
        # Get user input
        print("Please enter configuration (press Enter to use default values):")
        
        # Version selection
        print(f"Supported versions: {', '.join(self.supported_versions)}")
        version = input(f"Release version [{config['version']}]: ").strip()
        if version and version in self.supported_versions:
            config['version'] = version
        
        # Model configuration
        model = input(f"Model name [{config['model']}]: ").strip()
        if model:
            config['model'] = model
        
        model_provider = input(f"Model provider [{config['model_provider']}]: ").strip()
        if model_provider:
            config['model_provider'] = model_provider
        
        strategy = input(f"Prompting strategy [{config['strategy']}]: ").strip()
        if strategy:
            config['strategy'] = strategy
        
        language = input(f"Programming language [{config['language']}]: ").strip()
        if language:
            config['language'] = language
        
        # Hyperparameters
        temp_input = input(f"Temperature [{config['temperature']}]: ").strip()
        if temp_input:
            try:
                config['temperature'] = float(temp_input)
            except ValueError:
                print("Invalid temperature value. Using default.")
        
        top_p_input = input(f"Top-p [{config['top_p']}]: ").strip()
        if top_p_input:
            try:
                config['top_p'] = float(top_p_input)
            except ValueError:
                print("Invalid Top-p value. Using default.")
        
        pass_k_input = input(f"Pass@k [{config['pass_at_k']}]: ").strip()
        if pass_k_input:
            try:
                config['pass_at_k'] = int(pass_k_input)
            except ValueError:
                print("Invalid Pass@k value. Using default.")
        
        verbose = input(f"Verbose level [{config['verbose']}]: ").strip()
        if verbose:
            config['verbose'] = verbose
        
        return config
    
    def show_statistics(self, version: str = "release_v6"):
        """Display dataset statistics"""
        data_path = f"data/LiveCodeBench/{version}/livecodebench.jsonl"
        if not os.path.exists(data_path):
            print(f"Data file not found: {data_path}")
            return
        
        try:
            import json
            with open(data_path, 'r', encoding='utf-8') as f:
                data = [json.loads(line) for line in f if line.strip()]
            
            print(f"\nLiveCodeBench {version} Statistics:")
            print("-" * 40)
            print(f"Total problems: {len(data)}")
            
            # Difficulty distribution
            difficulty_stats = {}
            platform_stats = {}
            category_stats = {}
            
            for item in data:
                difficulty = item.get('difficulty', 'Unknown')
                platform = item.get('platform', 'Unknown')
                category = item.get('category', 'Unknown')
                
                difficulty_stats[difficulty] = difficulty_stats.get(difficulty, 0) + 1
                platform_stats[platform] = platform_stats.get(platform, 0) + 1
                category_stats[category] = category_stats.get(category, 0) + 1
            
            print("\nDifficulty distribution:")
            for diff, count in sorted(difficulty_stats.items()):
                print(f"  {diff}: {count}")
            
            print("\nPlatform distribution:")
            for platform, count in sorted(platform_stats.items()):
                print(f"  {platform}: {count}")
            
            print("\nCategory distribution:")
            for category, count in sorted(category_stats.items()):
                print(f"  {category}: {count}")
                
        except Exception as e:
            print(f"Error loading statistics: {e}")
    
    def quick_evaluation(self, version: str = "release_v6", strategy: str = "Direct"):
        """Run quick evaluation"""
        config = self.default_config.copy()
        config['version'] = version
        config['strategy'] = strategy
        
        print(f"Running quick evaluation: {version} / {strategy}")
        return self.run_evaluation(config)
    
    def batch_evaluation(self, version: str = "release_v6", strategies: list = None):
        """Run batch evaluation with multiple strategies"""
        if strategies is None:
            strategies = ["Direct", "CoT", "CodeSIM"]
        
        print(f"Starting batch evaluation: {version}")
        print(f"Strategies: {', '.join(strategies)}")
        
        results = {}
        for strategy in strategies:
            print(f"\nEvaluating with {strategy} strategy...")
            success = self.quick_evaluation(version, strategy)
            results[strategy] = "Success" if success else "Failed"
        
        print("\nBatch evaluation results:")
        for strategy, result in results.items():
            print(f"  {strategy}: {result}")
        
        return results

    def validate_data(self, version: str = "release_v6"):
        """Validate LiveCodeBench data"""
        from src.datasets.convert_livecodebench import LiveCodeBenchConverter
        
        data_path = f"data/LiveCodeBench/{version}/livecodebench.jsonl"
        if not os.path.exists(data_path):
            print(f"Data file not found: {data_path}")
            return False
        
        converter = LiveCodeBenchConverter()
        validation_results = converter.validate_converted_data(data_path)
        
        if validation_results['valid']:
            print(f"Data validation passed for {version}")
            print(f"Total problems: {validation_results['total_problems']}")
        else:
            print(f"Data validation failed for {version}")
            print(f"Missing fields: {len(validation_results['missing_fields'])}")
            print(f"Invalid test cases: {len(validation_results['invalid_test_cases'])}")
            print(f"Empty descriptions: {len(validation_results['empty_descriptions'])}")
            if 'error' in validation_results:
                print(f"Error: {validation_results['error']}")
        
        return validation_results['valid']


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="LiveCodeBench Evaluation Runner")
    parser.add_argument("--version", "-v", default="release_v6",
                       choices=["release_v1", "release_v2", "release_v3", "release_v4", "release_v5", "release_v6"],
                       help="LiveCodeBench release version (default: release_v6)")
    parser.add_argument("--strategy", "-s", default="Direct",
                       choices=["Direct", "CoT", "SelfPlanning", "Analogical", "MapCoder", "CodeSIM"],
                       help="Prompting strategy (default: Direct)")
    parser.add_argument("--quick", "-q", action="store_true", help="Run quick evaluation")
    parser.add_argument("--batch", "-b", action="store_true", help="Run batch evaluation with multiple strategies")
    parser.add_argument("--stats", action="store_true", help="Display dataset statistics")
    parser.add_argument("--validate", action="store_true", help="Validate dataset data")
    parser.add_argument("--interactive", "-i", action="store_true", help="Interactive configuration mode")
    
    args = parser.parse_args()
    
    runner = LiveCodeBenchRunner()
    
    if args.stats:
        runner.show_statistics(args.version)
        return
    
    if args.validate:
        runner.validate_data(args.version)
        return
    
    if args.quick:
        runner.quick_evaluation(args.version, args.strategy)
    elif args.batch:
        runner.batch_evaluation(args.version)
    elif args.interactive:
        config = runner.interactive_config()
        print("\n" + "=" * 50)
        print("Final Configuration:")
        for key, value in config.items():
            print(f"  {key}: {value}")
        print("=" * 50)
        
        confirm = input("\nExecute with this configuration? (y/N): ").strip().lower()
        if confirm in ['y', 'yes']:
            success = runner.run_evaluation(config)
            if success:
                print("\nEvaluation completed successfully!")
                print("Results can be found in the results/LiveCodeBench/ folder.")
            else:
                print("\nError occurred during evaluation.")
        else:
            print("Execution cancelled.")
    else:
        # Default interactive mode
        config = runner.interactive_config()
        print("\n" + "=" * 50)
        print("Final Configuration:")
        for key, value in config.items():
            print(f"  {key}: {value}")
        print("=" * 50)
        
        confirm = input("\nExecute with this configuration? (y/N): ").strip().lower()
        if confirm in ['y', 'yes']:
            success = runner.run_evaluation(config)
            if success:
                print("\nEvaluation completed successfully!")
                print("Results can be found in the results/LiveCodeBench/ folder.")
            else:
                print("\nError occurred during evaluation.")
        else:
            print("Execution cancelled.")


if __name__ == "__main__":
    main()
