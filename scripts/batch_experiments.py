#!/usr/bin/env python3
"""
Run batch experiments with different parameter configurations
"""

import argparse
import sys
import yaml
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from experiments.base_experiment import BaseExperiment
from utils.config_loader import load_config


def run_experiment_batch(config_files, output_dir, verbose=False):
    """Run multiple experiments in batch"""
    results = []
    
    for config_file in config_files:
        if verbose:
            print(f"Running experiment with config: {config_file}")
        
        # Load configuration
        config = load_config(config_file)
        
        # Create and run experiment
        experiment = BaseExperiment(config)
        experiment.setup()
        
        # Run experiment
        final_state = experiment.run()
        
        # Analyze results
        results = experiment.analyze_results(final_state)
        
        # Save results with config-specific name
        experiment_name = config['experiment']['name']
        experiment_output = Path(output_dir) / experiment_name
        experiment.save_results(results, experiment_output)
        
        if verbose:
            print(f"Completed: {experiment_name}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Run batch experiments")
    parser.add_argument("--configs", nargs="+", required=True, help="List of config files")
    parser.add_argument("--output", default="data/results/batch", help="Output directory")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    
    args = parser.parse_args()
    
    # Create output directory
    Path(args.output).mkdir(parents=True, exist_ok=True)
    
    # Run batch experiments
    results = run_experiment_batch(args.configs, args.output, args.verbose)
    
    print(f"Batch experiment completed. Results saved to: {args.output}")


if __name__ == "__main__":
    main()
