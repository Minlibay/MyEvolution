#!/usr/bin/env python3
"""
Run a single experiment with specified configuration
"""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from experiments.base_experiment import BaseExperiment
from utils.config_loader import load_config


def main():
    parser = argparse.ArgumentParser(description="Run evolution simulation experiment")
    parser.add_argument("--config", required=True, help="Path to experiment config file")
    parser.add_argument("--output", default="data/results", help="Output directory for results")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Create and run experiment
    experiment = BaseExperiment(config)
    
    if args.verbose:
        print(f"Running experiment: {config['experiment']['name']}")
        print(f"Duration: {config['experiment']['duration']} steps")
        print(f"Output directory: {args.output}")
    
    # Setup experiment
    experiment.setup()
    
    # Run experiment
    final_state = experiment.run()
    
    # Analyze results
    results = experiment.analyze_results(final_state)
    
    # Save results
    experiment.save_results(results, args.output)
    
    if args.verbose:
        print("Experiment completed successfully!")
        print(f"Results saved to: {args.output}")


if __name__ == "__main__":
    main()
