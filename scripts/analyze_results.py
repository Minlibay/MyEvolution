#!/usr/bin/env python3
"""
Analyze results from completed experiments
"""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from utils.data_analysis import DataAnalyzer


def main():
    parser = argparse.ArgumentParser(description="Analyze experiment results")
    parser.add_argument("--experiment_id", required=True, help="Experiment ID to analyze")
    parser.add_argument("--data_dir", default="data", help="Data directory")
    parser.add_argument("--output", default="data/analysis", help="Analysis output directory")
    parser.add_argument("--report", action="store_true", help="Generate detailed report")
    
    args = parser.parse_args()
    
    # Initialize analyzer
    analyzer = DataAnalyzer(args.data_dir)
    
    # Load experiment data
    data = analyzer.load_simulation_data(args.experiment_id)
    
    # Analyze discovery patterns
    discovery_analysis = analyzer.analyze_discovery_patterns()
    
    # Analyze evolution trends
    evolution_analysis = analyzer.analyze_evolution_trends()
    
    print(f"Analysis for experiment: {args.experiment_id}")
    print("=" * 50)
    print(f"Discovery patterns: {discovery_analysis}")
    print(f"Evolution trends: {evolution_analysis}")
    
    if args.report:
        report = analyzer.generate_report(args.experiment_id)
        report_path = Path(args.output) / f"{args.experiment_id}_report.md"
        
        with open(report_path, 'w') as f:
            f.write(report)
        
        print(f"Detailed report saved to: {report_path}")


if __name__ == "__main__":
    main()
