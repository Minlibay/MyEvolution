# Agent-Based Simulation of Tool Evolution

## Overview

This is a scientific agent-based simulation designed to investigate how autonomous agents (models of early humans) evolve, discover tools, and develop technological behaviors through environmental interaction.

## Research Questions

1. How does instrumental behavior emerge?
2. How do random object combinations lead to technological discoveries?
3. How do tools improve survival rates?
4. How do cultural and biological evolution interact?
5. Which environmental parameters accelerate or slow technological progress?

## Features

- **Emergent Tool Creation**: Tools are not predefined but emerge from object combinations
- **Reinforcement Learning**: Agents learn through reward-based mechanisms
- **Evolutionary Dynamics**: Genetic evolution and cultural transmission
- **Rich Environment**: 2D grid with stochastic resource generation
- **Comprehensive Metrics**: Track technological progress, survival rates, and behavioral diversity

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run basic experiment
python scripts/run_experiment.py --config config/experiment_configs/basic_experiment.yaml

# Analyze results
python scripts/analyze_results.py --experiment_id basic_experiment_2024_02_17
```

## Project Structure

```
evolution_simulation/
├── config/                 # Configuration files
├── src/                    # Source code
│   ├── core/              # Core simulation components
│   ├── learning/          # Learning algorithms
│   ├── evolution/         # Evolution mechanisms
│   ├── utils/             # Utilities and analysis
│   └── experiments/       # Experiment definitions
├── tests/                 # Unit tests
├── data/                  # Logs, results, checkpoints
├── notebooks/             # Jupyter notebooks for analysis
└── scripts/               # Execution scripts
```

## Requirements

- Python 3.8+
- NumPy, Pandas, Matplotlib
- SciPy, Scikit-learn
- PyYAML, TQDM
- (Optional) Jupyter, Plotly for visualization

## License

MIT License - see LICENSE file for details
