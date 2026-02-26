"""
Configuration loader for loading experiment configurations from YAML files
"""

import yaml
from typing import Dict, Any
from pathlib import Path


def load_config(config_path: str) -> Dict[str, Any]:
    """Загружает конфигурацию из YAML файла"""
    config_file = Path(config_path)
    
    if not config_file.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_file, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # Валидация конфигурации
    validate_config(config)
    
    return config


def validate_config(config: Dict[str, Any]):
    """Валидирует структуру конфигурации"""
    required_sections = ['experiment', 'simulation', 'environment', 'logging']
    
    for section in required_sections:
        if section not in config:
            raise ValueError(f"Missing required section: {section}")
    
    # Валидация эксперимента
    experiment = config['experiment']
    required_experiment_fields = ['name', 'description', 'duration']
    for field in required_experiment_fields:
        if field not in experiment:
            raise ValueError(f"Missing required field in experiment: {field}")
    
    # Валидация симуляции
    simulation = config['simulation']
    if 'world' not in simulation:
        raise ValueError("Missing 'world' section in simulation")
    
    world = simulation['world']
    required_world_fields = ['width', 'height', 'seed']
    for field in required_world_fields:
        if field not in world:
            raise ValueError(f"Missing required field in world: {field}")
    
    # Валидация агентов
    if 'agents' not in simulation:
        raise ValueError("Missing 'agents' section in simulation")
    
    agents = simulation['agents']
    required_agent_fields = ['initial_population', 'max_population']
    for field in required_agent_fields:
        if field not in agents:
            raise ValueError(f"Missing required field in agents: {field}")


def get_default_config() -> Dict[str, Any]:
    """Возвращает конфигурацию по умолчанию"""
    return {
        'experiment': {
            'name': 'default_experiment',
            'description': 'Default experiment configuration',
            'duration': 10000
        },
        'simulation': {
            'world': {
                'width': 100,
                'height': 100,
                'seed': 42
            },
            'agents': {
                'initial_population': 20,
                'max_population': 100
            },
            'learning': {
                'learning_rate': 0.1,
                'discount_factor': 0.95,
                'epsilon_decay': 0.995
            },
            'evolution': {
                'reproduction_interval': 100,
                'mutation_rate': 0.1,
                'selection_pressure': 0.8
            }
        },
        'environment': {
            'resource_spawn_rate': 0.1,
            'season_length': 1000
        },
        'logging': {
            'log_interval': 10,
            'save_interval': 1000,
            'metrics': ['population', 'technology', 'behavior']
        }
    }


def save_config(config: Dict[str, Any], config_path: str):
    """Сохраняет конфигурацию в YAML файл"""
    config_file = Path(config_path)
    config_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(config_file, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)


def merge_configs(base_config: Dict[str, Any], override_config: Dict[str, Any]) -> Dict[str, Any]:
    """Сливает две конфигурации, override имеет приоритет"""
    merged = base_config.copy()
    
    for key, value in override_config.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = merge_configs(merged[key], value)
        else:
            merged[key] = value
    
    return merged
