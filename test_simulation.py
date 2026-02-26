"""
Simple test runner to verify the simulation works
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from core.simulation import Simulation
from utils.config_loader import load_config


def main():
    """Запускает простой тест симуляции"""
    print("Запуск тестовой симуляции...")
    
    # Загружаем базовую конфигурацию
    config = load_config("config/experiment_configs/basic_experiment.yaml")
    
    # Уменьшаем длительность для теста
    config['experiment']['duration'] = 500
    
    # Создаем и запускаем симуляцию
    simulation = Simulation(config)
    
    try:
        final_state = simulation.run(verbose=True)
        
        # Выводим финальную статистику
        stats = simulation.get_statistics()
        
        print("\nФинальная статистика:")
        print(f"Выжившие агенты: {stats.get('alive_agents', 0)}")
        print(f"Общие открытия: {stats.get('total_discoveries', 0)}")
        print(f"Всего инструментов: {stats.get('total_tools', 0)}")
        print(f"Поколений: {stats.get('generation', 0)}")
        
        # Проверяем базовую работоспособность
        if stats.get('alive_agents', 0) > 0:
            print("\n✅ Симуляция успешно завершена!")
        else:
            print("\n⚠️ Все агенты погибли, но симуляция завершилась")
        
    except Exception as e:
        print(f"\n❌ Ошибка в симуляции: {e}")
        return False
    
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
