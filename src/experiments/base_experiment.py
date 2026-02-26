"""
Base experiment class for running controlled experiments
"""

from typing import Dict, List, Any, Optional
import time
import json
from pathlib import Path

from core.simulation import Simulation
from utils.metrics import MetricsAnalyzer
from utils.visualization import SimulationVisualizer


class BaseExperiment:
    """Базовый класс для экспериментов"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.simulation: Optional[Simulation] = None
        self.results: Dict[str, Any] = {}
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None
        
        # Метаданные эксперимента
        self.experiment_name = config['experiment']['name']
        self.description = config['experiment']['description']
        self.duration = config['experiment']['duration']
    
    def setup(self):
        """Настройка эксперимента"""
        print(f"Настройка эксперимента: {self.experiment_name}")
        print(f"Описание: {self.description}")
        print(f"Длительность: {self.duration} шагов")
        
        # Создание симуляции
        self.simulation = Simulation(self.config)
        self.simulation.initialize()
        
        self.start_time = time.time()
        
        print("Эксперимент настроен и готов к запуску")
    
    def run(self) -> 'SimulationState':
        """Запускает эксперимент"""
        if not self.simulation:
            self.setup()
        
        print(f"Запуск эксперимента: {self.experiment_name}")
        
        try:
            # Запуск симуляции
            final_state = self.simulation.run(verbose=True)
            
            self.end_time = time.time()
            
            print(f"Эксперимент завершен за {self.end_time - self.start_time:.2f} секунд")
            
            return final_state
            
        except Exception as e:
            print(f"Ошибка в эксперименте: {e}")
            raise
    
    def analyze_results(self, state: 'SimulationState') -> Dict[str, Any]:
        """Анализирует результаты эксперимента"""
        print("Анализ результатов...")
        
        if not state.metrics_history:
            return {'error': 'No metrics data available'}
        
        # Базовая статистика
        basic_stats = self._calculate_basic_statistics(state)
        
        # Анализ метрик
        analyzer = MetricsAnalyzer(state.metrics_history)
        
        technology_analysis = analyzer.analyze_technology_progression()
        evolution_analysis = analyzer.analyze_evolution_patterns()
        cultural_analysis = analyzer.analyze_cultural_transmission()
        
        # Генерация выводов
        insights = analyzer.generate_insights()
        
        # Комплексный анализ
        comprehensive_analysis = self._perform_comprehensive_analysis(
            state, technology_analysis, evolution_analysis, cultural_analysis
        )
        
        self.results = {
            'experiment_info': {
                'name': self.experiment_name,
                'description': self.description,
                'duration': self.duration,
                'actual_duration': state.timestep,
                'execution_time': self.end_time - self.start_time if self.end_time else 0,
                'config': self.config
            },
            'basic_statistics': basic_stats,
            'technology_analysis': technology_analysis,
            'evolution_analysis': evolution_analysis,
            'cultural_analysis': cultural_analysis,
            'insights': insights,
            'comprehensive_analysis': comprehensive_analysis,
            'final_metrics': state.metrics_history[-1].to_dict() if state.metrics_history else {},
            'metrics_summary': state.metrics_calculator.calculate_summary_statistics()
        }
        
        print(f"Анализ завершен. Найдено {len(insights)} ключевых выводов")
        
        return self.results
    
    def _calculate_basic_statistics(self, state: 'SimulationState') -> Dict[str, Any]:
        """Рассчитывает базовую статистику"""
        stats = state.get_statistics()
        
        return {
            'final_population': stats.get('alive_agents', 0),
            'total_births': stats.get('total_births', 0),
            'total_deaths': stats.get('total_deaths', 0),
            'total_discoveries': stats.get('total_discoveries', 0),
            'total_tools': stats.get('total_tools', 0),
            'unique_tool_types': stats.get('unique_tool_types', 0),
            'final_generation': stats.get('generation', 0),
            'genetic_diversity': stats.get('genetic_diversity', 0),
            'average_intelligence': stats.get('average_intelligence', 0),
            'average_exploration_rate': stats.get('average_exploration_rate', 0),
            'resource_density': stats.get('resource_density', 0)
        }
    
    def _perform_comprehensive_analysis(self, state: 'SimulationState', 
                                     tech_analysis: Dict, evo_analysis: Dict, 
                                     cultural_analysis: Dict) -> Dict[str, Any]:
        """Выполняет комплексный анализ результатов"""
        
        # Оценка успеха эксперимента
        success_metrics = self._evaluate_experiment_success(state)
        
        # Сравнение с ожиданиями
        expectation_comparison = self._compare_with_expectations(state)
        
        # Ключевые паттерны
        key_patterns = self._identify_key_patterns(
            tech_analysis, evo_analysis, cultural_analysis
        )
        
        # Рекомендации для будущих экспериментов
        recommendations = self._generate_recommendations(
            state, tech_analysis, evo_analysis, cultural_analysis
        )
        
        return {
            'success_evaluation': success_metrics,
            'expectation_comparison': expectation_comparison,
            'key_patterns': key_patterns,
            'recommendations': recommendations
        }
    
    def _evaluate_experiment_success(self, state: 'SimulationState') -> Dict[str, Any]:
        """Оценивает успешность эксперимента"""
        basic_stats = self._calculate_basic_statistics(state)
        
        # Критерии успеха
        survival_score = min(1.0, basic_stats['final_population'] / 
                           self.config['simulation']['agents']['initial_population'])
        
        discovery_score = min(1.0, basic_stats['total_discoveries'] / 
                            max(1, self.duration // 100))  # Ожидаем хотя бы 1 открытие на 100 шагов
        
        tool_score = min(1.0, basic_stats['total_tools'] / 
                        max(1, basic_stats['final_population']))
        
        diversity_score = min(1.0, basic_stats['genetic_diversity'])
        
        # Общая оценка
        overall_score = (survival_score + discovery_score + 
                         tool_score + diversity_score) / 4
        
        return {
            'survival_score': survival_score,
            'discovery_score': discovery_score,
            'tool_score': tool_score,
            'diversity_score': diversity_score,
            'overall_score': overall_score,
            'success_level': self._get_success_level(overall_score)
        }
    
    def _get_success_level(self, score: float) -> str:
        """Определяет уровень успеха"""
        if score >= 0.8:
            return "Отлично"
        elif score >= 0.6:
            return "Хорошо"
        elif score >= 0.4:
            return "Удовлетворительно"
        else:
            return "Плохо"
    
    def _compare_with_expectations(self, state: 'SimulationState') -> Dict[str, Any]:
        """Сравнивает результаты с ожиданиями"""
        basic_stats = self._calculate_basic_statistics(state)
        
        comparisons = {}
        
        # Ожидания по популяции
        expected_min_population = self.config['simulation']['agents']['initial_population'] // 2
        population_comparison = "Выше ожиданий" if basic_stats['final_population'] > expected_min_population else "Ниже ожиданий"
        comparisons['population'] = {
            'expected': expected_min_population,
            'actual': basic_stats['final_population'],
            'comparison': population_comparison
        }
        
        # Ожидания по открытиям
        expected_discoveries = max(1, self.duration // 200)
        discovery_comparison = "Выше ожиданий" if basic_stats['total_discoveries'] > expected_discoveries else "Ниже ожиданий"
        comparisons['discoveries'] = {
            'expected': expected_discoveries,
            'actual': basic_stats['total_discoveries'],
            'comparison': discovery_comparison
        }
        
        return comparisons
    
    def _identify_key_patterns(self, tech_analysis: Dict, evo_analysis: Dict, 
                             cultural_analysis: Dict) -> List[str]:
        """Идентифицирует ключевые паттерны"""
        patterns = []
        
        # Технологические паттерны
        if tech_analysis.get('discovery_trend', 0) > 0.1:
            patterns.append("Ускоренный технологический прогресс")
        
        if tech_analysis.get('discovery_acceleration_points'):
            patterns.append(f"Найдено {len(tech_analysis['discovery_acceleration_points'])} точек ускорения открытий")
        
        # Эволюционные паттерны
        if evo_analysis.get('genetic_diversity_trend', 0) > 0.01:
            patterns.append("Рост генетического разнообразия")
        elif evo_analysis.get('genetic_diversity_trend', 0) < -0.01:
            patterns.append("Снижение генетического разнообразия (риск вырождения)")
        
        if evo_analysis.get('intelligence_evolution', 0) > 0.01:
            patterns.append("Эволюция интеллекта")
        
        # Культурные паттерны
        if cultural_analysis.get('knowledge_spread_rate', 0) > 0.5:
            patterns.append("Быстрое распространение знаний")
        
        cultural_complexity = cultural_analysis.get('cultural_complexity_index', 0)
        if cultural_complexity > 0.7:
            patterns.append("Высокая культурная сложность")
        elif cultural_complexity < 0.3:
            patterns.append("Низкая культурная сложность")
        
        return patterns
    
    def _generate_recommendations(self, state: 'SimulationState', 
                                tech_analysis: Dict, evo_analysis: Dict, 
                                cultural_analysis: Dict) -> List[str]:
        """Генерирует рекомендации для будущих экспериментов"""
        recommendations = []
        
        basic_stats = self._calculate_basic_statistics(state)
        
        # Рекомендации по популяции
        if basic_stats['final_population'] < self.config['simulation']['agents']['initial_population'] // 2:
            recommendations.append("Рассмотрите увеличение начальной популяции или улучшение условий выживания")
        
        # Рекомендации по технологиям
        if basic_stats['total_discoveries'] == 0:
            recommendations.append("Увеличьте exploration_rate или добавьте больше ресурсов для стимуляции открытий")
        elif tech_analysis.get('discovery_trend', 0) < 0:
            recommendations.append("Технологический прогресс замедляется - рассмотрите изменение параметров среды")
        
        # Рекомендации по эволюции
        if evo_analysis.get('genetic_diversity_trend', 0) < -0.01:
            recommendations.append("Генетическое разнообразие снижается - уменьшите давление селекции или увеличьте мутации")
        
        # Рекомендации по культуре
        if cultural_analysis.get('knowledge_spread_rate', 0) < 0.1:
            recommendations.append("Знания распространяются медленно - увеличьте культурную передачу")
        
        return recommendations
    
    def save_results(self, results: Dict[str, Any], output_dir: str):
        """Сохраняет результаты эксперимента"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Сохранение JSON с результатами
        results_file = output_path / f"{self.experiment_name}_results.json"
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False, default=str)
        
        # Сохранение метрик
        if self.simulation and self.simulation.state:
            metrics_file = output_path / f"{self.experiment_name}_metrics.json"
            self.simulation.state.metrics_calculator.export_metrics(metrics_file)
            
            # Визуализация
            visualizer = SimulationVisualizer()
            
            # График метрик
            metrics_plot = output_path / f"{self.experiment_name}_metrics.png"
            visualizer.plot_metrics_timeline(
                self.simulation.state.metrics_history, 
                save_path=str(metrics_plot)
            )
            
            # Технологическая эволюция
            tech_plot = output_path / f"{self.experiment_name}_technology.png"
            visualizer.plot_technology_evolution(
                self.simulation.state.metrics_history,
                save_path=str(tech_plot)
            )
            
            # Эволюционная динамика
            evo_plot = output_path / f"{self.experiment_name}_evolution.png"
            visualizer.plot_evolution_dynamics(
                self.simulation.state.metrics_history,
                save_path=str(evo_plot)
            )
        
        print(f"Результаты сохранены в: {output_path}")
        
        # Сохранение конфигурации
        config_file = output_path / f"{self.experiment_name}_config.json"
        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(self.config, f, indent=2, ensure_ascii=False)
        
        return output_path
