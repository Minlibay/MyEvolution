"""
Metrics module - comprehensive metrics collection and analysis
"""

from typing import Dict, List, Any, Optional, Tuple
import numpy as np
from collections import defaultdict, deque
import json

from ..core.agent import Agent
from ..core.environment import Environment
from ..core.objects import Object
from ..core.tools import Tool


class SimulationMetrics:
    """Метрики симуляции на каждом шаге"""
    
    def __init__(self, timestamp: int):
        self.timestamp = timestamp
        
        # Популяционные метрики
        self.population_size = 0
        self.average_age = 0.0
        self.average_health = 0.0
        self.average_energy = 0.0
        self.average_hunger = 0.0
        self.birth_rate = 0.0
        self.death_rate = 0.0
        
        # Генетические метрики
        self.genetic_diversity = 0.0
        self.average_intelligence = 0.0
        self.average_exploration_bias = 0.0
        self.average_strength = 0.0
        
        # Технологические метрики
        self.total_tools = 0
        self.unique_tool_types = 0
        self.average_tool_complexity = 0.0
        self.discovery_rate = 0.0
        self.total_discoveries = 0
        
        # Поведенческие метрики
        self.average_exploration_rate = 0.0
        self.action_diversity = 0.0
        self.cooperation_events = 0
        self.total_actions = 0
        
        # Средовые метрики
        self.resource_density = 0.0
        self.object_types_distribution = {}
        self.season = 0
        self.temperature = 0.0
        
        # Экономические метрики
        self.total_resources_collected = 0
        self.total_resources_consumed = 0
        self.average_inventory_size = 0.0
        
        # Обучения метрики
        self.total_learning_episodes = 0
        self.average_q_table_size = 0.0
        self.convergence_metrics = {}
    
    def to_dict(self) -> Dict[str, Any]:
        """Преобразует метрики в словарь"""
        return {
            'timestamp': self.timestamp,
            'population': {
                'size': self.population_size,
                'average_age': self.average_age,
                'average_health': self.average_health,
                'average_energy': self.average_energy,
                'average_hunger': self.average_hunger,
                'birth_rate': self.birth_rate,
                'death_rate': self.death_rate
            },
            'genetics': {
                'diversity': self.genetic_diversity,
                'average_intelligence': self.average_intelligence,
                'average_exploration_bias': self.average_exploration_bias,
                'average_strength': self.average_strength
            },
            'technology': {
                'total_tools': self.total_tools,
                'unique_tool_types': self.unique_tool_types,
                'average_complexity': self.average_tool_complexity,
                'discovery_rate': self.discovery_rate,
                'total_discoveries': self.total_discoveries
            },
            'behavior': {
                'average_exploration_rate': self.average_exploration_rate,
                'action_diversity': self.action_diversity,
                'cooperation_events': self.cooperation_events,
                'total_actions': self.total_actions
            },
            'environment': {
                'resource_density': self.resource_density,
                'object_types_distribution': self.object_types_distribution,
                'season': self.season,
                'temperature': self.temperature
            },
            'economics': {
                'total_resources_collected': self.total_resources_collected,
                'total_resources_consumed': self.total_resources_consumed,
                'average_inventory_size': self.average_inventory_size
            },
            'learning': {
                'total_learning_episodes': self.total_learning_episodes,
                'average_q_table_size': self.average_q_table_size,
                'convergence_metrics': self.convergence_metrics
            }
        }


class MetricsCalculator:
    """Калькулятор метрик симуляции"""
    
    def __init__(self):
        self.metrics_history: List[SimulationMetrics] = []
        self.previous_population_size = 0
        self.previous_discoveries = 0
        self.action_counts = defaultdict(int)
        self.resource_counts = defaultdict(int)
    
    def calculate_metrics(self, agents: List[Agent], environment: Environment, 
                         timestamp: int) -> SimulationMetrics:
        """Рассчитывает все метрики для текущего состояния"""
        metrics = SimulationMetrics(timestamp)
        
        # Популяционные метрики
        self._calculate_population_metrics(agents, metrics)
        
        # Генетические метрики
        self._calculate_genetic_metrics(agents, metrics)
        
        # Технологические метрики
        self._calculate_technology_metrics(agents, environment, metrics)
        
        # Поведенческие метрики
        self._calculate_behavioral_metrics(agents, metrics)
        
        # Средовые метрики
        self._calculate_environmental_metrics(environment, metrics)
        
        # Экономические метрики
        self._calculate_economic_metrics(agents, metrics)
        
        # Метрики обучения
        self._calculate_learning_metrics(agents, metrics)
        
        # Обновляем историю
        self.metrics_history.append(metrics)
        
        # Обновляем предыдущие значения
        self.previous_population_size = metrics.population_size
        self.previous_discoveries = metrics.total_discoveries
        
        return metrics
    
    def _calculate_population_metrics(self, agents: List[Agent], metrics: SimulationMetrics):
        """Рассчитывает популяционные метрики"""
        if not agents:
            return
        
        metrics.population_size = len(agents)
        
        ages = [agent.age for agent in agents]
        health_values = [agent.health for agent in agents]
        energy_values = [agent.energy for agent in agents]
        hunger_values = [agent.hunger for agent in agents]
        
        metrics.average_age = np.mean(ages)
        metrics.average_health = np.mean(health_values)
        metrics.average_energy = np.mean(energy_values)
        metrics.average_hunger = np.mean(hunger_values)
        
        # Смертность и рождаемость
        if self.previous_population_size > 0:
            metrics.death_rate = max(0, self.previous_population_size - metrics.population_size) / self.previous_population_size
            metrics.birth_rate = max(0, metrics.population_size - self.previous_population_size) / self.previous_population_size
    
    def _calculate_genetic_metrics(self, agents: List[Agent], metrics: SimulationMetrics):
        """Рассчитывает генетические метрики"""
        if not agents:
            return
        
        # Генетическое разнообразие
        gene_vectors = [agent.genes.to_dict() for agent in agents]
        metrics.genetic_diversity = self._calculate_genetic_diversity(gene_vectors)
        
        # Средние значения генов
        intelligence_values = [agent.genes.intelligence for agent in agents]
        exploration_values = [agent.genes.exploration_bias for agent in agents]
        strength_values = [agent.genes.strength for agent in agents]
        
        metrics.average_intelligence = np.mean(intelligence_values)
        metrics.average_exploration_bias = np.mean(exploration_values)
        metrics.average_strength = np.mean(strength_values)
    
    def _calculate_genetic_diversity(self, gene_vectors: List[Dict]) -> float:
        """Рассчитывает генетическое разнообразие"""
        if len(gene_vectors) < 2:
            return 0.0
        
        total_distance = 0.0
        pair_count = 0
        
        for i in range(len(gene_vectors)):
            for j in range(i + 1, len(gene_vectors)):
                distance = self._gene_distance(gene_vectors[i], gene_vectors[j])
                total_distance += distance
                pair_count += 1
        
        return total_distance / pair_count if pair_count > 0 else 0.0
    
    def _gene_distance(self, genes1: Dict, genes2: Dict) -> float:
        """Рассчитывает расстояние между двумя наборами генов"""
        distance = 0.0
        for gene_name in genes1:
            distance += abs(genes1[gene_name] - genes2[gene_name])
        return distance / len(genes1)
    
    def _calculate_technology_metrics(self, agents: List[Agent], environment: Environment, 
                                     metrics: SimulationMetrics):
        """Рассчитывает технологические метрики"""
        metrics.total_tools = sum(len(agent.tools) for agent in agents)
        
        # Уникальные типы инструментов
        tool_types = set()
        total_complexity = 0.0
        tool_count = 0
        
        for agent in agents:
            for tool_id in agent.tools:
                tool = environment.tools.get(tool_id)
                if tool:
                    tool_types.add(tool.get_tool_type())
                    total_complexity += tool.get_complexity_score()
                    tool_count += 1
        
        metrics.unique_tool_types = len(tool_types)
        metrics.average_tool_complexity = total_complexity / max(1, tool_count)
        
        # Открытия
        metrics.total_discoveries = sum(len(agent.discoveries_made) for agent in agents)
        
        if self.previous_discoveries > 0:
            metrics.discovery_rate = (metrics.total_discoveries - self.previous_discoveries) / self.previous_discoveries
        else:
            metrics.discovery_rate = 0.0
    
    def _calculate_behavioral_metrics(self, agents: List[Agent], metrics: SimulationMetrics):
        """Рассчитывает поведенческие метрики"""
        if not agents:
            return
        
        exploration_rates = [agent.exploration_rate for agent in agents]
        metrics.average_exploration_rate = np.mean(exploration_rates)
        
        # Разнообразие действий (на основе Q-таблиц)
        all_actions = set()
        total_q_entries = 0
        
        for agent in agents:
            for (state, action) in agent.q_table.keys():
                all_actions.add(action)
                total_q_entries += 1
        
        metrics.action_diversity = len(all_actions)
        metrics.total_actions = total_q_entries
    
    def _calculate_environmental_metrics(self, environment: Environment, metrics: SimulationMetrics):
        """Рассчитывает средовые метрики"""
        stats = environment.get_statistics()
        
        metrics.resource_density = stats.get('occupation_rate', 0.0)
        metrics.object_types_distribution = stats.get('object_types', {})
        metrics.season = environment.season
        metrics.temperature = environment.temperature
    
    def _calculate_economic_metrics(self, agents: List[Agent], metrics: SimulationMetrics):
        """Рассчитывает экономические метрики"""
        if not agents:
            return
        
        # Ресурсы в инвентаре
        inventory_sizes = [len(agent.inventory) for agent in agents]
        metrics.average_inventory_size = np.mean(inventory_sizes)
        
        # Общее количество ресурсов (упрощенно)
        metrics.total_resources_collected = sum(len(agent.inventory) for agent in agents)
        
        # Потребление (на основе статистической памяти)
        total_consumed = 0
        for agent in agents:
            consume_success = agent.statistical_memory.get_statistic("consume_success")
            total_consumed += consume_success * 10  # Упрощенная оценка
        
        metrics.total_resources_consumed = total_consumed
    
    def _calculate_learning_metrics(self, agents: List[Agent], metrics: SimulationMetrics):
        """Рассчитывает метрики обучения"""
        if not agents:
            return
        
        # Размеры Q-таблиц
        q_table_sizes = [len(agent.q_table) for agent in agents]
        metrics.average_q_table_size = np.mean(q_table_sizes)
        
        # Эпизоды обучения (упрощенно)
        total_episodes = sum(len(agent.episodic_memory.episodes) for agent in agents)
        metrics.total_learning_episodes = total_episodes
        
        # Метрики сходимости
        metrics.convergence_metrics = self._calculate_convergence_metrics(agents)
    
    def _calculate_convergence_metrics(self, agents: List[Agent]) -> Dict[str, float]:
        """Рассчитывает метрики сходимости обучения"""
        if not agents:
            return {}
        
        # Дисперсия Q-значений (меньше = больше сходимости)
        all_q_values = []
        for agent in agents:
            all_q_values.extend(agent.q_table.values())
        
        if all_q_values:
            q_variance = np.var(all_q_values)
            q_std = np.std(all_q_values)
        else:
            q_variance = 0.0
            q_std = 0.0
        
        return {
            'q_value_variance': q_variance,
            'q_value_std': q_std,
            'average_q_value': np.mean(all_q_values) if all_q_values else 0.0
        }
    
    def get_metrics_history(self, limit: Optional[int] = None) -> List[SimulationMetrics]:
        """Возвращает историю метрик"""
        if limit is None:
            return self.metrics_history.copy()
        return self.metrics_history[-limit:]
    
    def calculate_summary_statistics(self) -> Dict[str, Any]:
        """Рассчитывает сводную статистику за всю симуляцию"""
        if not self.metrics_history:
            return {}
        
        # Собираем значения по всем метрикам
        data = defaultdict(list)
        
        for metrics in self.metrics_history:
            data['population_size'].append(metrics.population_size)
            data['total_discoveries'].append(metrics.total_discoveries)
            data['genetic_diversity'].append(metrics.genetic_diversity)
            data['average_intelligence'].append(metrics.average_intelligence)
            data['total_tools'].append(metrics.total_tools)
            data['action_diversity'].append(metrics.action_diversity)
        
        summary = {}
        for key, values in data.items():
            if values:
                summary[key] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'min': np.min(values),
                    'max': np.max(values),
                    'final': values[-1],
                    'trend': np.polyfit(range(len(values)), values, 1)[0] if len(values) > 1 else 0.0
                }
        
        return summary
    
    def export_metrics(self, filepath: str, format: str = 'json'):
        """Экспортирует метрики в файл"""
        data = {
            'summary': self.calculate_summary_statistics(),
            'history': [metrics.to_dict() for metrics in self.metrics_history]
        }
        
        if format.lower() == 'json':
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2)
        else:
            raise ValueError(f"Unsupported format: {format}")


class MetricsAnalyzer:
    """Анализатор метрик для выявления паттернов и трендов"""
    
    def __init__(self, metrics_history: List[SimulationMetrics]):
        self.metrics_history = metrics_history
    
    def analyze_technology_progression(self) -> Dict[str, Any]:
        """Анализирует прогресс технологий"""
        if not self.metrics_history:
            return {}
        
        discoveries = [m.total_discoveries for m in self.metrics_history]
        tools = [m.total_tools for m in self.metrics_history]
        complexity = [m.average_tool_complexity for m in self.metrics_history]
        
        # Находим ключевые моменты
        discovery_acceleration = self._find_acceleration_points(discoveries)
        tool_surges = self._find_surge_points(tools)
        
        return {
            'total_discoveries': discoveries[-1] if discoveries else 0,
            'discovery_trend': np.polyfit(range(len(discoveries)), discoveries, 1)[0] if len(discoveries) > 1 else 0.0,
            'discovery_acceleration_points': discovery_acceleration,
            'tool_surge_points': tool_surges,
            'final_complexity': complexity[-1] if complexity else 0.0,
            'complexity_trend': np.polyfit(range(len(complexity)), complexity, 1)[0] if len(complexity) > 1 else 0.0
        }
    
    def analyze_evolution_patterns(self) -> Dict[str, Any]:
        """Анализирует эволюционные паттерны"""
        if not self.metrics_history:
            return {}
        
        diversity = [m.genetic_diversity for m in self.metrics_history]
        intelligence = [m.average_intelligence for m in self.metrics_history]
        exploration = [m.average_exploration_bias for m in self.metrics_history]
        population = [m.population_size for m in self.metrics_history]
        
        # Корреляции между метриками
        correlations = {
            'diversity_intelligence': np.corrcoef(diversity, intelligence)[0, 1] if len(diversity) > 1 else 0.0,
            'diversity_population': np.corrcoef(diversity, population)[0, 1] if len(diversity) > 1 else 0.0,
            'intelligence_discoveries': np.corrcoef(intelligence, [m.total_discoveries for m in self.metrics_history])[0, 1] if len(intelligence) > 1 else 0.0
        }
        
        return {
            'genetic_diversity_trend': np.polyfit(range(len(diversity)), diversity, 1)[0] if len(diversity) > 1 else 0.0,
            'intelligence_evolution': np.polyfit(range(len(intelligence)), intelligence, 1)[0] if len(intelligence) > 1 else 0.0,
            'exploration_evolution': np.polyfit(range(len(exploration)), exploration, 1)[0] if len(exploration) > 1 else 0.0,
            'correlations': correlations,
            'diversity_peaks': self._find_peaks(diversity),
            'intelligence_peaks': self._find_peaks(intelligence)
        }
    
    def analyze_cultural_transmission(self) -> Dict[str, Any]:
        """Анализирует культурную передачу"""
        if not self.metrics_history:
            return {}
        
        action_diversity = [m.action_diversity for m in self.metrics_history]
        learning_episodes = [m.total_learning_episodes for m in self.metrics_history]
        
        # Анализ скорости распространения знаний
        knowledge_spread_rate = self._calculate_knowledge_spread_rate()
        
        return {
            'action_diversity_evolution': np.polyfit(range(len(action_diversity)), action_diversity, 1)[0] if len(action_diversity) > 1 else 0.0,
            'learning_acceleration': np.polyfit(range(len(learning_episodes)), learning_episodes, 1)[0] if len(learning_episodes) > 1 else 0.0,
            'knowledge_spread_rate': knowledge_spread_rate,
            'cultural_complexity_index': self._calculate_cultural_complexity_index()
        }
    
    def _find_acceleration_points(self, values: List[float], threshold: float = 2.0) -> List[int]:
        """Находит точки ускорения роста"""
        if len(values) < 3:
            return []
        
        acceleration_points = []
        for i in range(1, len(values) - 1):
            prev_rate = values[i] - values[i-1]
            curr_rate = values[i+1] - values[i]
            
            if prev_rate > 0 and curr_rate > prev_rate * threshold:
                acceleration_points.append(i)
        
        return acceleration_points
    
    def _find_surge_points(self, values: List[float], threshold: float = 1.5) -> List[int]:
        """Находит точки резкого роста"""
        if len(values) < 2:
            return []
        
        surge_points = []
        mean_value = np.mean(values)
        
        for i in range(1, len(values)):
            if values[i] > mean_value * threshold and values[i-1] <= mean_value:
                surge_points.append(i)
        
        return surge_points
    
    def _find_peaks(self, values: List[float]) -> List[Tuple[int, float]]:
        """Находит пики в значениях"""
        if len(values) < 3:
            return []
        
        peaks = []
        for i in range(1, len(values) - 1):
            if values[i] > values[i-1] and values[i] > values[i+1]:
                peaks.append((i, values[i]))
        
        return peaks
    
    def _calculate_knowledge_spread_rate(self) -> float:
        """Рассчитывает скорость распространения знаний"""
        action_diversity = [m.action_diversity for m in self.metrics_history]
        
        if len(action_diversity) < 2:
            return 0.0
        
        # Скорость изменения разнообразия действий
        changes = []
        for i in range(1, len(action_diversity)):
            change = action_diversity[i] - action_diversity[i-1]
            changes.append(change)
        
        return np.mean(changes) if changes else 0.0
    
    def _calculate_cultural_complexity_index(self) -> float:
        """Рассчитывает индекс культурной сложности"""
        if not self.metrics_history:
            return 0.0
        
        latest_metrics = self.metrics_history[-1]
        
        # Комбинированный индекс на основе разных метрик
        action_diversity = latest_metrics.action_diversity
        tool_complexity = latest_metrics.average_tool_complexity
        learning_episodes = latest_metrics.total_learning_episodes
        
        # Нормализация
        normalized_diversity = min(1.0, action_diversity / 10.0)
        normalized_complexity = min(1.0, tool_complexity)
        normalized_learning = min(1.0, learning_episodes / 1000.0)
        
        return (normalized_diversity + normalized_complexity + normalized_learning) / 3.0
    
    def generate_insights(self) -> List[str]:
        """Генерирует выводы на основе анализа метрик"""
        insights = []
        
        tech_analysis = self.analyze_technology_progression()
        evo_analysis = self.analyze_evolution_patterns()
        cultural_analysis = self.analyze_cultural_transmission()
        
        # Технологические выводы
        if tech_analysis.get('discovery_trend', 0) > 0.1:
            insights.append("Технологический прогресс ускоряется со временем")
        elif tech_analysis.get('discovery_trend', 0) < -0.1:
            insights.append("Технологический прогресс замедляется")
        
        if tech_analysis.get('discovery_acceleration_points'):
            insights.append(f"Обнаружено {len(tech_analysis['discovery_acceleration_points'])} точек ускорения открытий")
        
        # Эволюционные выводы
        if evo_analysis.get('genetic_diversity_trend', 0) > 0.01:
            insights.append("Генетическое разнообразие увеличивается")
        elif evo_analysis.get('genetic_diversity_trend', 0) < -0.01:
            insights.append("Генетическое разнообразие уменьшается (риск вырождения)")
        
        # Культурные выводы
        if cultural_analysis.get('knowledge_spread_rate', 0) > 0.5:
            insights.append("Знания распространяются быстро")
        elif cultural_analysis.get('knowledge_spread_rate', 0) < -0.5:
            insights.append("Знания распространяются медленно")
        
        return insights
