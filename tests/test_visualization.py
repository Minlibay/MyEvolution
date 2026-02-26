"""
Tests for visualization and metrics modules
"""

import pytest
import numpy as np
from src.utils.metrics import SimulationMetrics, MetricsCalculator, MetricsAnalyzer
from src.utils.visualization import SimulationVisualizer
from src.core.agent import Agent, AgentGenes, AgentFactory
from src.core.environment import Environment, EnvironmentConfig


class TestSimulationMetrics:
    """Тесты метрик симуляции"""
    
    def test_metrics_initialization(self):
        """Тест инициализации метрик"""
        metrics = SimulationMetrics(100)
        
        assert metrics.timestamp == 100
        assert metrics.population_size == 0
        assert metrics.total_discoveries == 0
        assert metrics.genetic_diversity == 0.0
    
    def test_metrics_to_dict(self):
        """Тест преобразования метрик в словарь"""
        metrics = SimulationMetrics(100)
        metrics.population_size = 10
        metrics.total_discoveries = 5
        metrics.genetic_diversity = 0.5
        
        data = metrics.to_dict()
        
        assert isinstance(data, dict)
        assert data['timestamp'] == 100
        assert 'population' in data
        assert 'technology' in data
        assert 'genetics' in data
        assert data['population']['size'] == 10
        assert data['technology']['total_discoveries'] == 5
        assert data['genetics']['diversity'] == 0.5


class TestMetricsCalculator:
    """Тесты калькулятора метрик"""
    
    def setup_method(self):
        """Настройка тестов"""
        self.calculator = MetricsCalculator()
        
        # Создаем тестовую среду и агентов
        config = EnvironmentConfig(width=10, height=10)
        self.environment = Environment(config)
        
        self.agents = []
        for i in range(5):
            agent = AgentFactory.create_random_agent(f"agent_{i}", (i, i))
            # Устанавливаем разные значения для тестирования
            agent.health = 0.5 + i * 0.1
            agent.age = 100 + i * 50
            agent.energy = 0.6 + i * 0.05
            self.agents.append(agent)
    
    def test_calculate_metrics(self):
        """Тест расчета метрик"""
        metrics = self.calculator.calculate_metrics(self.agents, self.environment, 100)
        
        assert metrics.timestamp == 100
        assert metrics.population_size == len(self.agents)
        assert metrics.average_age > 0
        assert metrics.average_health > 0
        assert metrics.average_energy > 0
    
    def test_population_metrics(self):
        """Тест популяционных метрик"""
        metrics = self.calculator.calculate_metrics(self.agents, self.environment, 100)
        
        # Проверяем средние значения
        expected_health = np.mean([agent.health for agent in self.agents])
        expected_age = np.mean([agent.age for agent in self.agents])
        
        assert abs(metrics.average_health - expected_health) < 0.01
        assert abs(metrics.average_age - expected_age) < 0.01
    
    def test_genetic_metrics(self):
        """Тест генетических метрик"""
        metrics = self.calculator.calculate_metrics(self.agents, self.environment, 100)
        
        assert metrics.genetic_diversity >= 0.0
        assert metrics.average_intelligence >= 0.0
        assert metrics.average_exploration_bias >= 0.0
        assert metrics.average_strength >= 0.0
    
    def test_genetic_diversity(self):
        """Тест расчета генетического разнообразия"""
        # Одинаковые гены - нулевое разнообразие
        identical_genes = AgentGenes(metabolism_speed=0.5, strength=0.5)
        agents1 = [Agent(f"id_{i}", (0, 0), identical_genes) for i in range(3)]
        
        diversity1 = self.calculator._calculate_genetic_diversity([
            agent.genes.to_dict() for agent in agents1
        ])
        assert diversity1 == 0.0
        
        # Разные гены - положительное разнообразие
        diversity2 = self.calculator._calculate_genetic_diversity([
            agent.genes.to_dict() for agent in self.agents
        ])
        assert diversity2 > 0.0
    
    def test_technology_metrics(self):
        """Тест технологических метрик"""
        # Добавляем инструменты агентам
        for i, agent in enumerate(self.agents[:3]):
            agent.tools = [f"tool_{i}"]
            agent.discoveries_made = [f"discovery_{i}"]
        
        metrics = self.calculator.calculate_metrics(self.agents, self.environment, 100)
        
        assert metrics.total_tools == 3
        assert metrics.total_discoveries == 3
    
    def test_metrics_history(self):
        """Тест истории метрик"""
        # Рассчитываем метрики несколько раз
        for i in range(3):
            self.calculator.calculate_metrics(self.agents, self.environment, i * 100)
        
        history = self.calculator.get_metrics_history()
        
        assert len(history) == 3
        assert history[0].timestamp == 0
        assert history[1].timestamp == 100
        assert history[2].timestamp == 200
    
    def test_summary_statistics(self):
        """Тест сводной статистики"""
        # Рассчитываем метрики несколько раз
        for i in range(5):
            self.calculator.calculate_metrics(self.agents, self.environment, i)
        
        summary = self.calculator.calculate_summary_statistics()
        
        assert 'population_size' in summary
        assert 'total_discoveries' in summary
        assert 'genetic_diversity' in summary
        
        # Проверяем структуру статистики
        pop_stats = summary['population_size']
        assert 'mean' in pop_stats
        assert 'std' in pop_stats
        assert 'min' in pop_stats
        assert 'max' in pop_stats
        assert 'final' in pop_stats
        assert 'trend' in pop_stats


class TestMetricsAnalyzer:
    """Тесты анализатора метрик"""
    
    def setup_method(self):
        """Настройка тестов"""
        # Создаем историю метрик
        self.metrics_history = []
        
        for i in range(10):
            metrics = SimulationMetrics(i)
            metrics.population_size = 10 + i
            metrics.total_discoveries = i * 2
            metrics.genetic_diversity = 0.5 + i * 0.05
            metrics.average_intelligence = 0.4 + i * 0.03
            metrics.total_tools = i
            metrics.action_diversity = 5 + i
            
            self.metrics_history.append(metrics)
        
        self.analyzer = MetricsAnalyzer(self.metrics_history)
    
    def test_analyze_technology_progression(self):
        """Тест анализа технологического прогресса"""
        analysis = self.analyzer.analyze_technology_progression()
        
        assert 'total_discoveries' in analysis
        assert 'discovery_trend' in analysis
        assert 'final_complexity' in analysis
        
        assert analysis['total_discoveries'] == 18  # 9 * 2
        assert analysis['discovery_trend'] > 0  # Рост
    
    def test_analyze_evolution_patterns(self):
        """Тест анализа эволюционных паттернов"""
        analysis = self.analyzer.analyze_evolution_patterns()
        
        assert 'genetic_diversity_trend' in analysis
        assert 'intelligence_evolution' in analysis
        assert 'correlations' in analysis
        
        assert analysis['genetic_diversity_trend'] > 0  # Рост разнообразия
        assert analysis['intelligence_evolution'] > 0  # Рост интеллекта
    
    def test_analyze_cultural_transmission(self):
        """Тест анализа культурной передачи"""
        analysis = self.analyzer.analyze_cultural_transmission()
        
        assert 'action_diversity_evolution' in analysis
        assert 'learning_acceleration' in analysis
        assert 'cultural_complexity_index' in analysis
        
        assert analysis['action_diversity_evolution'] > 0  # Рост разнообразия действий
    
    def test_find_acceleration_points(self):
        """Тест поиска точек ускорения"""
        values = [1, 2, 3, 4, 10, 20, 30]  # Резкое ускорение после индекса 4
        points = self.analyzer._find_acceleration_points(values, threshold=1.5)
        
        assert len(points) > 0
        # Точка ускорения может быть в разных местах в зависимости от алгоритма
        assert points[0] >= 3  # В районе ускорения
    
    def test_find_peaks(self):
        """Тест поиска пиков"""
        values = [1, 3, 5, 3, 1, 4, 6, 4, 2]  # Пики в индексах 2 и 6
        peaks = self.analyzer._find_peaks(values)
        
        assert len(peaks) == 2
        assert peaks[0][0] == 2  # Первый пик
        assert peaks[1][0] == 6  # Второй пик
        assert peaks[0][1] == 5  # Значение первого пика
        assert peaks[1][1] == 6  # Значение второго пика
    
    def test_generate_insights(self):
        """Тест генерации выводов"""
        insights = self.analyzer.generate_insights()
        
        assert isinstance(insights, list)
        assert len(insights) > 0
        
        # Проверяем, что выводы содержат ожидаемые паттерны
        insight_text = ' '.join(insights)
        assert 'технологический' in insight_text.lower() or 'генетическое' in insight_text.lower()


class TestSimulationVisualizer:
    """Тесты визуализатора симуляции"""
    
    def setup_method(self):
        """Настройка тестов"""
        self.visualizer = SimulationVisualizer()
        
        # Создаем тестовую среду и агентов
        config = EnvironmentConfig(width=10, height=10)
        self.environment = Environment(config)
        
        self.agents = []
        for i in range(3):
            agent = AgentFactory.create_random_agent(f"agent_{i}", (i, i))
            agent.health = 0.8
            self.agents.append(agent)
    
    def test_initialization(self):
        """Тест инициализации визуализатора"""
        assert self.visualizer.figsize == (15, 10)
        assert len(self.visualizer.color_palette) == 10
    
    def test_get_object_color(self):
        """Тест получения цвета объекта"""
        assert self.visualizer._get_object_color('stone') == 'gray'
        assert self.visualizer._get_object_color('wood') == 'brown'
        assert self.visualizer._get_object_color('plant') == 'green'
        assert self.visualizer._get_object_color('unknown') == 'blue'
    
    def test_get_agent_color(self):
        """Тест получения цвета агента"""
        # Здоровый агент
        healthy_agent = Agent("test", (0, 0), AgentGenes())
        healthy_agent.health = 0.8
        assert self.visualizer._get_agent_color(healthy_agent) == 'green'
        
        # Раненый агент
        injured_agent = Agent("test", (0, 0), AgentGenes())
        injured_agent.health = 0.5
        assert self.visualizer._get_agent_color(injured_agent) == 'yellow'
        
        # Больной агент
        sick_agent = Agent("test", (0, 0), AgentGenes())
        sick_agent.health = 0.2
        assert self.visualizer._get_agent_color(sick_agent) == 'red'
    
    def test_get_object_size(self):
        """Тест получения размера объекта"""
        from src.core.objects import Object
        
        # Легкий объект
        light_obj = Object("test", "stone", (0, 0), 0.5, 0.0, 0.0, 0.3, 0.5, 0.0, 0.2, 0)
        light_size = self.visualizer._get_object_size(light_obj)
        
        # Тяжелый объект
        heavy_obj = Object("test", "stone", (0, 0), 0.5, 0.0, 0.0, 0.8, 0.5, 0.0, 0.2, 0)
        heavy_size = self.visualizer._get_object_size(heavy_obj)
        
        assert heavy_size > light_size


if __name__ == '__main__':
    pytest.main([__file__])
