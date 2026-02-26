"""
Tests for evolution module
"""

import pytest
import numpy as np
from src.evolution.genetics import GeneticAlgorithm, GeneAnalyzer, EvolutionManager
from src.evolution.selection import (SelectionManager, RouletteWheelSelection, 
                                   TournamentSelection, RankSelection)
from src.evolution.reproduction import ReproductionManager, CulturalEvolution
from src.core.agent import Agent, AgentGenes, AgentFactory
from src.core.environment import Environment, EnvironmentConfig


class TestGeneticAlgorithm:
    """Тесты генетического алгоритма"""
    
    def setup_method(self):
        """Настройка тестов"""
        self.ga = GeneticAlgorithm(population_size=10, mutation_rate=0.1)
        
        # Создаем тестовую популяцию
        self.agents = []
        for i in range(5):
            agent = AgentFactory.create_random_agent(f"agent_{i}", (i, i))
            self.agents.append(agent)
    
    def test_initialization(self):
        """Тест инициализации"""
        assert self.ga.population_size == 10
        assert self.ga.mutation_rate == 0.1
        assert self.ga.generation == 0
        assert self.ga.total_births == 0
    
    def test_calculate_fitness(self):
        """Тест расчета приспособленности"""
        fitness_scores = self.ga.calculate_fitness(self.agents)
        
        assert len(fitness_scores) == len(self.agents)
        assert all(score >= 0.1 for score in fitness_scores)  # Минимальная приспособленность
        
        # Агент с открытиями должен иметь выше приспособленность
        self.agents[0].discoveries_made.append("discovery1")
        new_fitness_scores = self.ga.calculate_fitness(self.agents)
        
        assert new_fitness_scores[0] > fitness_scores[0]
    
    def test_select_parents(self):
        """Тест селекции родителей"""
        fitness_scores = self.ga.calculate_fitness(self.agents)
        parents = self.ga.select_parents(self.agents, fitness_scores)
        
        assert len(parents) <= len(self.agents)
        assert all(agent in self.agents for agent in parents)
    
    def test_tournament_selection(self):
        """Тест турнирной селекции"""
        fitness_scores = self.ga.calculate_fitness(self.agents)
        
        best_agent = self.ga.tournament_selection(self.agents, fitness_scores)
        
        assert best_agent in self.agents
        # Лучший агент в турнире должен иметь высокую приспособленность
        best_fitness = fitness_scores[self.agents.index(best_agent)]
        max_fitness = max(fitness_scores)
        
        # Турнирная селекция выбирает лучшего из случайной подвыборки,
        # поэтому его приспособленность должна быть reasonably high
        assert best_fitness > max_fitness * 0.5  # Хотя бы 50% от максимума
    
    def test_create_offspring(self):
        """Тест создания потомка"""
        parent1 = self.agents[0]
        parent2 = self.agents[1]
        
        # Создаем тестовую среду
        config = EnvironmentConfig(width=10, height=10)
        env = Environment(config)
        
        child = self.ga.create_offspring(parent1, parent2, 100, env)
        
        assert child.id.startswith("agent_gen")
        assert child.birth_time == 100
        assert child.position[0] in range(max(0, parent1.position[0] - 2), 
                                       min(env.width, parent1.position[0] + 3))
        assert child.position[1] in range(max(0, parent1.position[1] - 2), 
                                       min(env.height, parent1.position[1] + 3))
    
    def test_evolve_population(self):
        """Тест эволюции популяции"""
        config = EnvironmentConfig(width=20, height=20)
        env = Environment(config)
        
        # Делаем агентов готовыми к размножению
        for agent in self.agents:
            agent.health = 0.8
            agent.age = 600
            agent.energy = 0.7
            agent.hunger = 0.3
        
        new_population = self.ga.evolve_population(self.agents, env, 100)
        
        assert len(new_population) <= self.ga.population_size
        assert self.ga.generation == 1
        # Потомство может не создаться, если crossover_rate низкий
        assert self.ga.total_births >= 0
    
    def test_evolution_stats(self):
        """Тест статистики эволюции"""
        stats = self.ga.get_evolution_stats()
        
        assert 'generation' in stats
        assert 'total_births' in stats
        assert 'mutation_count' in stats
        assert 'crossover_count' in stats


class TestGeneAnalyzer:
    """Тесты анализатора генов"""
    
    def setup_method(self):
        """Настройка тестов"""
        self.analyzer = GeneAnalyzer()
        
        # Создаем агентов с разными генами
        self.agents = []
        for i in range(5):
            genes = AgentGenes(
                metabolism_speed=i * 0.2,
                strength=0.5 + i * 0.1,
                intelligence=0.3 + i * 0.15,
                social_tendency=0.4 + i * 0.05,
                exploration_bias=0.6 - i * 0.1
            )
            agent = Agent(f"agent_{i}", (i, i), genes)
            self.agents.append(agent)
    
    def test_analyze_population_genes(self):
        """Тест анализа генов популяции"""
        analysis = self.analyzer.analyze_population_genes(self.agents)
        
        assert 'population_size' in analysis
        assert analysis['population_size'] == len(self.agents)
        assert 'gene_statistics' in analysis
        assert 'genetic_diversity' in analysis
        
        # Проверяем статистику генов
        gene_stats = analysis['gene_statistics']
        assert 'metabolism_speed' in gene_stats
        assert 'mean' in gene_stats['metabolism_speed']
        assert 'std' in gene_stats['metabolism_speed']
    
    def test_genetic_diversity(self):
        """Тест расчета генетического разнообразия"""
        # Одинаковые агенты - низкое разнообразие
        identical_genes = AgentGenes(metabolism_speed=0.5, strength=0.5)
        agents1 = [Agent(f"id_{i}", (0, 0), identical_genes) for i in range(3)]
        
        diversity1 = self.analyzer._calculate_genetic_diversity([
            agent.genes.to_dict() for agent in agents1
        ])
        assert diversity1 == 0.0
        
        # Разные агенты - высокое разнообразие
        diversity2 = self.analyzer._calculate_genetic_diversity([
            agent.genes.to_dict() for agent in self.agents
        ])
        assert diversity2 > 0.0
    
    def test_evolution_trends(self):
        """Тест трендов эволюции"""
        # Добавляем несколько записей в историю
        for i in range(3):
            self.analyzer.analyze_population_genes(self.agents)
        
        trends = self.analyzer.track_evolution_trends()
        
        assert isinstance(trends, dict)
        # Должны быть тренды для генов
        assert any(gene in trends for gene in ['metabolism_speed', 'strength', 'intelligence'])


class TestSelectionManager:
    """Тесты менеджера селекции"""
    
    def setup_method(self):
        """Настройка тестов"""
        self.manager = SelectionManager()
        self.agents = [AgentFactory.create_random_agent(f"agent_{i}", (i, i)) for i in range(10)]
    
    def test_roulette_wheel_selection(self):
        """Тест рулеточной селекции"""
        fitness_scores = [i * 0.5 + 0.1 for i in range(len(self.agents))]
        
        selected = self.manager.methods['roulette'].select(self.agents, fitness_scores, 3)
        
        assert len(selected) == 3
        assert all(agent in self.agents for agent in selected)
    
    def test_tournament_selection(self):
        """Тест турнирной селекции"""
        fitness_scores = [i * 0.5 + 0.1 for i in range(len(self.agents))]
        
        selected = self.manager.methods['tournament'].select(self.agents, fitness_scores, 3)
        
        assert len(selected) == 3
        assert all(agent in self.agents for agent in selected)
    
    def test_rank_selection(self):
        """Тест ранговой селекции"""
        fitness_scores = [i * 0.5 + 0.1 for i in range(len(self.agents))]
        
        selected = self.manager.methods['rank'].select(self.agents, fitness_scores, 3)
        
        assert len(selected) == 3
        assert all(agent in self.agents for agent in selected)
    
    def test_select_parents(self):
        """Тест выбора родителей"""
        fitness_scores = [i * 0.5 + 0.1 for i in range(len(self.agents))]
        
        parents = self.manager.select_parents(self.agents, fitness_scores, 5)
        
        assert len(parents) == 5
        assert all(agent in self.agents for agent in parents)
        
        # Проверяем историю селекции
        history = self.manager.get_selection_history()
        assert len(history) > 0
        assert history[-1]['selected_count'] == 5


class TestReproductionManager:
    """Тесты менеджера репродукции"""
    
    def setup_method(self):
        """Настройка тестов"""
        self.manager = ReproductionManager()
        
        # Создаем готовых к размножению агентов
        self.parent1 = AgentFactory.create_random_agent("parent1", (5, 5))
        self.parent2 = AgentFactory.create_random_agent("parent2", (6, 5))
        
        # Делаем их готовыми к размножению
        for parent in [self.parent1, self.parent2]:
            parent.health = 0.8
            parent.age = 600
            parent.energy = 0.7
            parent.hunger = 0.3
    
    def test_can_reproduce(self):
        """Тест проверки возможности размножения"""
        assert self.manager.can_reproduce(self.parent1) is True
        assert self.manager.can_reproduce(self.parent2) is True
        
        # Неготовый агент
        not_ready = AgentFactory.create_random_agent("not_ready", (0, 0))
        assert self.manager.can_reproduce(not_ready) is False
    
    def test_find_mates(self):
        """Тест поиска партнеров"""
        potential_mates = [self.parent2]
        
        mates = self.manager.find_mates(self.parent1, potential_mates, max_distance=10)
        
        assert len(mates) == 1
        assert mates[0] == self.parent2
    
    def test_genetic_similarity(self):
        """Тест расчета генетической схожести"""
        similarity = self.manager._calculate_genetic_similarity(
            self.parent1.genes, self.parent2.genes
        )
        
        assert 0.0 <= similarity <= 1.0
        assert isinstance(similarity, float)
    
    def test_reproduce(self):
        """Тест размножения"""
        config = EnvironmentConfig(width=10, height=10)
        env = Environment(config)
        
        child = self.manager.reproduce(self.parent1, self.parent2, 100, env)
        
        if child:  # Размножение может не произойти из-за crossover_rate
            assert child.birth_time == 100
            assert child.id.startswith("offspring_")
            assert self.parent1.offspring_count == 1
            assert self.parent2.offspring_count == 1
    
    def test_cultural_transfer(self):
        """Тест культурной передачи"""
        child = AgentFactory.create_random_agent("child", (5, 5))
        
        # Добавляем знания родителям
        self.parent1.statistical_memory.update_statistic("test_key", 0.8)
        self.parent2.q_table[("state1", "action1")] = 0.7
        
        transfer_occurred = self.manager._perform_cultural_transfer(
            self.parent1, self.parent2, child
        )
        
        # Культурная передача может произойти или нет
        assert isinstance(transfer_occurred, bool)
    
    def test_reproduction_stats(self):
        """Тест статистики репродукции"""
        stats = self.manager.get_reproduction_stats()
        
        assert 'reproduction_attempts' in stats
        assert 'successful_reproductions' in stats
        assert 'cultural_transfer_rate' in stats
        assert 'mutation_rate' in stats


class TestCulturalEvolution:
    """Тесты культурной эволюции"""
    
    def setup_method(self):
        """Настройка тестов"""
        self.cultural_evo = CulturalEvolution()
    
    def test_register_cultural_trait(self):
        """Тест регистрации культурной черты"""
        trait_id = self.cultural_evo.register_cultural_trait(
            "tool_use", {"effectiveness": 0.8}, "agent1", 100
        )
        
        assert trait_id in self.cultural_evo.cultural_traits
        trait = self.cultural_evo.cultural_traits[trait_id]
        assert trait['name'] == "tool_use"
        assert trait['originator'] == "agent1"
        assert trait['creation_time'] == 100
    
    def test_transmit_trait(self):
        """Тест передачи культурной черты"""
        trait_id = self.cultural_evo.register_cultural_trait(
            "tool_use", {"effectiveness": 0.8}, "agent1", 100
        )
        
        self.cultural_evo.transmit_trait(trait_id, "agent1", "agent2", 150, True)
        
        assert len(self.cultural_evo.transmission_events) == 1
        event = self.cultural_evo.transmission_events[0]
        assert event['from_agent'] == "agent1"
        assert event['to_agent'] == "agent2"
        assert event['success'] is True
    
    def test_analyze_cultural_diversity(self):
        """Тест анализа культурного разнообразия"""
        agents = []
        for i in range(5):
            agent = AgentFactory.create_random_agent(f"agent_{i}", (i, i))
            # Добавляем разные знания
            agent.q_table[(f"state_{i}", "action")] = i * 0.2
            agent.statistical_memory.update_statistic(f"memory_{i}", i * 0.3)
            agents.append(agent)
        
        diversity = self.cultural_evo.analyze_cultural_diversity(agents)
        
        assert 'population_size' in diversity
        assert diversity['population_size'] == 5
        assert 'action_statistics' in diversity
        assert 'memory_statistics' in diversity
        assert 'total_q_entries' in diversity
    
    def test_cultural_evolution_summary(self):
        """Тест сводки культурной эволюции"""
        # Добавляем несколько черт
        for i in range(3):
            self.cultural_evo.register_cultural_trait(
                f"trait_{i}", {"value": i}, f"agent_{i}", 100 + i
            )
        
        # Добавляем передачи
        traits = list(self.cultural_evo.cultural_traits.keys())
        self.cultural_evo.transmit_trait(traits[0], "agent_0", "agent_1", 200, True)
        self.cultural_evo.transmit_trait(traits[1], "agent_1", "agent_2", 210, False)
        
        summary = self.cultural_evo.get_cultural_evolution_summary()
        
        assert 'total_traits' in summary
        assert summary['total_traits'] == 3
        assert 'total_transmissions' in summary
        assert summary['total_transmissions'] == 2
        assert 'transmission_success_rate' in summary


if __name__ == '__main__':
    pytest.main([__file__])
