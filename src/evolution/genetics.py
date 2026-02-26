"""
Genetics module - genetic algorithms and evolution mechanisms
"""

from typing import Dict, List, Tuple, Any, Optional
import random
import numpy as np

from ..core.agent import Agent, AgentGenes, AgentFactory


class GeneticAlgorithm:
    """Основной класс генетического алгоритма"""
    
    def __init__(self, population_size: int = 50, mutation_rate: float = 0.1,
                 crossover_rate: float = 0.8, selection_pressure: float = 0.8):
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.selection_pressure = selection_pressure
        
        # Статистика эволюции
        self.generation = 0
        self.total_births = 0
        self.total_deaths = 0
        self.mutation_count = 0
        self.crossover_count = 0
    
    def calculate_fitness(self, agents: List[Agent]) -> List[float]:
        """Рассчитывает приспособленность агентов"""
        fitness_scores = []
        
        for agent in agents:
            fitness = self._calculate_individual_fitness(agent)
            fitness_scores.append(fitness)
        
        return fitness_scores
    
    def _calculate_individual_fitness(self, agent: Agent) -> float:
        """Рассчитывает приспособленность отдельного агента"""
        # Базовая приспособленность based on выживаемость
        base_fitness = agent.age * agent.health
        
        # Бонус за открытия
        discovery_bonus = len(agent.discoveries_made) * 10
        
        # Бонус за репродукцию
        reproduction_bonus = agent.offspring_count * 5
        
        # Бонус за накопленные ресурсы
        resource_bonus = len(agent.inventory) * 2
        
        # Бонус за инструменты
        tool_bonus = len(agent.tools) * 3
        
        # Штраф за низкую энергию
        energy_penalty = (1.0 - agent.energy) * 2
        
        # Штраф за высокий голод
        hunger_penalty = agent.hunger * 3
        
        # Бонус за хорошие гены
        gene_bonus = (
            agent.genes.intelligence * 5 +
            agent.genes.strength * 3 +
            agent.genes.exploration_bias * 4
        )
        
        total_fitness = (
            base_fitness + discovery_bonus + reproduction_bonus + 
            resource_bonus + tool_bonus + gene_bonus - 
            energy_penalty - hunger_penalty
        )
        
        return max(0.1, total_fitness)
    
    def select_parents(self, agents: List[Agent], fitness_scores: List[float]) -> List[Agent]:
        """Выбирает родителей для следующего поколения"""
        if len(agents) < 2:
            return agents
        
        # Нормализация оценок приспособленности
        total_fitness = sum(fitness_scores)
        if total_fitness == 0:
            probabilities = [1.0 / len(agents)] * len(agents)
        else:
            probabilities = [score / total_fitness for score in fitness_scores]
        
        # Рулеточная селекция
        parents = []
        num_parents = min(len(agents), int(self.population_size * self.selection_pressure))
        
        for _ in range(num_parents):
            parent = np.random.choice(agents, p=probabilities)
            parents.append(parent)
        
        return parents
    
    def tournament_selection(self, agents: List[Agent], fitness_scores: List[float],
                           tournament_size: int = 3) -> Agent:
        """Турнирная селекция"""
        tournament_indices = random.sample(range(len(agents)), 
                                         min(tournament_size, len(agents)))
        
        best_index = max(tournament_indices, key=lambda i: fitness_scores[i])
        return agents[best_index]
    
    def create_offspring(self, parent1: Agent, parent2: Agent, 
                        birth_time: int, environment) -> Agent:
        """Создает потомка от двух родителей"""
        # Генетический кроссовер
        child_genes = parent1.genes.crossover(parent2.genes)
        
        # Мутации
        child_genes.mutate(self.mutation_rate)
        if random.random() < self.mutation_rate:
            self.mutation_count += 1
        
        # Создание потомка
        child_id = f"agent_gen{self.generation}_{self.total_births}"
        child_position = self._get_birth_position(parent1, environment)
        
        child = AgentFactory.create_offspring(
            parent1, parent2, child_id, birth_time
        )
        
        # Обновляем счетчики
        self.total_births += 1
        self.crossover_count += 1
        
        return child
    
    def _get_birth_position(self, parent: Agent, environment) -> Tuple[int, int]:
        """Определяет позицию рождения потомка"""
        # Рождается near родителя
        x = parent.position[0] + random.randint(-2, 2)
        y = parent.position[1] + random.randint(-2, 2)
        
        # Проверяем границы
        x = max(0, min(environment.width - 1, x))
        y = max(0, min(environment.height - 1, y))
        
        return (x, y)
    
    def evolve_population(self, agents: List[Agent], environment, 
                          birth_time: int) -> List[Agent]:
        """Выполняет один шаг эволюции популяции"""
        if len(agents) < 2:
            return agents
        
        # Расчет приспособленности
        fitness_scores = self.calculate_fitness(agents)
        
        # Селекция родителей
        parents = self.select_parents(agents, fitness_scores)
        
        # Создание нового поколения
        new_population = agents.copy()  # Сохраняем лучших
        
        # Добавляем потомков
        offspring_count = 0
        max_offspring = min(len(parents) // 2, self.population_size - len(new_population))
        
        for i in range(0, len(parents) - 1, 2):
            if offspring_count >= max_offspring:
                break
            
            parent1 = parents[i]
            parent2 = parents[i + 1]
            
            # Проверяем, могут ли родители размножаться
            if parent1.can_reproduce() and parent2.can_reproduce():
                if random.random() < self.crossover_rate:
                    child = self.create_offspring(parent1, parent2, birth_time, environment)
                    new_population.append(child)
                    offspring_count += 1
        
        # Обновляем поколение
        self.generation += 1
        
        # Ограничиваем размер популяции
        if len(new_population) > self.population_size:
            # Сортируем по приспособленности и оставляем лучших
            new_fitness = self.calculate_fitness(new_population)
            sorted_indices = sorted(range(len(new_population)), 
                                 key=lambda i: new_fitness[i], reverse=True)
            new_population = [new_population[i] for i in sorted_indices[:self.population_size]]
        
        return new_population
    
    def get_evolution_stats(self) -> Dict[str, Any]:
        """Возвращает статистику эволюции"""
        return {
            'generation': self.generation,
            'total_births': self.total_births,
            'total_deaths': self.total_deaths,
            'mutation_count': self.mutation_count,
            'crossover_count': self.crossover_count,
            'mutation_rate': self.mutation_rate,
            'crossover_rate': self.crossover_rate,
            'selection_pressure': self.selection_pressure
        }


class GeneAnalyzer:
    """Анализ генетических изменений в популяции"""
    
    def __init__(self):
        self.gene_history = []
        self.diversity_metrics = []
    
    def analyze_population_genes(self, agents: List[Agent]) -> Dict[str, Any]:
        """Анализирует гены популяции"""
        if not agents:
            return {}
        
        # Собираем все гены
        all_genes = []
        for agent in agents:
            genes_dict = agent.genes.to_dict()
            all_genes.append(genes_dict)
        
        # Расчет статистики по каждому гену
        gene_stats = {}
        for gene_name in all_genes[0].keys():
            values = [genes[gene_name] for genes in all_genes]
            gene_stats[gene_name] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'min': np.min(values),
                'max': np.max(values),
                'range': np.max(values) - np.min(values)
            }
        
        # Расчет разнообразия
        diversity = self._calculate_genetic_diversity(all_genes)
        
        # Расчет корреляций между генами
        correlations = self._calculate_gene_correlations(all_genes)
        
        analysis = {
            'population_size': len(agents),
            'gene_statistics': gene_stats,
            'genetic_diversity': diversity,
            'gene_correlations': correlations,
            'timestamp': 0  # Будет установлено извне
        }
        
        self.gene_history.append(analysis)
        return analysis
    
    def _calculate_genetic_diversity(self, all_genes: List[Dict]) -> float:
        """Рассчитывает генетическое разнообразие"""
        if len(all_genes) < 2:
            return 0.0
        
        # Среднее расстояние между всеми парами генов
        total_distance = 0.0
        pair_count = 0
        
        for i in range(len(all_genes)):
            for j in range(i + 1, len(all_genes)):
                distance = self._gene_distance(all_genes[i], all_genes[j])
                total_distance += distance
                pair_count += 1
        
        return total_distance / pair_count if pair_count > 0 else 0.0
    
    def _gene_distance(self, genes1: Dict, genes2: Dict) -> float:
        """Рассчитывает расстояние между двумя наборами генов"""
        distance = 0.0
        for gene_name in genes1:
            distance += abs(genes1[gene_name] - genes2[gene_name])
        return distance / len(genes1)
    
    def _calculate_gene_correlations(self, all_genes: List[Dict]) -> Dict[str, Dict[str, float]]:
        """Рассчитывает корреляции между генами"""
        if len(all_genes) < 3:
            return {}
        
        gene_names = list(all_genes[0].keys())
        correlations = {}
        
        for gene1 in gene_names:
            correlations[gene1] = {}
            values1 = [genes[gene1] for genes in all_genes]
            
            for gene2 in gene_names:
                values2 = [genes[gene2] for genes in all_genes]
                
                if len(set(values1)) > 1 and len(set(values2)) > 1:
                    correlation = np.corrcoef(values1, values2)[0, 1]
                    correlations[gene1][gene2] = correlation if not np.isnan(correlation) else 0.0
                else:
                    correlations[gene1][gene2] = 0.0
        
        return correlations
    
    def track_evolution_trends(self) -> Dict[str, List[float]]:
        """Отслеживает тренды в эволюции"""
        if len(self.gene_history) < 2:
            return {}
        
        trends = {}
        gene_names = list(self.gene_history[0]['gene_statistics'].keys())
        
        for gene_name in gene_names:
            means = [history['gene_statistics'][gene_name]['mean'] 
                    for history in self.gene_history]
            
            # Расчет тренда (простая линейная регрессия)
            if len(means) > 1:
                x = list(range(len(means)))
                slope = np.polyfit(x, means, 1)[0]
                trends[gene_name] = slope
            else:
                trends[gene_name] = 0.0
        
        # Тренд разнообразия
        diversity_values = [history['genetic_diversity'] for history in self.gene_history]
        if len(diversity_values) > 1:
            x = list(range(len(diversity_values)))
            diversity_trend = np.polyfit(x, diversity_values, 1)[0]
            trends['diversity'] = diversity_trend
        
        return trends
    
    def get_generation_comparison(self, generation1: int, generation2: int) -> Dict[str, Any]:
        """Сравнивает два поколения"""
        if generation1 >= len(self.gene_history) or generation2 >= len(self.gene_history):
            return {}
        
        hist1 = self.gene_history[generation1]
        hist2 = self.gene_history[generation2]
        
        comparison = {
            'generation1': generation1,
            'generation2': generation2,
            'population_change': hist2['population_size'] - hist1['population_size'],
            'diversity_change': hist2['genetic_diversity'] - hist1['genetic_diversity']
        }
        
        # Сравнение статистик генов
        gene_changes = {}
        for gene_name in hist1['gene_statistics']:
            stats1 = hist1['gene_statistics'][gene_name]
            stats2 = hist2['gene_statistics'][gene_name]
            
            gene_changes[gene_name] = {
                'mean_change': stats2['mean'] - stats1['mean'],
                'std_change': stats2['std'] - stats1['std'],
                'range_change': stats2['range'] - stats1['range']
            }
        
        comparison['gene_changes'] = gene_changes
        return comparison


class EvolutionManager:
    """Менеджер эволюционных процессов"""
    
    def __init__(self, population_size: int = 50, mutation_rate: float = 0.1,
                 selection_pressure: float = 0.8):
        self.genetic_algorithm = GeneticAlgorithm(
            population_size, mutation_rate, selection_pressure=selection_pressure
        )
        self.gene_analyzer = GeneAnalyzer()
        
        # История эволюции
        self.evolution_history = []
        self.current_generation = 0
        
        # Параметры эволюции
        self.reproduction_interval = 100  # Интервал между поколениями
        self.last_reproduction_time = 0
    
    def should_evolve(self, current_time: int) -> bool:
        """Проверяет, пора ли эволюционировать"""
        return current_time - self.last_reproduction_time >= self.reproduction_interval
    
    def evolve(self, agents: List[Agent], environment, current_time: int) -> List[Agent]:
        """Выполняет эволюцию популяции"""
        if not self.should_evolve(current_time):
            return agents
        
        # Эволюция популяции
        new_agents = self.genetic_algorithm.evolve_population(agents, environment, current_time)
        
        # Анализ генов
        gene_analysis = self.gene_analyzer.analyze_population_genes(new_agents)
        gene_analysis['timestamp'] = current_time
        
        # Сохраняем историю
        evolution_record = {
            'timestamp': current_time,
            'generation': self.genetic_algorithm.generation,
            'population_size': len(new_agents),
            'gene_analysis': gene_analysis,
            'evolution_stats': self.genetic_algorithm.get_evolution_stats()
        }
        
        self.evolution_history.append(evolution_record)
        self.current_generation = self.genetic_algorithm.generation
        self.last_reproduction_time = current_time
        
        return new_agents
    
    def get_evolution_summary(self) -> Dict[str, Any]:
        """Возвращает сводку эволюции"""
        if not self.evolution_history:
            return {}
        
        latest_record = self.evolution_history[-1]
        
        # Тренды
        trends = self.gene_analyzer.track_evolution_trends()
        
        # Статистика
        evolution_stats = self.genetic_algorithm.get_evolution_stats()
        
        return {
            'current_generation': self.current_generation,
            'total_generations': len(self.evolution_history),
            'latest_population_size': latest_record['population_size'],
            'evolution_stats': evolution_stats,
            'gene_trends': trends,
            'latest_gene_analysis': latest_record['gene_analysis']
        }
    
    def get_detailed_history(self) -> List[Dict[str, Any]]:
        """Возвращает подробную историю эволюции"""
        return self.evolution_history.copy()
    
    def reset(self):
        """Сбрасывает состояние эволюции"""
        self.genetic_algorithm = GeneticAlgorithm(
            self.genetic_algorithm.population_size,
            self.genetic_algorithm.mutation_rate,
            selection_pressure=self.genetic_algorithm.selection_pressure
        )
        self.gene_analyzer = GeneAnalyzer()
        self.evolution_history = []
        self.current_generation = 0
        self.last_reproduction_time = 0
    
    def set_parameters(self, population_size: int = None, mutation_rate: float = None,
                       selection_pressure: float = None, reproduction_interval: int = None):
        """Устанавливает параметры эволюции"""
        if population_size is not None:
            self.genetic_algorithm.population_size = population_size
        
        if mutation_rate is not None:
            self.genetic_algorithm.mutation_rate = mutation_rate
        
        if selection_pressure is not None:
            self.genetic_algorithm.selection_pressure = selection_pressure
        
        if reproduction_interval is not None:
            self.reproduction_interval = reproduction_interval
