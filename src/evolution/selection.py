"""
Selection module - selection mechanisms for evolution
"""

from typing import List, Tuple, Dict, Any
import random
import numpy as np

from ..core.agent import Agent


class SelectionMechanism:
    """Базовый класс механизмов селекции"""
    
    def __init__(self, name: str):
        self.name = name
        self.selection_count = 0
    
    def select(self, agents: List[Agent], fitness_scores: List[float], 
              count: int) -> List[Agent]:
        """Выбирает агентов для селекции"""
        raise NotImplementedError("Subclasses must implement select method")
    
    def get_stats(self) -> Dict[str, Any]:
        """Возвращает статистику селекции"""
        return {
            'name': self.name,
            'selection_count': self.selection_count
        }


class RouletteWheelSelection(SelectionMechanism):
    """Рулеточная селекция"""
    
    def __init__(self):
        super().__init__("RouletteWheel")
    
    def select(self, agents: List[Agent], fitness_scores: List[float], 
              count: int) -> List[Agent]:
        """Выбирает агентов методом рулетки"""
        if not agents or count <= 0:
            return []
        
        # Нормализация оценок приспособленности
        total_fitness = sum(fitness_scores)
        if total_fitness == 0:
            # Если все оценки нулевые, используем равномерное распределение
            probabilities = [1.0 / len(agents)] * len(agents)
        else:
            probabilities = [score / total_fitness for score in fitness_scores]
        
        # Выборка
        selected = []
        for _ in range(min(count, len(agents))):
            selected_idx = np.random.choice(len(agents), p=probabilities)
            selected.append(agents[selected_idx])
            self.selection_count += 1
        
        return selected


class TournamentSelection(SelectionMechanism):
    """Турнирная селекция"""
    
    def __init__(self, tournament_size: int = 3):
        super().__init__(f"Tournament(tournament_size={tournament_size})")
        self.tournament_size = tournament_size
    
    def select(self, agents: List[Agent], fitness_scores: List[float], 
              count: int) -> List[Agent]:
        """Выбирает агентов методом турнира"""
        if not agents or count <= 0:
            return []
        
        selected = []
        tournament_size = min(self.tournament_size, len(agents))
        
        for _ in range(min(count, len(agents))):
            # Выбор участников турнира
            tournament_indices = random.sample(range(len(agents)), tournament_size)
            
            # Находим победителя
            best_idx = max(tournament_indices, key=lambda i: fitness_scores[i])
            selected.append(agents[best_idx])
            self.selection_count += 1
        
        return selected


class RankSelection(SelectionMechanism):
    """Селекция на основе рангов"""
    
    def __init__(self):
        super().__init__("RankSelection")
    
    def select(self, agents: List[Agent], fitness_scores: List[float], 
              count: int) -> List[Agent]:
        """Выбирает агентов на основе рангов приспособленности"""
        if not agents or count <= 0:
            return []
        
        # Сортировка по приспособленности
        sorted_indices = sorted(range(len(agents)), key=lambda i: fitness_scores[i], reverse=True)
        
        # Присвоение рангов
        ranks = [0] * len(agents)
        for rank, idx in enumerate(sorted_indices):
            ranks[idx] = rank + 1
        
        # Вероятности на основе рангов (лучшие получают большие вероятности)
        max_rank = len(agents)
        probabilities = [(max_rank - rank + 1) / (max_rank * (max_rank + 1) / 2) 
                       for rank in ranks]
        
        # Нормализация
        total_prob = sum(probabilities)
        probabilities = [p / total_prob for p in probabilities]
        
        # Выборка
        selected = []
        for _ in range(min(count, len(agents))):
            selected_idx = np.random.choice(len(agents), p=probabilities)
            selected.append(agents[selected_idx])
            self.selection_count += 1
        
        return selected


class StochasticUniversalSelection(SelectionMechanism):
    """Стохастическая универсальная селекция"""
    
    def __init__(self):
        super().__init__("StochasticUniversal")
    
    def select(self, agents: List[Agent], fitness_scores: List[float], 
              count: int) -> List[Agent]:
        """Выбирает агентов методом SUS"""
        if not agents or count <= 0:
            return []
        
        # Нормализация оценок
        total_fitness = sum(fitness_scores)
        if total_fitness == 0:
            probabilities = [1.0 / len(agents)] * len(agents)
        else:
            probabilities = [score / total_fitness for score in fitness_scores]
        
        # Кумулятивные вероятности
        cumulative_probs = []
        cumulative = 0.0
        for prob in probabilities:
            cumulative += prob
            cumulative_probs.append(cumulative)
        
        # Равномерно расположенные указатели
        pointers = [(i + random.random()) / count for i in range(count)]
        
        # Выборка
        selected = []
        for pointer in pointers:
            for i, cum_prob in enumerate(cumulative_probs):
                if pointer <= cum_prob:
                    selected.append(agents[i])
                    self.selection_count += 1
                    break
        
        return selected[:count]


class ElitistSelection(SelectionMechanism):
    """Элитарная селекция - выбор лучших агентов"""
    
    def __init__(self, elite_ratio: float = 0.1):
        super().__init__(f"Elitist(elite_ratio={elite_ratio})")
        self.elite_ratio = elite_ratio
    
    def select(self, agents: List[Agent], fitness_scores: List[float], 
              count: int) -> List[Agent]:
        """Выбирает лучших агентов"""
        if not agents or count <= 0:
            return []
        
        # Количество элитных агентов
        elite_count = max(1, int(len(agents) * self.elite_ratio))
        elite_count = min(elite_count, count, len(agents))
        
        # Сортировка по приспособленности
        sorted_agents = sorted(zip(agents, fitness_scores), key=lambda x: x[1], reverse=True)
        
        selected = [agent for agent, _ in sorted_agents[:elite_count]]
        self.selection_count += elite_count
        
        return selected


class AdaptiveSelection(SelectionMechanism):
    """Адаптивная селекция - меняет стратегию в зависимости от состояния популяции"""
    
    def __init__(self):
        super().__init__("Adaptive")
        self.methods = [
            RouletteWheelSelection(),
            TournamentSelection(tournament_size=3),
            RankSelection()
        ]
        self.current_method = 0
        self.performance_history = []
    
    def select(self, agents: List[Agent], fitness_scores: List[float], 
              count: int) -> List[Agent]:
        """Адаптивно выбирает метод селекции"""
        if not agents or count <= 0:
            return []
        
        # Анализ состояния популяции
        diversity = self._calculate_diversity(fitness_scores)
        
        # Выбор метода на основе состояния
        if diversity < 0.1:  # Низкое разнообразие - используем турнирную селекцию
            method = self.methods[1]  # Tournament
        elif diversity > 0.5:  # Высокое разнообразие - используем ранговую селекцию
            method = self.methods[2]  # Rank
        else:  # Среднее разнообразие - используем рулеточную селекцию
            method = self.methods[0]  # RouletteWheel
        
        selected = method.select(agents, fitness_scores, count)
        self.selection_count += count
        
        return selected
    
    def _calculate_diversity(self, fitness_scores: List[float]) -> float:
        """Рассчитывает разнообразие популяции"""
        if len(fitness_scores) < 2:
            return 0.0
        
        # Коэффициент вариации
        mean_fitness = np.mean(fitness_scores)
        std_fitness = np.std(fitness_scores)
        
        if mean_fitness == 0:
            return 0.0
        
        return std_fitness / mean_fitness


class SelectionManager:
    """Менеджер селекции с различными стратегиями"""
    
    def __init__(self):
        self.methods = {
            'roulette': RouletteWheelSelection(),
            'tournament': TournamentSelection(),
            'rank': RankSelection(),
            'stochastic_universal': StochasticUniversalSelection(),
            'elitist': ElitistSelection(),
            'adaptive': AdaptiveSelection()
        }
        self.current_method = 'tournament'
        self.selection_history = []
    
    def set_method(self, method_name: str):
        """Устанавливает метод селекции"""
        if method_name in self.methods:
            self.current_method = method_name
        else:
            raise ValueError(f"Unknown selection method: {method_name}")
    
    def select_parents(self, agents: List[Agent], fitness_scores: List[float],
                      count: int, method_name: str = None) -> List[Agent]:
        """Выбирает родителей с указанным методом"""
        method = method_name or self.current_method
        
        if method not in self.methods:
            method = self.current_method
        
        selected = self.methods[method].select(agents, fitness_scores, count)
        
        # Записываем в историю
        self.selection_history.append({
            'timestamp': 0,  # Будет установлено извне
            'method': method,
            'population_size': len(agents),
            'selected_count': len(selected),
            'avg_fitness': np.mean(fitness_scores) if fitness_scores else 0.0,
            'fitness_std': np.std(fitness_scores) if fitness_scores else 0.0
        })
        
        return selected
    
    def get_method_stats(self, method_name: str = None) -> Dict[str, Any]:
        """Возвращает статистику метода селекции"""
        if method_name is None:
            method_name = self.current_method
        
        if method_name not in self.methods:
            return {}
        
        return self.methods[method_name].get_stats()
    
    def get_all_stats(self) -> Dict[str, Dict[str, Any]]:
        """Возвращает статистику всех методов"""
        return {name: method.get_stats() for name, method in self.methods.items()}
    
    def get_selection_history(self, method_name: str = None, 
                           limit: int = 100) -> List[Dict[str, Any]]:
        """Возвращает историю селекции"""
        history = self.selection_history
        
        if method_name:
            history = [record for record in history if record['method'] == method_name]
        
        return history[-limit:] if limit > 0 else history
    
    def analyze_selection_pressure(self) -> Dict[str, float]:
        """Анализирует давление селекции"""
        if not self.selection_history:
            return {}
        
        recent_history = self.selection_history[-20:]  # Последние 20 селекций
        
        # Давление селекции как отношение выбранного к размеру популяции
        pressures = []
        for record in recent_history:
            if record['population_size'] > 0:
                pressure = record['selected_count'] / record['population_size']
                pressures.append(pressure)
        
        if not pressures:
            return {}
        
        return {
            'average_pressure': np.mean(pressures),
            'pressure_std': np.std(pressures),
            'pressure_trend': np.polyfit(range(len(pressures)), pressures, 1)[0] if len(pressures) > 1 else 0.0
        }
    
    def reset_stats(self):
        """Сбрасывает статистику всех методов"""
        for method in self.methods.values():
            method.selection_count = 0
        self.selection_history = []
