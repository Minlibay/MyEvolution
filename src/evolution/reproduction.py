"""
Reproduction module - reproduction mechanisms and cultural transmission
"""

from typing import List, Dict, Any, Optional, Tuple
import random
import numpy as np

from ..core.agent import Agent, AgentGenes, AgentFactory
from ..core.objects import Object


class ReproductionManager:
    """Менеджер репродукции"""
    
    def __init__(self, cultural_transfer_rate: float = 0.3,
                 mutation_rate: float = 0.1, crossover_rate: float = 0.8):
        self.cultural_transfer_rate = cultural_transfer_rate
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        
        # Статистика репродукции
        self.reproduction_attempts = 0
        self.successful_reproductions = 0
        self.cultural_transfers = 0
        self.mutations_applied = 0
    
    def can_reproduce(self, agent: Agent) -> bool:
        """Проверяет, может ли агент размножаться"""
        return agent.can_reproduce()
    
    def find_mates(self, agent: Agent, potential_mates: List[Agent],
                  max_distance: int = 5) -> List[Agent]:
        """Находит партнеров для размножения"""
        mates = []
        
        for mate in potential_mates:
            if mate.id == agent.id:
                continue
            
            # Проверяем расстояние
            distance = abs(agent.position[0] - mate.position[0]) + \
                      abs(agent.position[1] - mate.position[1])
            
            if distance <= max_distance and self.can_reproduce(mate):
                mates.append(mate)
        
        return mates
    
    def select_mate(self, agent: Agent, mates: List[Agent]) -> Optional[Agent]:
        """Выбирает партнера для размножения"""
        if not mates:
            return None
        
        # Предпочтение based on схожести генов и социальную тенденцию
        mate_scores = []
        
        for mate in mates:
            score = 0.0
            
            # Социальная совместимость (схожесть exploration_bias)
            social_compatibility = 1.0 - abs(agent.genes.social_tendency - mate.genes.social_tendency)
            score += social_compatibility * agent.genes.social_tendency
            
            # Генетическая совместимость (средняя схожесть всех генов)
            genetic_similarity = self._calculate_genetic_similarity(agent.genes, mate.genes)
            score += genetic_similarity * 0.3
            
            # Здоровье партнера
            score += mate.health * 0.2
            
            mate_scores.append((mate, score))
        
        # Выбор партнера (weighted random)
        if mate_scores:
            total_score = sum(score for _, score in mate_scores)
            if total_score > 0:
                probabilities = [score / total_score for _, score in mate_scores]
                selected_idx = np.random.choice(len(mate_scores), p=probabilities)
                return mate_scores[selected_idx][0]
        
        # Если не удалось выбрать взвешенно, выбираем случайно
        return random.choice(mates)
    
    def _calculate_genetic_similarity(self, genes1: AgentGenes, genes2: AgentGenes) -> float:
        """Рассчитывает генетическую схожесть"""
        similarity = 0.0
        gene_names = ['metabolism_speed', 'strength', 'intelligence', 
                     'social_tendency', 'exploration_bias']
        
        for gene_name in gene_names:
            value1 = getattr(genes1, gene_name)
            value2 = getattr(genes2, gene_name)
            similarity += 1.0 - abs(value1 - value2)
        
        return similarity / len(gene_names)
    
    def reproduce(self, parent1: Agent, parent2: Agent, 
                  birth_time: int, environment) -> Optional[Agent]:
        """Создает потомство от двух родителей"""
        self.reproduction_attempts += 1
        
        # Проверяем вероятность кроссовера
        if random.random() > self.crossover_rate:
            return None
        
        # Создание потомка
        child_id = f"offspring_{birth_time}_{self.successful_reproductions}"
        child = AgentFactory.create_offspring(parent1, parent2, child_id, birth_time)
        
        # Культурная передача
        cultural_transfer = self._perform_cultural_transfer(parent1, parent2, child)
        
        # Мутации
        mutations_applied = self._apply_mutations(child)
        
        # Обновляем статистику
        self.successful_reproductions += 1
        if cultural_transfer:
            self.cultural_transfers += 1
        if mutations_applied:
            self.mutations_applied += 1
        
        # Обновляем счетчики потомства у родителей
        parent1.offspring_count += 1
        parent2.offspring_count += 1
        
        # Энергетическая стоимость для родителей
        energy_cost = 0.2
        parent1.energy = max(0.0, parent1.energy - energy_cost)
        parent2.energy = max(0.0, parent2.energy - energy_cost)
        
        return child
    
    def _perform_cultural_transfer(self, parent1: Agent, parent2: Agent, 
                                 child: Agent) -> bool:
        """Выполняет культурную передачу знаний"""
        transfer_occurred = False
        
        # Передача статистической памяти
        for parent in [parent1, parent2]:
            for key, value in parent.statistical_memory.statistics.items():
                if random.random() < self.cultural_transfer_rate:
                    # Передача с "забыванием"
                    transferred_value = value * random.uniform(0.5, 0.9)
                    child.statistical_memory.update_statistic(key, transferred_value)
                    transfer_occurred = True
        
        # Передача ключевых эпизодов памяти
        for parent in [parent1, parent2]:
            if len(parent.episodic_memory.episodes) > 0:
                # Выбираем наиболее важные эпизоды
                important_episodes = []
                for episode in parent.episodic_memory.episodes:
                    if isinstance(episode, tuple):  # (episode, priority)
                        ep, priority = episode
                        if priority > 1.5:  # Порог важности
                            important_episodes.append(ep)
                
                # Передаем часть важных эпизодов
                transfer_count = min(3, len(important_episodes))
                if transfer_count > 0:
                    selected_episodes = random.sample(important_episodes, transfer_count)
                    for episode in selected_episodes:
                        # Создаем копию эпизода для потомка
                        child_episode = Episode(
                            timestamp=child.birth_time,
                            state=episode.state,
                            action=episode.action,
                            reward=episode.reward * 0.7,  # Небольшое "забывание"
                            next_state=episode.next_state,
                            context=episode.context.copy()
                        )
                        child.episodic_memory.add_episode(child_episode)
                        transfer_occurred = True
        
        # Передача базовых навыков (Q-значения для важных состояний)
        for parent in [parent1, parent2]:
            important_q_values = []
            for (state, action), q_value in parent.q_table.items():
                if q_value > 0.5:  # Важные действия
                    important_q_values.append(((state, action), q_value))
            
            # Передаем часть важных Q-значений
            transfer_count = min(5, len(important_q_values))
            if transfer_count > 0:
                selected_q_values = random.sample(important_q_values, transfer_count)
                for (state, action), q_value in selected_q_values:
                    transferred_q = q_value * random.uniform(0.6, 0.8)
                    child.q_table[(state, action)] = transferred_q
                    transfer_occurred = True
        
        return transfer_occurred
    
    def _apply_mutations(self, child: Agent) -> bool:
        """Применяет мутации к потомку"""
        mutations_occurred = False
        
        # Мутации генов
        original_genes = child.genes.to_dict()
        child.genes.mutate(self.mutation_rate)
        
        new_genes = child.genes.to_dict()
        for gene_name in original_genes:
            if original_genes[gene_name] != new_genes[gene_name]:
                mutations_occurred = True
                break
        
        # Мутации параметров обучения (редко)
        if random.random() < self.mutation_rate * 0.5:
            child.learning_rate *= random.uniform(0.8, 1.2)
            child.learning_rate = max(0.01, min(0.5, child.learning_rate))
            mutations_occurred = True
        
        if random.random() < self.mutation_rate * 0.5:
            child.discount_factor *= random.uniform(0.9, 1.1)
            child.discount_factor = max(0.5, min(0.99, child.discount_factor))
            mutations_occurred = True
        
        return mutations_occurred
    
    def get_reproduction_stats(self) -> Dict[str, Any]:
        """Возвращает статистику репродукции"""
        success_rate = (self.successful_reproductions / max(1, self.reproduction_attempts))
        cultural_rate = (self.cultural_transfers / max(1, self.successful_reproductions))
        mutation_rate = (self.mutations_applied / max(1, self.successful_reproductions))
        
        return {
            'reproduction_attempts': self.reproduction_attempts,
            'successful_reproductions': self.successful_reproductions,
            'success_rate': success_rate,
            'cultural_transfers': self.cultural_transfers,
            'cultural_transfer_rate': cultural_rate,
            'mutations_applied': self.mutations_applied,
            'mutation_rate': mutation_rate,
            'cultural_transfer_parameter': self.cultural_transfer_rate,
            'mutation_parameter': self.mutation_rate,
            'crossover_parameter': self.crossover_rate
        }
    
    def reset_stats(self):
        """Сбрасывает статистику"""
        self.reproduction_attempts = 0
        self.successful_reproductions = 0
        self.cultural_transfers = 0
        self.mutations_applied = 0


class CulturalEvolution:
    """Отслеживание культурной эволюции"""
    
    def __init__(self):
        self.cultural_traits = {}  # {trait_id: trait_data}
        self.trait_history = []   # История культурных черт
        self.transmission_events = []  # События передачи знаний
        
    def register_cultural_trait(self, trait_name: str, trait_data: Dict[str, Any],
                               agent_id: str, timestamp: int) -> str:
        """Регистрирует культурную черту"""
        trait_id = f"trait_{len(self.cultural_traits)}"
        
        self.cultural_traits[trait_id] = {
            'name': trait_name,
            'data': trait_data,
            'originator': agent_id,
            'creation_time': timestamp,
            'transmission_count': 0,
            'current_carriers': {agent_id}
        }
        
        return trait_id
    
    def transmit_trait(self, trait_id: str, from_agent: str, to_agent: str,
                      timestamp: int, success: bool = True):
        """Регистрирует передачу культурной черты"""
        if trait_id not in self.cultural_traits:
            return
        
        trait = self.cultural_traits[trait_id]
        
        if success:
            trait['transmission_count'] += 1
            trait['current_carriers'].add(to_agent)
        
        self.transmission_events.append({
            'trait_id': trait_id,
            'trait_name': trait['name'],
            'from_agent': from_agent,
            'to_agent': to_agent,
            'timestamp': timestamp,
            'success': success
        })
    
    def analyze_cultural_diversity(self, agents: List[Agent]) -> Dict[str, Any]:
        """Анализирует культурное разнообразие"""
        if not agents:
            return {}
        
        # Анализ Q-таблиц
        all_actions = set()
        all_states = set()
        q_value_distributions = {}
        
        for agent in agents:
            for (state, action), q_value in agent.q_table.items():
                all_actions.add(action)
                all_states.add(state)
                
                if action not in q_value_distributions:
                    q_value_distributions[action] = []
                q_value_distributions[action].append(q_value)
        
        # Распределение Q-значений по действиям
        action_stats = {}
        for action, values in q_value_distributions.items():
            if values:
                action_stats[action] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'min': np.min(values),
                    'max': np.max(values),
                    'count': len(values)
                }
        
        # Анализ статистической памяти
        all_memory_keys = set()
        memory_value_distributions = {}
        
        for agent in agents:
            for key, value in agent.statistical_memory.statistics.items():
                all_memory_keys.add(key)
                
                if key not in memory_value_distributions:
                    memory_value_distributions[key] = []
                memory_value_distributions[key].append(value)
        
        memory_stats = {}
        for key, values in memory_value_distributions.items():
            if values:
                memory_stats[key] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'prevalence': len(values) / len(agents)
                }
        
        return {
            'population_size': len(agents),
            'unique_actions': len(all_actions),
            'unique_states': len(all_states),
            'action_statistics': action_stats,
            'memory_statistics': memory_stats,
            'total_q_entries': sum(len(agent.q_table) for agent in agents),
            'total_memory_entries': sum(len(agent.statistical_memory.statistics) for agent in agents)
        }
    
    def track_cultural_drift(self, time_window: int = 100) -> Dict[str, float]:
        """Отслеживает культурный дрейф"""
        if len(self.trait_history) < 2:
            return {}
        
        # Анализ изменений в культурных чертах
        recent_traits = self.trait_history[-time_window:] if time_window > 0 else self.trait_history
        
        # Простая метрика дрейфа - изменение в распространенности черт
        drift_metrics = {}
        
        for trait_id in self.cultural_traits:
            trait = self.cultural_traits[trait_id]
            
            # Изменение в количестве носителей
            current_carriers = len(trait['current_carriers'])
            
            # Исторические данные о распространении
            historical_carriers = []
            for record in recent_traits:
                if trait_id in record.get('trait_carriers', {}):
                    historical_carriers.append(record['trait_carriers'][trait_id])
            
            if len(historical_carriers) > 1:
                # Тренд распространенности
                trend = np.polyfit(range(len(historical_carriers)), historical_carriers, 1)[0]
                drift_metrics[trait['name']] = trend
        
        return drift_metrics
    
    def get_cultural_evolution_summary(self) -> Dict[str, Any]:
        """Возвращает сводку культурной эволюции"""
        total_traits = len(self.cultural_traits)
        total_transmissions = len(self.transmission_events)
        
        # Наиболее передаваемые черты
        trait_transmissions = {}
        for trait_id, trait in self.cultural_traits.items():
            trait_transmissions[trait['name']] = trait['transmission_count']
        
        most_transmitted = max(trait_transmissions.items(), key=lambda x: x[1]) if trait_transmissions else None
        
        # Успешность передач
        successful_transmissions = sum(1 for event in self.transmission_events if event['success'])
        transmission_success_rate = successful_transmissions / max(1, total_transmissions)
        
        return {
            'total_traits': total_traits,
            'total_transmissions': total_transmissions,
            'transmission_success_rate': transmission_success_rate,
            'most_transmitted_trait': most_transmitted,
            'trait_transmission_counts': trait_transmissions,
            'current_carriers': {
                trait['name']: len(trait['current_carriers'])
                for trait in self.cultural_traits.values()
            }
        }
