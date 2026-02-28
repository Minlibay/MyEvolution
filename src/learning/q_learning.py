"""
Q-learning module - implements reinforcement learning for agents
"""

from typing import Dict, List, Tuple, Optional, Any
import random
import numpy as np

from ..core.agent import Agent, Episode
from ..core.agent_actions import ActionResult


class QLearningAgent:
    """Q-learning алгоритм для агента"""
    
    def __init__(self, learning_rate: float = 0.1, discount_factor: float = 0.95, 
                 epsilon: float = 0.1, epsilon_decay: float = 0.995):
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = 0.01
        
        # Q-таблица: (state, action) -> q_value
        self.q_table: Dict[Tuple[str, str], float] = {}
        
        # Статистика обучения
        self.learning_episodes = 0
        self.total_reward = 0.0
    
    def get_q_value(self, state: str, action: str) -> float:
        """Возвращает Q-значение для состояния-действия"""
        key = (state, action)
        return self.q_table.get(key, 0.0)
    
    def update_q_value(self, state: str, action: str, reward: float, 
                      next_state: str, available_actions: List[str]) -> None:
        """Обновляет Q-значение по формуле Q-learning"""
        key = (state, action)
        current_q = self.q_table.get(key, 0.0)
        
        # Максимальное Q-значение для следующего состояния
        max_next_q = 0.0
        if available_actions:
            max_next_q = max(
                self.get_q_value(next_state, next_action) 
                for next_action in available_actions
            )
        
        # Q-learning обновление
        new_q = current_q + self.learning_rate * (
            reward + self.discount_factor * max_next_q - current_q
        )
        
        self.q_table[key] = new_q
        self.learning_episodes += 1
        self.total_reward += reward
    
    def select_action(self, state: str, available_actions: List[str]) -> str:
        """Выбирает действие используя ε-greedy стратегию"""
        if not available_actions:
            return "rest"  # Действие по умолчанию
        
        # ε-greedy выбор
        if random.random() < self.epsilon:
            return random.choice(available_actions)
        
        # Выбор действия с максимальным Q-значением
        best_action = None
        best_q_value = float('-inf')
        
        for action in available_actions:
            q_value = self.get_q_value(state, action)
            if q_value > best_q_value:
                best_q_value = q_value
                best_action = action
        
        return best_action if best_action else random.choice(available_actions)
    
    def decay_epsilon(self):
        """Уменьшает epsilon со временем"""
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)
    
    def get_learning_stats(self) -> Dict[str, float]:
        """Возвращает статистику обучения"""
        avg_reward = self.total_reward / max(1, self.learning_episodes)
        
        return {
            'learning_episodes': self.learning_episodes,
            'total_reward': self.total_reward,
            'average_reward': avg_reward,
            'current_epsilon': self.epsilon,
            'q_table_size': len(self.q_table)
        }
    
    def reset(self):
        """Сбрасывает состояние обучения"""
        self.q_table.clear()
        self.learning_episodes = 0
        self.total_reward = 0.0
        self.epsilon = max(0.1, self.epsilon)  # Восстанавливаем epsilon


class StateEncoder:
    """Кодирует состояние агента в строковое представление"""
    
    @staticmethod
    def encode_state(agent: Agent, local_env: Dict) -> str:
        """Кодирует состояние в строку для Q-таблицы"""
        # Дискретизация непрерывных параметров
        hunger_level = StateEncoder._discretize(agent.hunger, bins=5)
        energy_level = StateEncoder._discretize(agent.energy, bins=5)
        health_level = StateEncoder._discretize(agent.health, bins=3)
        
        # Категоризация локальных объектов
        object_counts = {}
        for obj in local_env.get('perceived_objects', []):
            object_counts[obj.type] = object_counts.get(obj.type, 0) + 1
        
        # Информация об инвентаре
        inventory_size = min(len(agent.inventory), 5)
        tool_count = min(len(agent.tools), 3)
        
        # Создание строкового представления состояния
        state_parts = [
            f"hunger_{hunger_level}",
            f"energy_{energy_level}",
            f"health_{health_level}",
            f"pos_{agent.position[0]}_{agent.position[1]}",
            f"inv_{inventory_size}",
            f"tools_{tool_count}"
        ]
        
        # Добавляем информацию об объектах (ограничиваем количество)
        for obj_type, count in sorted(object_counts.items())[:5]:
            state_parts.append(f"{obj_type}_{min(count, 3)}")
        
        # Добавляем информацию о сезоне
        season = local_env.get('season', 0)
        state_parts.append(f"season_{season}")
        
        return "|".join(state_parts)
    
    @staticmethod
    def _discretize(value: float, bins: int) -> int:
        """Дискретизирует значение в указанное количество бинов"""
        bin_size = 1.0 / bins
        return min(bins - 1, int(value / bin_size))
    
    @staticmethod
    def get_state_features(agent: Agent, local_env: Dict) -> np.ndarray:
        """Возвращает вектор признаков состояния для нейронных сетей"""
        features = []
        
        # Физиологические признаки
        features.extend([
            agent.hunger,
            agent.energy,
            agent.health,
            agent.exploration_rate
        ])
        
        # Позиционные признаки (нормализованные)
        features.extend([
            agent.position[0] / 100.0,  # Предполагаем мир 100x100
            agent.position[1] / 100.0
        ])
        
        # Признаки инвентаря
        features.extend([
            len(agent.inventory) / 10.0,
            len(agent.tools) / 5.0
        ])
        
        # Признаки локальных объектов
        object_types = ['stone', 'wood', 'plant', 'berry', 'bone', 'fiber', 'animal']
        for obj_type in object_types:
            count = sum(1 for obj in local_env.get('perceived_objects', []) 
                       if obj.type == obj_type)
            features.append(min(count, 5) / 5.0)
        
        # Сезонный признак
        season = local_env.get('season', 0) / 3.0
        features.append(season)
        
        return np.array(features)


class DecisionMaker:
    """Система принятия решений агента"""
    
    def __init__(self, agent: Agent, learning_algorithm: QLearningAgent):
        self.agent = agent
        self.agent_id = agent.id  # Добавляем agent_id
        self.learner = learning_algorithm
        self.state_encoder = StateEncoder()
        
        # Параметры принятия решений
        self.memory_influence = 0.2  # Влияние памяти
        self.exploration_bonus = 0.1  # Бонус за исследование
    
    def select_action(self, local_env: Dict, available_actions: List[str]) -> str:
        """Выбирает действие на основе Q-значений и памяти"""
        if not available_actions:
            return "rest"
        
        # Кодируем текущее состояние
        current_state = self.state_encoder.encode_state(self.agent, local_env)
        
        # Получаем Q-значения для доступных действий
        action_values = {}
        for action in available_actions:
            q_value = self.learner.get_q_value(current_state, action)
            
            # Добавляем влияние памяти
            memory_bonus = self._calculate_memory_bonus(action, local_env)
            
            # Добавляем бонус за исследование
            exploration_bonus = 0.0
            if action == 'move' and self.agent.exploration_rate > 0.5:
                exploration_bonus = self.exploration_bonus
            
            total_value = q_value + memory_bonus + exploration_bonus
            action_values[action] = total_value
        
        # Выбираем действие с максимальным значением
        best_action = max(action_values, key=action_values.get)
        
        # ε-greedy с учетом индивидуальной exploration_rate агента
        if random.random() < self.agent.exploration_rate:
            return random.choice(available_actions)
        
        return best_action
    
    def _calculate_memory_bonus(self, action: str, local_env: Dict) -> float:
        """Рассчитывает бонус от памяти для действия"""
        bonus = 0.0
        
        # Эпизодическая память
        similar_episodes = self.agent.episodic_memory.recall_similar(
            self.state_encoder.encode_state(self.agent, local_env), 
            k=3
        )
        
        if similar_episodes:
            # Если похожие эпизоды были успешными
            successful_episodes = [ep for ep in similar_episodes if ep.reward > 0]
            if successful_episodes:
                # Учитываем только если действие совпадает
                matching_episodes = [ep for ep in successful_episodes if ep.action == action]
                if matching_episodes:
                    bonus += self.memory_influence * 0.5
        
        # Статистическая память
        if action == 'gather':
            # Проверяем статистику по объектам
            for obj in local_env.get('perceived_objects', []):
                familiarity = self.agent.statistical_memory.get_statistic(f"object_{obj.type}")
                if obj.is_edible() and familiarity > 0.5:
                    bonus += self.memory_influence * familiarity * 0.3
        
        elif action == 'consume':
            # Проверяем статистику по потреблению
            consume_success = self.agent.statistical_memory.get_statistic("consume_success")
            if consume_success > 0.3:
                bonus += self.memory_influence * consume_success * 0.4
        
        return bonus
    
    def learn_from_experience(self, previous_state: str, action: str, 
                           result: ActionResult, current_state: str,
                           available_actions: List[str]):
        """Обучается на основе опыта"""
        # Обновляем Q-значения
        self.learner.update_q_value(
            previous_state, action, result.reward, 
            current_state, available_actions
        )
        
        # Обновляем статистическую память
        self._update_statistical_memory(action, result)
        
        # Добавляем эпизод в память
        episode = Episode(
            timestamp=self.agent.age,
            state=previous_state,
            action=action,
            reward=result.reward,
            next_state=current_state,
            context={
                'success': result.success,
                'energy_cost': result.energy_cost,
                'data': result.data
            }
        )
        self.agent.episodic_memory.add_episode(episode)
        
        # Уменьшаем epsilon
        self.learner.decay_epsilon()
        self.agent.exploration_rate = self.learner.epsilon
    
    def _update_statistical_memory(self, action: str, result: ActionResult):
        """Обновляет статистическую память на основе результата"""
        if result.success:
            self.agent.statistical_memory.update_statistic(f"{action}_success", 1.0)
        else:
            self.agent.statistical_memory.update_statistic(f"{action}_success", 0.0)
        
        # Обновляем специфическую статистику
        if action == 'consume' and result.success:
            energy_gain = result.data.get('energy_gain', 0.0)
            self.agent.statistical_memory.update_statistic("consume_energy_gain", energy_gain)
        
        elif action == 'gather' and result.success:
            gathered_count = len(result.data.get('gathered_objects', []))
            self.agent.statistical_memory.update_statistic("gather_count", gathered_count)
        
        elif action == 'combine' and result.success:
            discovery_type = result.data.get('discovery_type', 'known')
            if discovery_type == 'new_discovery':
                self.agent.statistical_memory.update_statistic("new_discoveries", 1.0)
    
    def get_decision_stats(self) -> Dict[str, float]:
        """Возвращает статистику принятия решений"""
        learner_stats = self.learner.get_learning_stats()
        
        # Добавляем информацию о памяти
        memory_stats = {
            'episodic_memory_size': len(self.agent.episodic_memory.episodes),
            'statistical_memory_size': len(self.agent.statistical_memory.statistics),
            'exploration_rate': self.agent.exploration_rate
        }
        
        return {**learner_stats, **memory_stats}


class LearningManager:
    """Менеджер обучения для всех агентов"""
    
    def __init__(self):
        self.learners: Dict[str, DecisionMaker] = {}
        self.global_stats = {
            'total_learning_episodes': 0,
            'total_rewards': 0.0,
            'active_learners': 0
        }

        self._bootstrap_q_table: Dict[Tuple[str, str], float] = {}
        self._bootstrap_epsilon: Optional[float] = None

    def set_bootstrap(self, q_table: Optional[Dict[Tuple[str, str], float]] = None, epsilon: Optional[float] = None):
        if q_table is None:
            self._bootstrap_q_table = {}
        else:
            self._bootstrap_q_table = dict(q_table)
        self._bootstrap_epsilon = None if epsilon is None else float(epsilon)
    
    def register_agent(self, agent: Agent):
        """Регистрирует агента в системе обучения"""
        epsilon = agent.exploration_rate
        if self._bootstrap_epsilon is not None:
            epsilon = float(self._bootstrap_epsilon)

        learner = QLearningAgent(
            learning_rate=agent.learning_rate,
            discount_factor=agent.discount_factor,
            epsilon=epsilon
        )

        if self._bootstrap_q_table:
            learner.q_table = dict(self._bootstrap_q_table)
            agent.exploration_rate = float(learner.epsilon)
        
        decision_maker = DecisionMaker(agent, learner)
        self.learners[agent.id] = decision_maker
        self.global_stats['active_learners'] += 1

    def export_global_learning_state(self) -> Dict[str, Any]:
        merged: Dict[Tuple[str, str], list[float]] = {}
        eps: list[float] = []
        episodes = 0

        for dm in self.learners.values():
            try:
                q = getattr(dm.learner, 'q_table', {}) or {}
                for (state, action), val in q.items():
                    key = (str(state), str(action))
                    if key not in merged:
                        merged[key] = []
                    merged[key].append(float(val))
                eps.append(float(getattr(dm.learner, 'epsilon', 0.1) or 0.1))
                episodes += int(getattr(dm.learner, 'learning_episodes', 0) or 0)
            except Exception:
                continue

        out_q: Dict[Tuple[str, str], float] = {}
        for k, vals in merged.items():
            if not vals:
                continue
            out_q[k] = float(sum(vals) / len(vals))

        out_eps = float(sum(eps) / len(eps)) if eps else None

        items = [[s, a, float(v)] for (s, a), v in out_q.items()]
        return {
            'version': 1,
            'epsilon': out_eps,
            'episodes': int(episodes),
            'q_table': items,
        }
    
    def unregister_agent(self, agent_id: str):
        """Удаляет агента из системы обучения"""
        if agent_id in self.learners:
            del self.learners[agent_id]
            self.global_stats['active_learners'] -= 1
    
    def get_decision_maker(self, agent_id: str) -> Optional[DecisionMaker]:
        """Возвращает систему принятия решений для агента"""
        return self.learners.get(agent_id)
    
    def update_global_stats(self):
        """Обновляет глобальную статистику"""
        total_episodes = 0
        total_rewards = 0.0
        
        for decision_maker in self.learners.values():
            stats = decision_maker.get_decision_stats()
            total_episodes += stats['learning_episodes']
            total_rewards += stats['total_reward']
        
        self.global_stats.update({
            'total_learning_episodes': total_episodes,
            'total_rewards': total_rewards,
            'average_reward_per_episode': total_rewards / max(1, total_episodes)
        })
    
    def get_global_stats(self) -> Dict[str, float]:
        """Возвращает глобальную статистику обучения"""
        self.update_global_stats()
        return self.global_stats.copy()
    
    def get_top_performers(self, n: int = 5) -> List[Tuple[str, float]]:
        """Возвращает топ-n агентов по общей награде"""
        performer_rewards = []
        
        for agent_id, decision_maker in self.learners.items():
            stats = decision_maker.get_decision_stats()
            performer_rewards.append((agent_id, stats['total_reward']))
        
        performer_rewards.sort(key=lambda x: x[1], reverse=True)
        return performer_rewards[:n]
