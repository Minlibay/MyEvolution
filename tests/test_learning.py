"""
Tests for learning module
"""

import pytest
import numpy as np
from src.learning.q_learning import QLearningAgent, StateEncoder, DecisionMaker, LearningManager
from src.learning.memory import AdvancedEpisodicMemory, SemanticMemory, ProceduralMemory, IntegratedMemorySystem
from src.core.agent import Agent, AgentGenes, Episode
from src.core.agent_actions import ActionResult


class TestQLearningAgent:
    """Тесты Q-learning агента"""
    
    def setup_method(self):
        """Настройка тестового агента"""
        self.learner = QLearningAgent(learning_rate=0.1, discount_factor=0.95, epsilon=0.1)
    
    def test_initialization(self):
        """Тест инициализации"""
        assert self.learner.learning_rate == 0.1
        assert self.learner.discount_factor == 0.95
        assert self.learner.epsilon == 0.1
        assert len(self.learner.q_table) == 0
    
    def test_get_q_value(self):
        """Тест получения Q-значения"""
        # Несуществующее состояние-действие
        q_value = self.learner.get_q_value("state1", "action1")
        assert q_value == 0.0
        
        # Добавляем значение в Q-таблицу
        self.learner.q_table[("state1", "action1")] = 0.5
        q_value = self.learner.get_q_value("state1", "action1")
        assert q_value == 0.5
    
    def test_update_q_value(self):
        """Тест обновления Q-значения"""
        # Обновляем Q-значение
        self.learner.update_q_value("state1", "action1", 1.0, "state2", ["action1", "action2"])
        
        # Проверяем, что значение обновилось
        q_value = self.learner.get_q_value("state1", "action1")
        assert q_value > 0.0
        
        # Проверяем статистику
        assert self.learner.learning_episodes == 1
        assert self.learner.total_reward == 1.0
    
    def test_select_action(self):
        """Тест выбора действия"""
        available_actions = ["action1", "action2", "action3"]
        
        # С epsilon = 0.1, должно выбирать жадно чаще всего
        action_counts = {"action1": 0, "action2": 0, "action3": 0}
        
        # Устанавливаем epsilon = 0 для детерминированного теста
        self.learner.epsilon = 0.0
        
        # Устанавливаем разные Q-значения
        self.learner.q_table[("state1", "action1")] = 0.1
        self.learner.q_table[("state1", "action2")] = 0.5
        self.learner.q_table[("state1", "action3")] = 0.3
        
        # Выбираем действие много раз
        for _ in range(100):
            action = self.learner.select_action("state1", available_actions)
            action_counts[action] += 1
        
        # Должно всегда выбирать action2 (максимальное Q-значение)
        assert action_counts["action2"] == 100
        assert action_counts["action1"] == 0
        assert action_counts["action3"] == 0
    
    def test_decay_epsilon(self):
        """Тест уменьшения epsilon"""
        initial_epsilon = self.learner.epsilon
        
        self.learner.decay_epsilon()
        
        assert self.learner.epsilon < initial_epsilon
        assert self.learner.epsilon >= self.learner.min_epsilon
    
    def test_learning_stats(self):
        """Тест статистики обучения"""
        # Добавляем несколько эпизодов обучения
        self.learner.update_q_value("state1", "action1", 1.0, "state2", ["action1"])
        self.learner.update_q_value("state2", "action2", -0.5, "state3", ["action2"])
        
        stats = self.learner.get_learning_stats()
        
        assert stats['learning_episodes'] == 2
        assert stats['total_reward'] == 0.5
        assert stats['average_reward'] == 0.25
        assert stats['q_table_size'] == 2


class TestStateEncoder:
    """Тесты кодировщика состояний"""
    
    def setup_method(self):
        """Настройка тестового агента"""
        self.genes = AgentGenes()
        self.agent = Agent("test_agent", (5, 5), self.genes)
    
    def test_encode_state(self):
        """Тест кодирования состояния"""
        local_env = {
            'perceived_objects': [],
            'season': 1
        }
        
        state = StateEncoder.encode_state(self.agent, local_env)
        
        assert isinstance(state, str)
        assert "hunger_" in state
        assert "energy_" in state
        assert "health_" in state
        assert "pos_5_5" in state
        assert "season_1" in state
    
    def test_discretize(self):
        """Тест дискретизации"""
        # Тест с разными значениями
        assert StateEncoder._discretize(0.0, 5) == 0
        assert StateEncoder._discretize(0.5, 5) == 2
        assert StateEncoder._discretize(1.0, 5) == 4  # Максимальный индекс
        
        # Тест с разным количеством бинов
        assert StateEncoder._discretize(0.5, 3) == 1
        assert StateEncoder._discretize(0.8, 3) == 2
    
    def test_get_state_features(self):
        """Тест получения вектора признаков"""
        local_env = {
            'perceived_objects': [],
            'season': 0
        }
        
        features = StateEncoder.get_state_features(self.agent, local_env)
        
        assert isinstance(features, np.ndarray)
        assert len(features) > 0
        assert all(0.0 <= feature <= 1.0 for feature in features)


class TestDecisionMaker:
    """Тесты системы принятия решений"""
    
    def setup_method(self):
        """Настройка тестов"""
        self.genes = AgentGenes()
        self.agent = Agent("test_agent", (5, 5), self.genes)
        self.learner = QLearningAgent()
        self.decision_maker = DecisionMaker(self.agent, self.learner)
    
    def test_select_action(self):
        """Тест выбора действия"""
        local_env = {
            'perceived_objects': [],
            'season': 0
        }
        available_actions = ["move", "rest"]
        
        action = self.decision_maker.select_action(local_env, available_actions)
        
        assert action in available_actions
    
    def test_learn_from_experience(self):
        """Тест обучения на основе опыта"""
        previous_state = "state1"
        action = "move"
        result = ActionResult("move", True, 0.5, 0.1, {"new_position": (6, 5)})
        current_state = "state2"
        available_actions = ["move", "rest"]
        
        self.decision_maker.learn_from_experience(
            previous_state, action, result, current_state, available_actions
        )
        
        # Проверяем, что Q-значение обновилось
        q_value = self.learner.get_q_value(previous_state, action)
        assert q_value > 0.0
        
        # Проверяем, что эпизод добавлен в память
        assert len(self.agent.episodic_memory.episodes) > 0
    
    def test_memory_bonus(self):
        """Тест бонуса от памяти"""
        local_env = {
            'perceived_objects': [],
            'season': 0
        }
        
        # Тестируем с пустой памятью
        bonus = self.decision_maker._calculate_memory_bonus("move", local_env)
        assert bonus >= 0.0
        
        # Добавляем успешный эпизод в память
        episode = Episode(1, "state1", "move", 1.0, "state2", {})
        self.agent.episodic_memory.add_episode(episode)
        
        # Теперь бонус должен быть выше
        bonus = self.decision_maker._calculate_memory_bonus("move", local_env)
        assert bonus > 0.0


class TestLearningManager:
    """Тесты менеджера обучения"""
    
    def setup_method(self):
        """Настройка тестов"""
        self.manager = LearningManager()
        self.agent = Agent("test_agent", (5, 5), AgentGenes())
    
    def test_register_agent(self):
        """Тест регистрации агента"""
        self.manager.register_agent(self.agent)
        
        assert self.agent.id in self.manager.learners
        assert self.manager.global_stats['active_learners'] == 1
    
    def test_unregister_agent(self):
        """Тест удаления агента"""
        self.manager.register_agent(self.agent)
        self.manager.unregister_agent(self.agent.id)
        
        assert self.agent.id not in self.manager.learners
        assert self.manager.global_stats['active_learners'] == 0
    
    def test_global_stats(self):
        """Тест глобальной статистики"""
        # Регистрируем несколько агентов
        agent1 = Agent("agent1", (1, 1), AgentGenes())
        agent2 = Agent("agent2", (2, 2), AgentGenes())
        
        self.manager.register_agent(agent1)
        self.manager.register_agent(agent2)
        
        stats = self.manager.get_global_stats()
        
        assert 'active_learners' in stats
        assert 'total_learning_episodes' in stats
        assert 'total_rewards' in stats
        assert stats['active_learners'] == 2


class TestAdvancedEpisodicMemory:
    """Тесты улучшенной эпизодической памяти"""
    
    def setup_method(self):
        """Настройка тестов"""
        self.memory = AdvancedEpisodicMemory(capacity=5)
    
    def test_add_episode_with_priority(self):
        """Тест добавления эпизода с приоритетом"""
        # Эпизод с высокой наградой
        high_reward_episode = Episode(1, "state1", "action1", 1.0, "state2", {})
        self.memory.add_episode(high_reward_episode)
        
        # Эпизод с низкой наградой
        low_reward_episode = Episode(2, "state2", "action2", -0.5, "state3", {})
        self.memory.add_episode(low_reward_episode)
        
        # Эпизод с открытием
        discovery_episode = Episode(3, "state3", "action3", 0.5, "state4", {
            'data': {'discovery': True}
        })
        self.memory.add_episode(discovery_episode)
        
        assert len(self.memory.episodes) == 3
        
        # Эпизод с открытием должен иметь высокий приоритет
        # и быть ближе к началу списка
        episodes = [ep for ep, _ in self.memory.episodes]
        discovery_index = episodes.index(discovery_episode)
        low_reward_index = episodes.index(low_reward_episode)
        
        assert discovery_index < low_reward_index
    
    def test_recall_with_context_filter(self):
        """Тест воспоминания с фильтром контекста"""
        episode1 = Episode(1, "state1", "action1", 0.5, "state2", {'energy': 0.8})
        episode2 = Episode(2, "state2", "action2", 0.3, "state3", {'energy': 0.2})
        
        self.memory.add_episode(episode1)
        self.memory.add_episode(episode2)
        
        # Фильтр по высокой энергии
        context_filter = {'energy': 0.8}
        recalled = self.memory.recall_similar("state1", context_filter=context_filter)
        
        assert len(recalled) == 1
        assert recalled[0].timestamp == 1
    
    def test_memory_statistics(self):
        """Тест статистики памяти"""
        for i in range(5):
            episode = Episode(i, f"state{i}", f"action{i}", i * 0.2, f"state{i+1}", {})
            self.memory.add_episode(episode)
        
        stats = self.memory.get_memory_statistics()
        
        assert stats['total_episodes'] == 5
        assert 'average_reward' in stats
        assert 'action_distribution' in stats
        assert 'time_span' in stats


class TestSemanticMemory:
    """Тесты семантической памяти"""
    
    def setup_method(self):
        """Настройка тестов"""
        self.memory = SemanticMemory()
    
    def test_add_concept(self):
        """Тест добавления концепции"""
        concept_id = self.memory.add_concept(
            "object", 
            {"type": "stone", "hardness": 0.8}
        )
        
        assert concept_id in self.memory.concepts
        assert self.memory.concepts[concept_id]['type'] == "object"
        assert self.memory.concepts[concept_id]['features']['type'] == "stone"
    
    def test_add_relation(self):
        """Тест добавления отношения"""
        concept1_id = self.memory.add_concept("object", {"type": "stone"})
        concept2_id = self.memory.add_concept("object", {"type": "wood"})
        
        relation_id = self.memory.add_relation(
            concept1_id, concept2_id, "can_combine", 0.8
        )
        
        assert relation_id in self.memory.relations
        assert self.memory.relations[relation_id][2] == "can_combine"
    
    def test_find_concepts(self):
        """Тест поиска концепций"""
        self.memory.add_concept("object", {"type": "stone", "hardness": 0.8})
        self.memory.add_concept("object", {"type": "wood", "hardness": 0.4})
        self.memory.add_concept("tool", {"type": "hammer"})
        
        # Поиск по типу
        object_concepts = self.memory.find_concepts(concept_type="object")
        assert len(object_concepts) == 2
        
        # Поиск по признакам
        hard_objects = self.memory.find_concepts(
            features_filter={"hardness": 0.8}
        )
        assert len(hard_objects) == 1
    
    def test_concept_activation(self):
        """Тест активации концепций"""
        concept1_id = self.memory.add_concept("object", {"type": "stone"})
        concept2_id = self.memory.add_concept("object", {"type": "wood"})
        
        # Добавляем отношение
        self.memory.add_relation(concept1_id, concept2_id, "related", 0.5)
        
        # Активируем первую концепцию
        self.memory.activate_concept(concept1_id, 1.0)
        
        # Вторая концепция тоже должна активироваться (через отношение)
        assert self.memory.concepts[concept1_id]['activation'] > 0
        assert self.memory.concepts[concept2_id]['activation'] > 0


if __name__ == '__main__':
    pytest.main([__file__])
