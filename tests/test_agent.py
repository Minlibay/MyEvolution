"""
Tests for agent module
"""

import pytest
from src.core.agent import Agent, AgentGenes, EpisodicMemory, StatisticalMemory, AgentFactory, Episode
from src.core.objects import ObjectFactory


class TestAgentGenes:
    """Тесты генов агента"""
    
    def test_gene_validation(self):
        """Тест валидации генов"""
        # Нормальные гены
        genes = AgentGenes(
            metabolism_speed=0.5,
            strength=0.7,
            intelligence=0.3,
            social_tendency=0.8,
            exploration_bias=0.4
        )
        assert genes.metabolism_speed == 0.5
        assert genes.strength == 0.7
    
    def test_gene_validation_error(self):
        """Тест ошибки валидации генов"""
        with pytest.raises(ValueError):
            AgentGenes(metabolism_speed=1.5)  # Выходит за пределы [0,1]
    
    def test_mutation(self):
        """Тест мутации генов"""
        genes = AgentGenes(metabolism_speed=0.5)
        original_value = genes.metabolism_speed
        
        # Мутация с 100% вероятностью для теста
        genes.mutate(mutation_rate=1.0)
        
        # Значение должно измениться
        assert genes.metabolism_speed != original_value
        assert 0.0 <= genes.metabolism_speed <= 1.0
    
    def test_crossover(self):
        """Тест кроссовера генов"""
        genes1 = AgentGenes(metabolism_speed=0.2, strength=0.8)
        genes2 = AgentGenes(metabolism_speed=0.9, strength=0.1)
        
        child_genes = genes1.crossover(genes2)
        
        # Ребенок должен иметь гены от одного из родителей
        assert child_genes.metabolism_speed in [0.2, 0.9]
        assert child_genes.strength in [0.8, 0.1]


class TestEpisodicMemory:
    """Тесты эпизодической памяти"""
    
    def test_add_episode(self):
        """Тест добавления эпизода"""
        memory = EpisodicMemory(capacity=3)
        
        episode1 = Episode(1, "state1", "action1", 0.5, "state2", {})
        episode2 = Episode(2, "state2", "action2", 0.3, "state3", {})
        
        memory.add_episode(episode1)
        memory.add_episode(episode2)
        
        assert len(memory.episodes) == 2
        assert memory.episodes[0] == episode1
        assert memory.episodes[1] == episode2
    
    def test_capacity_limit(self):
        """Тест ограничения емкости памяти"""
        memory = EpisodicMemory(capacity=2)
        
        for i in range(4):
            episode = Episode(i, f"state{i}", f"action{i}", 0.1, f"state{i+1}", {})
            memory.add_episode(episode)
        
        # Должно остаться только 2 последних эпизода
        assert len(memory.episodes) == 2
        assert memory.episodes[0].timestamp == 2
        assert memory.episodes[1].timestamp == 3
    
    def test_recall_similar(self):
        """Тест воспоминания похожих эпизодов"""
        memory = EpisodicMemory()
        
        episode1 = Episode(1, "hunger_high|energy_low", "eat", 1.0, "hunger_low", {})
        episode2 = Episode(2, "hunger_high|energy_high", "move", 0.1, "hunger_high", {})
        episode3 = Episode(3, "health_low|energy_low", "rest", 0.5, "health_medium", {})
        
        memory.add_episode(episode1)
        memory.add_episode(episode2)
        memory.add_episode(episode3)
        
        similar = memory.recall_similar("hunger_high|energy_medium", k=2)
        
        # Должны вернуть эпизоды с hunger_high
        assert len(similar) <= 2
        assert any(ep.timestamp == 1 for ep in similar)
        assert any(ep.timestamp == 2 for ep in similar)


class TestStatisticalMemory:
    """Тесты статистической памяти"""
    
    def test_update_statistic(self):
        """Тест обновления статистики"""
        memory = StatisticalMemory()
        
        memory.update_statistic("test_key", 0.5)
        assert memory.get_statistic("test_key") == 0.5
        
        memory.update_statistic("test_key", 1.0)
        # Должно быть скользящее среднее
        assert 0.5 < memory.get_statistic("test_key") < 1.0
    
    def test_get_top_statistics(self):
        """Тест получения топ статистик"""
        memory = StatisticalMemory()
        
        memory.update_statistic("key1", 0.8)
        memory.update_statistic("key2", 0.6)
        memory.update_statistic("key3", 0.9)
        
        top_stats = memory.get_top_statistics(2)
        
        assert len(top_stats) == 2
        assert top_stats[0][0] == "key3"  # Наибольшее значение
        assert top_stats[1][0] == "key1"
    
    def test_decay(self):
        """Тест затухания статистик"""
        memory = StatisticalMemory()
        
        memory.update_statistic("test_key", 1.0)
        original_value = memory.get_statistic("test_key")
        
        memory.decay_all()
        
        # Значение должно уменьшиться
        assert memory.get_statistic("test_key") < original_value


class TestAgent:
    """Тесты агента"""
    
    def setup_method(self):
        """Настройка тестового агента"""
        self.genes = AgentGenes(
            metabolism_speed=0.5,
            strength=0.6,
            intelligence=0.7,
            social_tendency=0.4,
            exploration_bias=0.3
        )
        
        self.agent = Agent(
            id="test_agent",
            position=(5, 5),
            genes=self.genes
        )
    
    def test_initialization(self):
        """Тест инициализации агента"""
        assert self.agent.id == "test_agent"
        assert self.agent.position == (5, 5)
        assert self.agent.hunger == 0.0
        assert self.agent.health == 1.0
        assert self.agent.energy == 1.0
        assert self.agent.age == 0
        assert self.agent.exploration_rate == 0.3  # Из генов
    
    def test_update_physiology(self):
        """Тест обновления физиологии"""
        initial_hunger = self.agent.hunger
        initial_age = self.agent.age
        
        self.agent.update_physiology()
        
        # Голод должен увеличиться
        assert self.agent.hunger > initial_hunger
        # Возраст должен увеличиться
        assert self.agent.age > initial_age
    
    def test_is_alive(self):
        """Тест проверки жизнеспособности"""
        assert self.agent.is_alive() is True
        
        # Мертвый от здоровья
        self.agent.health = 0.0
        assert self.agent.is_alive() is False
        
        # Восстанавливаем здоровье
        self.agent.health = 1.0
        
        # Мертвый от возраста
        self.agent.age = 6000  # Больше max_age
        assert self.agent.is_alive() is False
        
        # Восстанавливаем возраст
        self.agent.age = 100
        
        # Мертвый от голода
        self.agent.hunger = 1.0
        assert self.agent.is_alive() is False
    
    def test_inventory_operations(self):
        """Тест операций с инвентарем"""
        # Добавление объекта
        result = self.agent.add_to_inventory("obj1")
        assert result is True
        assert "obj1" in self.agent.inventory
        
        # Удаление объекта
        result = self.agent.remove_from_inventory("obj1")
        assert result is True
        assert "obj1" not in self.agent.inventory
        
        # Попытка удалить несуществующий объект
        result = self.agent.remove_from_inventory("nonexistent")
        assert result is False
        
        # Попытка добавить превысив емкость
        for i in range(self.agent.inventory_capacity):
            self.agent.add_to_inventory(f"obj{i}")
        
        # Инвентарь должен быть заполнен до лимита
        assert len(self.agent.inventory) <= self.agent.inventory_capacity
    
    def test_tool_operations(self):
        """Тест операций с инструментами"""
        # Добавление инструмента
        result = self.agent.add_tool("tool1")
        assert result is True
        assert "tool1" in self.agent.tools
        
        # Удаление инструмента
        result = self.agent.remove_tool("tool1")
        assert result is True
        assert "tool1" not in self.agent.tools
        
        # Попытка удалить несуществующий инструмент
        result = self.agent.remove_tool("nonexistent")
        assert result is False
        
        # Попытка добавить превысив лимит
        for i in range(5):  # Лимит = 3
            self.agent.add_tool(f"tool{i}")
        
        # Количество инструментов должно быть в пределах лимита
        assert len(self.agent.tools) <= 3
    
    def test_can_reproduce(self):
        """Тест проверки возможности размножения"""
        # Не может размножаться при низком здоровье
        self.agent.health = 0.5
        assert self.agent.can_reproduce() is False
        
        # Не может размножаться при низком возрасте
        self.agent.health = 1.0
        self.agent.age = 100
        assert self.agent.can_reproduce() is False
        
        # Не может размножаться при низкой энергии
        self.agent.age = 600
        self.agent.energy = 0.3
        assert self.agent.can_reproduce() is False
        
        # Не может размножаться при высоком голоде
        self.agent.energy = 0.8
        self.agent.hunger = 0.8
        assert self.agent.can_reproduce() is False
        
        # Может размножаться при хороших условиях
        self.agent.hunger = 0.3
        assert self.agent.can_reproduce() is True
    
    def test_get_fitness(self):
        """Тест расчета приспособленности"""
        fitness = self.agent.get_fitness()
        assert fitness > 0.0
        
        # Открытия увеличивают приспособленность
        self.agent.discoveries_made.append("discovery1")
        new_fitness = self.agent.get_fitness()
        assert new_fitness > fitness
        
        # Потомство увеличивает приспособленность
        self.agent.offspring_count = 2
        new_fitness = self.agent.get_fitness()
        assert new_fitness > fitness


class TestAgentFactory:
    """Тесты фабрики агентов"""
    
    def test_create_random_agent(self):
        """Тест создания случайного агента"""
        agent = AgentFactory.create_random_agent("random_agent", (10, 10))
        
        assert agent.id == "random_agent"
        assert agent.position == (10, 10)
        assert 0.0 <= agent.genes.metabolism_speed <= 1.0
        assert 0.0 <= agent.genes.strength <= 1.0
        assert 0.0 <= agent.genes.intelligence <= 1.0
    
    def test_create_offspring(self):
        """Тест создания потомка"""
        parent1 = AgentFactory.create_random_agent("parent1", (5, 5))
        parent2 = AgentFactory.create_random_agent("parent2", (6, 6))
        
        offspring = AgentFactory.create_offspring(
            parent1, parent2, "child", 100
        )
        
        assert offspring.id == "child"
        assert offspring.birth_time == 100
        assert offspring.position[0] in [3, 4, 5, 6, 7]  # Near parent1
        assert offspring.position[1] in [3, 4, 5, 6, 7]
        
        # Гены должны быть от родителей
        assert offspring.genes.metabolism_speed in [
            parent1.genes.metabolism_speed, 
            parent2.genes.metabolism_speed
        ]


if __name__ == '__main__':
    pytest.main([__file__])
