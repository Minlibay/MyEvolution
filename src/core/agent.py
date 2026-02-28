"""
Agent module - defines agents with their genetics and behavior
"""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Any, Optional
import random
import numpy as np

from .objects import Object
from .tools import Tool


@dataclass
class AgentGenes:
    """Генетические параметры агента"""
    metabolism_speed: float = 0.5      # Скорость метаболизма
    strength: float = 0.5             # Физическая сила
    intelligence: float = 0.5         # Интеллект (скорость обучения)
    social_tendency: float = 0.5     # Склонность к социальному поведению
    exploration_bias: float = 0.5    # Генетическая склонность к исследованию
    
    def __post_init__(self):
        """Валидация генов после инициализации"""
        self._validate_genes()
    
    def _validate_genes(self):
        """Проверяет, что все гены в диапазоне [0,1]"""
        for gene_name, gene_value in self.__dict__.items():
            if not 0.0 <= gene_value <= 1.0:
                raise ValueError(f"Gene {gene_name} ({gene_value}) must be in range [0,1]")
    
    def mutate(self, mutation_rate: float = 0.1):
        """Применяет мутации к генам"""
        for field_name in self.__dict__:
            if random.random() < mutation_rate:
                current_value = getattr(self, field_name)
                mutation = random.gauss(0, 0.1)
                new_value = max(0.0, min(1.0, current_value + mutation))
                setattr(self, field_name, new_value)
    
    def crossover(self, other: 'AgentGenes') -> 'AgentGenes':
        """Создает потомка через кроссовер с другим набором генов"""
        child_genes = {}
        
        for field_name in self.__dict__:
            # Случайный выбор гена от одного из родителей
            if random.random() < 0.5:
                child_genes[field_name] = getattr(self, field_name)
            else:
                child_genes[field_name] = getattr(other, field_name)
        
        return AgentGenes(**child_genes)
    
    def to_dict(self) -> Dict[str, float]:
        """Преобразует гены в словарь"""
        return self.__dict__.copy()
    
    @classmethod
    def from_dict(cls, data: Dict[str, float]) -> 'AgentGenes':
        """Создает гены из словаря"""
        return cls(**data)


@dataclass
class Episode:
    """Эпизод памяти"""
    timestamp: int
    state: str
    action: str
    reward: float
    next_state: str
    context: Dict[str, Any]


class EpisodicMemory:
    """Эпизодическая память"""
    
    def __init__(self, capacity: int = 100):
        self.capacity = capacity
        self.episodes: List[Episode] = []
    
    def add_episode(self, episode: Episode):
        """Добавляет эпизод в память"""
        self.episodes.append(episode)
        
        # Удаляем старые эпизоды если превышена емкость
        if len(self.episodes) > self.capacity:
            self.episodes.pop(0)
    
    def recall_similar(self, state: str, k: int = 5) -> List[Episode]:
        """Вспоминает похожие эпизоды"""
        if not self.episodes:
            return []
        
        # Простое сходство based on совпадение ключевых слов в состоянии
        similarities = []
        for episode in self.episodes:
            similarity = self._calculate_similarity(state, episode.state)
            similarities.append((similarity, episode))
        
        # Сортируем по схожести и возвращаем top-k
        similarities.sort(key=lambda x: x[0], reverse=True)
        return [episode for _, episode in similarities[:k]]
    
    def _calculate_similarity(self, state1: str, state2: str) -> float:
        """Рассчитывает сходство между состояниями"""
        words1 = set(state1.split('|'))
        words2 = set(state2.split('|'))
        
        if not words1 or not words2:
            return 0.0
        
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        
        return intersection / union if union > 0 else 0.0
    
    def forget_old_episodes(self, age_threshold: int):
        """Забывает старые эпизоды"""
        current_time = max(ep.timestamp for ep in self.episodes) if self.episodes else 0
        
        self.episodes = [
            ep for ep in self.episodes 
            if current_time - ep.timestamp <= age_threshold
        ]


class StatisticalMemory:
    """Статистическая память"""
    
    def __init__(self):
        self.statistics: Dict[str, float] = {}
        self.decay_rate = 0.99  # Скорость "забывания"
    
    def update_statistic(self, key: str, value: float):
        """Обновляет статистику"""
        if key in self.statistics:
            # Экспоненциальное скользящее среднее
            self.statistics[key] = (
                self.statistics[key] * self.decay_rate + 
                value * (1 - self.decay_rate)
            )
        else:
            self.statistics[key] = value
    
    def get_statistic(self, key: str) -> float:
        """Возвращает статистику"""
        return self.statistics.get(key, 0.0)
    
    def get_top_statistics(self, n: int) -> List[Tuple[str, float]]:
        """Возвращает top-n статистик"""
        sorted_stats = sorted(
            self.statistics.items(), 
            key=lambda x: x[1], 
            reverse=True
        )
        return sorted_stats[:n]
    
    def decay_all(self):
        """Применяет затухание ко всем статистикам"""
        for key in self.statistics:
            self.statistics[key] *= self.decay_rate


@dataclass
class Agent:
    """Агент - модель раннего человека"""
    id: str
    position: Tuple[int, int]
    genes: AgentGenes

    # Идентичность / демография
    sex: str = "unknown"  # male/female
    display_name: Optional[str] = None
    
    # Физиологические параметры
    hunger: float = 0.0      # Голод [0,1]
    thirst: float = 0.0      # Жажда [0,1]
    sleepiness: float = 0.0  # Сонливость [0,1]
    health: float = 1.0      # Здоровье [0,1]
    energy: float = 1.0      # Энергия [0,1]
    age: int = 0             # Возраст в шагах симуляции
    max_age: int = 5000      # Максимальный возраст
    
    # Психологические параметры
    exploration_rate: float = 0.1  # Склонность к исследованию [0,1]
    risk_tolerance: float = 0.5    # Толерантность к риску [0,1]
    memory_capacity: int = 100     # Объем памяти
    perception_radius: int = 2     # Радиус восприятия
    inventory_capacity: int = 5    # Вместимость инвентаря
    
    # Память
    episodic_memory: EpisodicMemory = field(default_factory=EpisodicMemory)
    statistical_memory: StatisticalMemory = field(default_factory=StatisticalMemory)
    
    # Обучение (Q-таблица)
    q_table: Dict[Tuple[str, str], float] = field(default_factory=dict)
    learning_rate: float = 0.1
    discount_factor: float = 0.95
    
    # Инвентарь
    inventory: List[str] = field(default_factory=list)  # ID объектов
    tools: List[str] = field(default_factory=list)     # ID инструментов
    
    # Метрики
    discoveries_made: List[str] = field(default_factory=list)
    offspring_count: int = 0
    total_reward: float = 0.0
    
    # Временные метки
    last_action_time: int = 0
    birth_time: int = 0

    # Репродукция / развитие
    pregnant: bool = False
    pregnancy_father_id: Optional[str] = None
    pregnancy_remaining: int = 0
    gestation_length: int = 300
    mother_id: Optional[str] = None
    father_id: Optional[str] = None

    # Социальная зависимость (детёныш)
    adulthood_age: int = 800

    # Коммуникация (эмерджентный лексикон)
    lexicon_out: Dict[str, Dict[str, float]] = field(default_factory=dict)  # meaning -> token -> weight
    lexicon_in: Dict[str, Dict[str, float]] = field(default_factory=dict)   # token -> meaning -> weight
    last_utterance: Optional[str] = None
    last_intended_meaning: Optional[str] = None
    last_heard: Optional[str] = None

    def __post_init__(self):
        """Инициализация после создания"""
        # Устанавливаем exploration_rate на основе генов
        self.exploration_rate = self.genes.exploration_bias
        
        # Инициализируем память с правильной емкостью
        self.episodic_memory = EpisodicMemory(self.memory_capacity)

    def _invent_token(self) -> str:
        consonants = "bdgklmnprstfv"
        vowels = "aeiou"
        syl = lambda: random.choice(consonants) + random.choice(vowels)
        return syl() + syl()

    def invent_name(self) -> str:
        """Придумывает имя (фонетически похоже на токены языка агента)."""
        name = self._invent_token()
        return name.capitalize()

    def is_child(self) -> bool:
        return self.age < self.adulthood_age

    def choose_token_for_meaning(self, meaning: str, epsilon: float = 0.2) -> str:
        """Выбирает токен для смысла. Если нет подходящих, изобретает новый."""
        mapping = self.lexicon_out.setdefault(meaning, {})

        if not mapping or random.random() < epsilon:
            token = self._invent_token()
            mapping[token] = mapping.get(token, 0.0) + 1.0
        else:
            token = max(mapping.items(), key=lambda kv: kv[1])[0]

        # Обеспечиваем, что у слушателя есть шанс выучить эту связь
        self.lexicon_in.setdefault(token, {})
        return token

    def interpret_token(self, token: str) -> Optional[str]:
        """Интерпретирует токен в смысл. Возвращает None, если знаний нет."""
        meanings = self.lexicon_in.get(token)
        if not meanings:
            return None
        return max(meanings.items(), key=lambda kv: kv[1])[0]

    def update_communication(self, meaning: str, token: str, success: bool, lr: float = 0.2):
        """Усиливает/ослабляет связи meaning<->token в обоих направлениях."""
        out_map = self.lexicon_out.setdefault(meaning, {})
        in_map = self.lexicon_in.setdefault(token, {})

        delta = lr if success else -lr * 0.5

        out_map[token] = max(0.0, out_map.get(token, 0.0) + delta)
        in_map[meaning] = max(0.0, in_map.get(meaning, 0.0) + delta)

        # Небольшое затухание альтернатив, чтобы стабилизировать выбор
        for t in list(out_map.keys()):
            if t != token:
                out_map[t] *= 0.995
        for m in list(in_map.keys()):
            if m != meaning:
                in_map[m] *= 0.995

    def choose_communication_meaning(self, local_env: Dict[str, Any]) -> str:
        """Выбирает смысл сообщения из наблюдаемого состояния (без захардкоженных слов)."""
        # Набор смыслов фиксирован как исследовательский «набор задач», токены эволюционируют сами.
        if self.hunger > 0.6:
            return "need_food"
        perceived = local_env.get('perceived_objects', []) or []
        if any(getattr(o, 'is_edible', lambda: False)() for o in perceived):
            return "food_here"
        return "idle"
    
    def perceive(self, environment) -> Dict[str, Any]:
        """Воспринимает локальную среду"""
        is_daytime = getattr(environment, 'is_daytime', True)
        effective_radius = self.perception_radius if is_daytime else max(1, self.perception_radius - 1)

        local_env = environment.get_local_environment(
            self.position,
            effective_radius,
        )
        
        # Фильтрация объектов на основе памяти и интеллекта
        perceived_objects = []
        for obj in local_env['objects']:
            # Агент лучше воспринимает знакомые объекты
            familiarity = self.statistical_memory.get_statistic(f"object_{obj.type}")
            perception_prob = 0.5 + familiarity * 0.3 + self.genes.intelligence * 0.2

            # Ночью хуже видно
            if not local_env.get('is_daytime', True):
                perception_prob *= 0.6
            
            if random.random() < perception_prob:
                perceived_objects.append(obj)
        
        local_env['perceived_objects'] = perceived_objects
        return local_env
    
    def get_available_actions(self, local_env: Dict[str, Any]) -> List[str]:
        """Возвращает доступные действия в текущей ситуации"""
        actions = ['move', 'rest']
        
        # Проверяем наличие энергии для действий
        if self.energy > 0.1:
            actions.append('move')
            
            # Можно собрать объекты если они есть
            if local_env['perceived_objects']:
                actions.append('gather')
            
            # Можно съесть что-то если есть в инвентаре
            if self.inventory:
                actions.append('consume')
            
            # Можно скомбинировать объекты если >= 2 в инвентаре
            if len(self.inventory) >= 2:
                actions.append('combine')
            
            # Можно использовать инструменты
            if self.tools:
                actions.extend(['attack', 'break'])
        
        return actions
    
    def update_physiology(self):
        """Обновляет физиологическое состояние"""
        # Увеличение голода based on метаболизм
        hunger_increase = self.genes.metabolism_speed * 0.01
        self.hunger = min(1.0, self.hunger + hunger_increase)

        # Увеличение жажды (чуть медленнее, но критичнее)
        thirst_increase = 0.006 + self.genes.metabolism_speed * 0.004
        self.thirst = min(1.0, self.thirst + thirst_increase)

        # Сонливость накапливается: базово растёт, но учитывает цикл день/ночь
        # (само действие sleep будет резко снижать sleepiness)
        # Ночью сонливость растёт быстрее, чтобы формировать устойчивый ночной сон.
        is_daytime = getattr(self, 'is_daytime', None)
        if is_daytime is None:
            sleep_increase = 0.004
        else:
            sleep_increase = 0.003 if is_daytime else 0.006
        self.sleepiness = min(1.0, self.sleepiness + sleep_increase)
        
        # Влияние голода на здоровье
        if self.hunger > 0.8:
            self.health -= 0.02
        elif self.hunger < 0.2:
            self.health = min(1.0, self.health + 0.01)

        # Влияние жажды на здоровье
        if self.thirst > 0.8:
            self.health -= 0.03

        # Влияние сильной сонливости на здоровье
        if self.sleepiness > 0.9:
            self.health -= 0.01
        
        # Восстановление энергии при отдыхе
        energy_recovery = 0.01 * (1 + self.genes.strength * 0.5)
        self.energy = min(1.0, self.energy + energy_recovery)
        
        # Увеличение возраста
        self.age += 1
        
        # Обновление exploration rate (ε-decay)
        self.exploration_rate *= 0.9995
        self.exploration_rate = max(0.01, self.exploration_rate)
        
        # Применение затухания к статистической памяти
        if self.age % 100 == 0:
            self.statistical_memory.decay_all()
    
    def is_alive(self) -> bool:
        """Проверяет, жив ли агент"""
        return (self.health > 0 and 
                self.age < self.max_age and 
                self.hunger < 1.0 and
                self.thirst < 1.0)
    
    def get_fitness(self) -> float:
        """Рассчитывает приспособленность агента"""
        # Базовая приспособленность based on выживаемость
        base_fitness = self.age * self.health
        
        # Бонус за открытия
        discovery_bonus = len(self.discoveries_made) * 10
        
        # Бонус за репродукцию
        reproduction_bonus = self.offspring_count * 5
        
        # Бонус за накопленные ресурсы
        resource_bonus = len(self.inventory) * 2
        
        # Штраф за низкую энергию
        energy_penalty = (1.0 - self.energy) * 2
        
        total_fitness = (base_fitness + discovery_bonus + 
                        reproduction_bonus + resource_bonus - energy_penalty)
        
        return max(0.1, total_fitness)
    
    def can_reproduce(self) -> bool:
        """Проверяет, может ли агент размножаться"""
        return (self.health > 0.7 and 
                self.age > 500 and 
                self.energy > 0.6 and
                self.hunger < 0.5)
    
    def add_to_inventory(self, obj_id: str) -> bool:
        """Добавляет объект в инвентарь"""
        if len(self.inventory) < self.inventory_capacity:
            self.inventory.append(obj_id)
            return True
        return False
    
    def remove_from_inventory(self, obj_id: str) -> bool:
        """Удаляет объект из инвентаря"""
        if obj_id in self.inventory:
            self.inventory.remove(obj_id)
            return True
        return False
    
    def add_tool(self, tool_id: str) -> bool:
        """Добавляет инструмент"""
        if len(self.tools) < 3:  # Ограничение на количество инструментов
            self.tools.append(tool_id)
            return True
        return False
    
    def remove_tool(self, tool_id: str) -> bool:
        """Удаляет инструмент"""
        if tool_id in self.tools:
            self.tools.remove(tool_id)
            return True
        return False
    
    def get_best_tool_for_action(self, action: str, environment) -> Optional[str]:
        """Возвращает лучший инструмент для действия"""
        if not self.tools:
            return None
        
        best_tool_id = None
        best_effectiveness = 1.0  # Базовая эффективность без инструмента
        
        for tool_id in self.tools:
            tool = environment.tools.get(tool_id)
            if tool:
                effectiveness = tool.calculate_effectiveness(action)
                if effectiveness > best_effectiveness:
                    best_effectiveness = effectiveness
                    best_tool_id = tool_id
        
        return best_tool_id
    
    def to_dict(self) -> Dict[str, Any]:
        """Преобразует агента в словарь для сериализации"""
        return {
            'id': self.id,
            'position': self.position,
            'genes': self.genes.to_dict(),
            'hunger': self.hunger,
            'health': self.health,
            'energy': self.energy,
            'age': self.age,
            'max_age': self.max_age,
            'exploration_rate': self.exploration_rate,
            'risk_tolerance': self.risk_tolerance,
            'memory_capacity': self.memory_capacity,
            'perception_radius': self.perception_radius,
            'inventory_capacity': self.inventory_capacity,
            'inventory': self.inventory,
            'tools': self.tools,
            'discoveries_made': self.discoveries_made,
            'offspring_count': self.offspring_count,
            'total_reward': self.total_reward,
            'last_action_time': self.last_action_time,
            'birth_time': self.birth_time
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Agent':
        """Создает агента из словаря"""
        genes = AgentGenes.from_dict(data['genes'])
        
        agent = cls(
            id=data['id'],
            position=tuple(data['position']),
            genes=genes,
            hunger=data['hunger'],
            health=data['health'],
            energy=data['energy'],
            age=data['age'],
            max_age=data['max_age'],
            exploration_rate=data['exploration_rate'],
            risk_tolerance=data['risk_tolerance'],
            memory_capacity=data['memory_capacity'],
            perception_radius=data['perception_radius'],
            inventory_capacity=data['inventory_capacity'],
            inventory=data['inventory'],
            tools=data['tools'],
            discoveries_made=data['discoveries_made'],
            offspring_count=data['offspring_count'],
            total_reward=data['total_reward'],
            last_action_time=data['last_action_time'],
            birth_time=data['birth_time']
        )
        
        return agent


class AgentFactory:
    """Фабрика для создания агентов"""
    
    @staticmethod
    def create_random_agent(agent_id: str, position: Tuple[int, int], 
                          birth_time: int = 0) -> Agent:
        """Создает агента со случайными генами"""
        genes = AgentGenes(
            metabolism_speed=random.random(),
            strength=random.random(),
            intelligence=random.random(),
            social_tendency=random.random(),
            exploration_bias=random.random()
        )
        
        agent = Agent(
            id=agent_id,
            position=position,
            genes=genes,
            birth_time=birth_time
        )
        
        return agent
    
    @staticmethod
    def create_offspring(parent1: Agent, parent2: Agent, 
                         child_id: str, birth_time: int) -> Agent:
        """Создает потомка от двух родителей"""
        # Генетический кроссовер
        child_genes = parent1.genes.crossover(parent2.genes)
        child_genes.mutate(mutation_rate=0.1)
        
        # Позиция near родителя
        child_position = (
            max(0, parent1.position[0] + random.randint(-2, 2)),
            max(0, parent1.position[1] + random.randint(-2, 2))
        )
        
        child = Agent(
            id=child_id,
            position=child_position,
            genes=child_genes,
            birth_time=birth_time
        )
        
        # Культурная передача знаний
        AgentFactory._transfer_knowledge(parent1, child)
        AgentFactory._transfer_knowledge(parent2, child)
        
        return child
    
    @staticmethod
    def _transfer_knowledge(parent: Agent, child: Agent):
        """Передает знания от родителя потомку"""
        # Передача части статистической памяти
        for key, value in parent.statistical_memory.statistics.items():
            if random.random() < 0.3:  # 30% вероятность передачи
                child.statistical_memory.update_statistic(key, value * 0.8)
        
        # Передача части Q-значений
        for (state, action), q_value in parent.q_table.items():
            if random.random() < 0.2:  # 20% вероятность передачи
                child.q_table[(state, action)] = q_value * 0.7

        # Передача части лексикона (выборочно, чтобы не перенасыщать)
        try:
            for meaning, token_map in list(parent.lexicon_out.items())[:10]:
                if random.random() >= 0.25:  # 25% шанс на meaning
                    continue
                # Топ-1 токен по весу
                if not token_map:
                    continue
                top_token = max(token_map.items(), key=lambda kv: kv[1])[0]
                child.lexicon_out.setdefault(meaning, {})[top_token] = float(token_map.get(top_token, 0.0)) * 0.7
                # Также усилим понимание в lexicon_in
                child.lexicon_in.setdefault(top_token, {})[meaning] = float(token_map.get(top_token, 0.0)) * 0.7
        except Exception:
            pass
