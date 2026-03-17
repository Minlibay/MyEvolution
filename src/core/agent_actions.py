"""
Agent actions module - defines agent actions and their execution
"""

from typing import Dict, List, Tuple, Any, Optional
import random

from .agent import Agent
from .objects import Object, ObjectFactory
from .tools import Tool, ToolFactory
from .environment import Environment


# ── Рецепты построек (Settlement System) ──────────────────────────────────
BUILDING_RECIPES = {
    'storage_hut': {
        'materials': {'wood': 8, 'rope': 2},
        'skill': ('crafting', 0.3),
        'energy_cost': 0.20,
        'emoji': '🏚️',
        'label': 'Склад',
    },
    'workshop': {
        'materials': {'wood': 6, 'stone': 4, 'rope': 1},
        'skill': ('crafting', 0.5),
        'energy_cost': 0.30,
        'emoji': '🔨',
        'label': 'Мастерская',
    },
    'garden': {
        'materials': {'plant': 3, 'wood': 2, 'clay': 1},
        'skill': ('gathering', 0.3),
        'energy_cost': 0.15,
        'emoji': '🌱',
        'label': 'Огород',
    },
    'well': {
        'materials': {'stone': 6, 'rope': 2},
        'skill': ('crafting', 0.4),
        'energy_cost': 0.25,
        'emoji': '🪣',
        'label': 'Колодец',
    },
    'watchtower': {
        'materials': {'wood': 10, 'stone': 4},
        'skill': ('survival', 0.4),
        'energy_cost': 0.30,
        'emoji': '🗼',
        'label': 'Дозорная башня',
    },
    'drying_rack': {
        'materials': {'wood': 4, 'rope': 2},
        'skill': ('crafting', 0.2),
        'energy_cost': 0.12,
        'emoji': '🥩',
        'label': 'Сушилка',
    },
    'trading_post': {
        'materials': {'wood': 6, 'stone': 4, 'rope': 2},
        'skill': ('crafting', 0.4),
        'energy_cost': 0.25,
        'emoji': '🏪',
        'label': 'Торговый пост',
    },
}

# Типы зданий, которые нельзя подбирать при сборе
BUILDING_TYPES = frozenset(BUILDING_RECIPES.keys()) | {'shelter', 'campfire', 'stone_furnace', 'clay_oven'}


class ActionResult:
    """Результат выполнения действия"""
    
    def __init__(self, action: str, success: bool, reward: float = 0.0, 
                 energy_cost: float = 0.0, data: Dict[str, Any] = None):
        self.action = action
        self.success = success
        self.reward = reward
        self.energy_cost = energy_cost
        self.data = data or {}
        self.previous_state = None
        self.new_state = None


class AgentActions:
    """Класс для выполнения действий агента"""

    @staticmethod
    def _night_multiplier(environment: Environment, position: Tuple[int, int], radius: int = 1) -> float:
        local_env = environment.get_local_environment(position, radius=radius)
        return 1.1 if not local_env.get('is_daytime', True) else 1.0

    @staticmethod
    def _performance_mult(agent) -> float:
        """Закон Либиха: производительность ограничена самым критичным физиологическим показателем.
        worst=0 → 1.0 (полная); worst=0.5 → 0.5; worst=0.9 → 0.15 (почти заблокировано)."""
        worst = max(
            getattr(agent, 'hunger', 0.0),
            getattr(agent, 'thirst', 0.0),
            getattr(agent, 'sleepiness', 0.0),
        )
        return max(0.15, 1.0 - worst)

    @staticmethod
    def execute_move(agent: Agent, environment: Environment, 
                    target_position: Optional[Tuple[int, int]] = None) -> ActionResult:
        """Выполняет движение агента"""
        # Энергетическая стоимость движения
        energy_cost = 0.05 * (1 + len(agent.inventory) * 0.1)
        _bonuses = getattr(agent, 'research_bonuses', {})
        # Ночной штраф с учётом бонуса night_penalty_mult
        _nm = AgentActions._night_multiplier(environment, agent.position, radius=1)
        if _nm > 1.0:
            _nr = min(0.9, -_bonuses.get('night_penalty_mult', 0.0))
            _nm = 1.0 + (_nm - 1.0) * (1.0 - _nr)
        energy_cost *= _nm
        # Бонус move_energy_mult (отрицательный = дешевле)
        energy_cost *= max(0.2, 1.0 + _bonuses.get('move_energy_mult', 0.0))
        
        if agent.energy < energy_cost:
            return ActionResult('move', False, -0.2, 0.0, {'reason': 'insufficient_energy'})
        
        # Получаем локальную среду
        local_env = environment.get_local_environment(agent.position, agent.perception_radius)
        
        # Определение новой позиции
        if target_position is None:
            new_position = AgentActions._choose_movement_direction(agent, environment)
        else:
            new_position = target_position
        
        if new_position is None:
            return ActionResult('move', False, -0.1, 0.0, {'reason': 'no_valid_direction'})
        
        # Перемещение
        old_position = agent.position
        agent.position = new_position
        agent.energy -= energy_cost
        
        # Награда за исследование новых территорий
        exploration_reward = 0.0
        visited_key = f"visited_{new_position[0]}_{new_position[1]}"
        if agent.statistical_memory.get_statistic(visited_key) == 0.0:
            exploration_reward = 0.1
            agent.statistical_memory.update_statistic(visited_key, 1.0)
        
        return ActionResult(
            'move', 
            True, 
            exploration_reward, 
            energy_cost,
            {'old_position': old_position, 'new_position': new_position}
        )
    
    @staticmethod
    def _choose_movement_direction(agent: Agent, environment: Environment) -> Optional[Tuple[int, int]]:
        """Выбирает направление движения"""
        directions = [
            (0, 1), (0, -1), (1, 0), (-1, 0),
            (1, 1), (-1, -1), (1, -1), (-1, 1)
        ]
        
        valid_directions = list(directions)  # все 8 направлений всегда доступны (тороидальный мир)

        # Выбор направления с учетом исследования
        if random.random() < agent.exploration_rate:
            # Исследование - выбираем случайное направление
            dx, dy = random.choice(valid_directions)
        else:
            # Эксплуатация - движемся к ресурсам
            dx, dy = AgentActions._choose_direction_towards_resources(
                agent, environment, valid_directions
            )

        new_x = (agent.position[0] + dx) % environment.width
        new_y = (agent.position[1] + dy) % environment.height
        return (new_x, new_y)
    
    @staticmethod
    def _choose_direction_towards_resources(agent: Agent, environment: Environment,
                                         valid_directions: List[Tuple[int, int]]) -> Tuple[int, int]:
        """Выбирает направление в сторону ресурсов"""
        # Простая эвристика - движемся в сторону с наибольшим количеством ресурсов
        best_direction = random.choice(valid_directions)
        best_score = 0.0
        
        for dx, dy in valid_directions:
            new_x = agent.position[0] + dx
            new_y = agent.position[1] + dy
            new_position = (new_x, new_y)
            
            # Проверяем локальную среду
            local_env = environment.get_local_environment(new_position, radius=1)
            
            # Оцениваем привлекательность позиции
            score = 0.0
            for obj in local_env.get('objects', []):
                if obj.type == 'water':
                    score += 3.0
                if obj.is_edible():
                    score += 2.0
                elif obj.is_tool_material():
                    score += 1.0
            
            # Учитываем память
            familiarity = agent.statistical_memory.get_statistic(f"position_{new_x}_{new_y}")
            score += familiarity * 0.5
            
            if score > best_score:
                best_score = score
                best_direction = (dx, dy)
        
        return best_direction
    
    @staticmethod
    def execute_gather(agent: Agent, environment: Environment) -> ActionResult:
        """Выполняет сбор объектов"""
        _bonuses = getattr(agent, 'research_bonuses', {})
        energy_cost = 0.1
        # Ночной штраф с учётом бонуса night_penalty_mult
        _nm = AgentActions._night_multiplier(environment, agent.position, radius=1)
        if _nm > 1.0:
            _nr = min(0.9, -_bonuses.get('night_penalty_mult', 0.0))
            _nm = 1.0 + (_nm - 1.0) * (1.0 - _nr)
        energy_cost *= _nm
        # Бонус gather_energy_mult
        energy_cost *= max(0.2, 1.0 + _bonuses.get('gather_energy_mult', 0.0))

        if agent.energy < energy_cost:
            return ActionResult('gather', False, -0.2, 0.0, {'reason': 'insufficient_energy'})

        # Получаем локальную среду
        local_env = environment.get_local_environment(agent.position, agent.perception_radius)

        # Радиус сбора: gather_radius=0 только текущая клетка, =1 соседние, =2 ещё дальше
        _gr = int(_bonuses.get('gather_radius', 0))
        cell_objects = environment.get_objects_at_position(agent.position)
        if _gr >= 1:
            for _dx, _dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                _np = (agent.position[0] + _dx, agent.position[1] + _dy)
                cell_objects = cell_objects + environment.get_objects_at_position(_np)
        if _gr >= 2:
            for _dx, _dy in [(-2, 0), (2, 0), (0, -2), (0, 2), (-1, -1), (-1, 1), (1, -1), (1, 1)]:
                _np = (agent.position[0] + _dx, agent.position[1] + _dy)
                cell_objects = cell_objects + environment.get_objects_at_position(_np)
        gathered_objects = []
        _inv_cap = agent.inventory_capacity + int(_bonuses.get('inventory_bonus', 0))
        _bc = _bonuses.get('bountiful_chance', 0.0)
        _gs = _bonuses.get('gather_success', 0.0)

        # Целевой сбор: предпочитаем указанный тип
        _gather_target = getattr(agent, '_gather_target_type', None)
        if _gather_target:
            setattr(agent, '_gather_target_type', None)
            # Сортируем: целевые объекты первыми
            cell_objects = sorted(cell_objects, key=lambda o: (0 if o.type == _gather_target else 1))

        for obj in cell_objects[:]:  # Копия для безопасного удаления
            if len(agent.inventory) >= _inv_cap:
                break

            # Нельзя собирать: незрелый куст, костёр, воду, размещённые структуры
            if obj.type in BUILDING_TYPES or obj.type == 'water':
                continue
            if obj.type == 'berry_bush' and not getattr(obj, 'ripe', False):
                continue

            # Вероятность успешного сбора зависит от веса объекта, силы агента и бонусов
            success_prob = min(0.98, (1.0 - obj.weight * 0.5 + agent.genes.strength * 0.3 + _gs)
                               * AgentActions._performance_mult(agent))

            if random.random() < success_prob:
                # Дерево — ресурсный узел: берём 1 единицу, оставляем объект в мире
                if obj.type == 'wood' and obj.quantity > 1:
                    import uuid as _uuid
                    piece_id = f"wood_piece_{_uuid.uuid4().hex[:8]}"
                    piece = ObjectFactory.create_object('wood', obj.position, piece_id, obj.created_at)
                    piece.quantity = 1
                    environment.objects[piece_id] = piece  # в мире не появляется, сразу в инвентарь
                    agent.add_to_inventory(piece_id)
                    obj.quantity -= 1
                    if obj.quantity <= 0:
                        environment.detach_object_from_world(obj.id)
                # Спелый ягодный куст — собираем 20 ягод, куст сбрасывается и растёт снова
                elif obj.type == 'berry_bush' and getattr(obj, 'ripe', False):
                    import uuid as _uuid
                    berry_id = f"berry_harvest_{_uuid.uuid4().hex[:8]}"
                    berry_piece = ObjectFactory.create_object('berry', obj.position, berry_id, obj.created_at)
                    berry_piece.quantity = 20
                    environment.objects[berry_id] = berry_piece
                    agent.add_to_inventory(berry_id)
                    # Куст становится не спелым, цикл роста начинается снова
                    setattr(obj, 'ripe', False)
                    obj.nutrition = 0.15
                    setattr(obj, 'planted_at', environment.timestep)
                    agent.life_log.add(environment.timestep, 'craft', 'Собрал ягоды с куста 🍓', icon='🍓')
                else:
                    agent.add_to_inventory(obj.id)
                    environment.detach_object_from_world(obj.id)
                    # bountiful_chance: шанс получить копию ресурса
                    if (_bc > 0 and random.random() < _bc
                            and len(agent.inventory) < _inv_cap):
                        import uuid as _uuid_b
                        _bid = f"bonus_{obj.type}_{_uuid_b.uuid4().hex[:8]}"
                        _bobj = ObjectFactory.create_object(
                            obj.type, obj.position, _bid, getattr(environment, 'timestep', 0))
                        environment.objects[_bid] = _bobj
                        agent.add_to_inventory(_bid)
                        gathered_objects.append(_bid)

                gathered_objects.append(obj.id)

                # Обновление статистической памяти
                memory_key = f"object_{obj.type}"
                agent.statistical_memory.update_statistic(memory_key, 1.0)
        
        # Тратим энергию только если что-то собрали
        if not gathered_objects:
            return ActionResult('gather', False, -0.01, 0.0, {'reason': 'nothing_gathered'})

        agent.energy -= energy_cost

        # Награда за собранные объекты
        reward = len(gathered_objects) * 0.05

        return ActionResult(
            'gather',
            True,
            reward,
            energy_cost,
            {'gathered_objects': gathered_objects}
        )
    
    @staticmethod
    def execute_consume(agent: Agent, environment: Environment) -> ActionResult:
        """Выполняет потребление объекта"""
        if not agent.inventory:
            return ActionResult('consume', False, -0.1, 0.0, {'reason': 'empty_inventory'})
        
        # Выбираем объект для потребления (самый питательный за 1 единицу)
        best_obj_id = None
        best_energy_value = 0.0

        for obj_id in agent.inventory:
            obj = environment.objects.get(obj_id)
            if obj and obj.is_edible():
                # Для стаков оцениваем 1 единицу, а не весь стак
                unit_value = obj.nutrition * (1.0 - obj.toxicity)
                if unit_value > best_energy_value:
                    best_energy_value = unit_value
                    best_obj_id = obj_id

        if best_obj_id is None:
            return ActionResult('consume', False, -0.1, 0.0, {'reason': 'no_edible_objects'})

        obj = environment.objects[best_obj_id]

        # Потребление объекта — стаки (ягоды) едим по 1 штуке
        if obj.type == 'berry' and obj.quantity > 1:
            energy_gain = obj.nutrition * (1.0 - obj.toxicity)
            obj.quantity -= 1
            # объект остаётся в инвентаре
        else:
            energy_gain = obj.nutrition * (1.0 - obj.toxicity)
            agent.remove_from_inventory(best_obj_id)
            environment.remove_object(best_obj_id)

        # Восстановление энергии и уменьшение голода
        agent.energy = min(1.0, agent.energy + energy_gain)
        agent.hunger = max(0.0, agent.hunger - energy_gain * 0.8)

        # Награда за потребление
        reward = energy_gain * 2.0
        
        return ActionResult(
            'consume',
            True,
            reward,
            0.0,
            {'consumed_object': best_obj_id, 'energy_gain': energy_gain}
        )

    @staticmethod
    def execute_drink(agent: Agent, environment: Environment) -> ActionResult:
        """Пьёт воду из источника в текущей клетке или из колодца рядом."""
        cell_objects = environment.get_objects_at_position(agent.position)
        has_water = any(o.type == 'water' for o in cell_objects)
        # Колодец в радиусе 2 тоже считается источником воды
        if not has_water:
            ax, ay = agent.position
            for _dx in range(-2, 3):
                for _dy in range(-2, 3):
                    for _wo in environment.get_objects_at_position((ax + _dx, ay + _dy)):
                        if getattr(_wo, 'building_type', None) == 'well':
                            has_water = True
                            # Trust бонус при использовании чужого колодца
                            _well_owner = getattr(_wo, 'building_owner_id', None)
                            if _well_owner and _well_owner != agent.id and hasattr(agent, 'social'):
                                agent.social.add_interaction(_well_owner, 0.03)
                            break
                    if has_water:
                        break
                if has_water:
                    break
        if not has_water:
            return ActionResult('drink', False, -0.02, 0.0, {'reason': 'no_water_here'})

        energy_cost = 0.01
        energy_cost *= AgentActions._night_multiplier(environment, agent.position, radius=1)
        if agent.energy < energy_cost:
            return ActionResult('drink', False, -0.02, 0.0, {'reason': 'insufficient_energy'})

        # Вода восстанавливает жажду + немного энергии/здоровья
        thirst_reduction = 0.55
        agent.thirst = max(0.0, agent.thirst - thirst_reduction)
        agent.energy = min(1.0, agent.energy + 0.03)
        agent.health = min(1.0, agent.health + 0.01)
        agent.energy = max(0.0, agent.energy - energy_cost)

        return ActionResult(
            'drink',
            True,
            0.03,
            energy_cost,
            {'thirst_reduction': thirst_reduction}
        )

    @staticmethod
    def execute_communicate(agent: Agent, environment: Environment, other_agents: List[Agent]) -> ActionResult:
        """Коммуникация между агентами.

        Агент выбирает смысл на основе своего состояния/окружения, изобретает или выбирает токен.
        Слушатель пытается интерпретировать токен. При успехе обе стороны усиливают ассоциацию.
        """
        energy_cost = 0.02
        energy_cost *= AgentActions._night_multiplier(environment, agent.position, radius=1)
        energy_cost *= max(0.1, 1.0 + getattr(agent, 'research_bonuses', {}).get('comm_energy_mult', 0.0))
        if agent.energy < energy_cost:
            return ActionResult('communicate', False, -0.05, 0.0, {'reason': 'insufficient_energy'})

        if not other_agents:
            return ActionResult('communicate', False, -0.01, 0.0, {'reason': 'no_listener'})

        listener = random.choice(other_agents)

        # Смысл выбирается из состояния (без захардкоженных слов)
        local_env = agent.perceive(environment)
        meaning = agent.choose_communication_meaning(local_env)
        token = agent.choose_token_for_meaning(meaning, epsilon=0.2)

        # Слушатель интерпретирует
        interpreted = listener.interpret_token(token)
        success = interpreted == meaning

        # Обновляем лексиконы
        agent.update_communication(meaning, token, success=success, lr=0.25)
        listener.update_communication(meaning, token, success=success, lr=0.25)

        # Сохраняем "речь" для UI
        agent.last_utterance = token
        agent.last_intended_meaning = meaning
        listener.last_heard = token

        agent.energy -= energy_cost

        # Небольшая награда за успешную коммуникацию
        reward = 0.05 if success else -0.01
        return ActionResult(
            'communicate',
            success,
            reward,
            energy_cost,
            {
                'listener_id': listener.id,
                'meaning': meaning,
                'token': token,
                'interpreted': interpreted,
            },
        )

    @staticmethod
    def execute_mate(agent: Agent, environment: Environment, other_agents: List[Agent]) -> ActionResult:
        """Попытка зачатия ребёнка (беременность у female)."""
        energy_cost = 0.08
        energy_cost *= AgentActions._night_multiplier(environment, agent.position, radius=1)
        if agent.energy < energy_cost:
            return ActionResult('mate', False, -0.05, 0.0, {'reason': 'insufficient_energy'})

        if not other_agents:
            return ActionResult('mate', False, -0.01, 0.0, {'reason': 'no_partner'})

        partner = random.choice(other_agents)

        # Find female and male in pair
        a_sex = getattr(agent, 'sex', 'unknown')
        p_sex = getattr(partner, 'sex', 'unknown')
        if {a_sex, p_sex} != {'male', 'female'}:
            return ActionResult('mate', False, -0.02, 0.0, {'reason': 'incompatible_sex'})

        female = agent if a_sex == 'female' else partner
        male = agent if a_sex == 'male' else partner

        if getattr(female, 'pregnant', False):
            return ActionResult('mate', False, -0.01, 0.0, {'reason': 'already_pregnant'})

        # Condition thresholds
        if not (male.health > 0.7 and female.health > 0.7 and male.energy > 0.55 and female.energy > 0.55 and male.hunger < 0.6 and female.hunger < 0.6):
            return ActionResult('mate', False, -0.01, 0.0, {'reason': 'bad_condition'})

        # Chance of conception
        # Закон Ферхюльста: вероятность зачатия падает по мере насыщения среды
        logistic_factor = getattr(environment, '_logistic_factor', 1.0)
        conception_chance = (
            0.15 + 0.15 * min(male.genes.social_tendency, female.genes.social_tendency)
        ) * logistic_factor
        success = random.random() < conception_chance

        agent.energy -= energy_cost
        if not success:
            return ActionResult('mate', False, -0.01, energy_cost, {'partner_id': partner.id, 'reason': 'no_conception'})

        setattr(female, 'pregnant', True)
        setattr(female, 'pregnancy_father_id', male.id)
        setattr(female, 'pregnancy_remaining', int(getattr(female, 'gestation_length', 300)))

        return ActionResult(
            'mate',
            True,
            0.05,
            energy_cost,
            {
                'mother_id': female.id,
                'father_id': male.id,
                'pregnancy_remaining': int(getattr(female, 'pregnancy_remaining', 0)),
            },
        )

    @staticmethod
    def execute_care(agent: Agent, environment: Environment, other_agents: List[Agent]) -> ActionResult:
        """Уход за ребёнком: кормление и обучение (передача лексикона)."""
        if not other_agents:
            return ActionResult('care', False, -0.01, 0.0, {'reason': 'no_child'})

        # Choose the nearest child
        children = [a for a in other_agents if getattr(a, 'is_child', lambda: False)()]
        if not children:
            return ActionResult('care', False, -0.01, 0.0, {'reason': 'no_child'})

        child = min(children, key=lambda c: abs(c.position[0] - agent.position[0]) + abs(c.position[1] - agent.position[1]))

        energy_cost = 0.05
        energy_cost *= AgentActions._night_multiplier(environment, agent.position, radius=1)
        if agent.energy < energy_cost:
            return ActionResult('care', False, -0.02, 0.0, {'reason': 'insufficient_energy'})

        # Honest feeding: parent must spend an edible item from inventory
        food_id = None
        best_energy_value = 0.0
        for obj_id in agent.inventory:
            obj = environment.objects.get(obj_id)
            if obj and obj.is_edible():
                ev = obj.get_energy_value()
                if ev > best_energy_value:
                    best_energy_value = ev
                    food_id = obj_id

        if food_id is None:
            return ActionResult('care', False, -0.02, 0.0, {'reason': 'no_food_in_inventory'})

        food_obj = environment.objects.get(food_id)
        if food_obj is None:
            # Inventory desync fallback
            return ActionResult('care', False, -0.02, 0.0, {'reason': 'food_missing'})

        # Consume the food on behalf of child
        agent.remove_from_inventory(food_id)
        environment.remove_object(food_id)

        energy_gain = float(food_obj.get_energy_value())
        child.energy = min(1.0, child.energy + energy_gain)
        child.hunger = max(0.0, child.hunger - energy_gain * 0.8)

        agent.energy = max(0.0, agent.energy - energy_cost)
        agent.hunger = min(1.0, agent.hunger + 0.02)

        # Teaching: reinforce a token->meaning mapping on child
        meaning = agent.choose_communication_meaning(agent.perceive(environment))
        token = agent.choose_token_for_meaning(meaning, epsilon=0.1)
        child.update_communication(meaning, token, success=True, lr=0.35)
        child.last_heard = token
        agent.last_utterance = token
        agent.last_intended_meaning = meaning

        return ActionResult(
            'care',
            True,
            0.03,
            energy_cost,
            {
                'child_id': child.id,
                'token': token,
                'meaning': meaning,
                'food_id': food_id,
                'energy_gain': energy_gain,
            },
        )
    
    @staticmethod
    def execute_combine(agent: Agent, environment: Environment) -> ActionResult:
        """Выполняет комбинирование объектов (с поддержкой целевого крафта)"""
        if (len(agent.inventory) + len(getattr(agent, 'tools', []) or [])) < 2:
            return ActionResult('combine', False, -0.1, 0.0, {'reason': 'insufficient_objects'})

        energy_cost = 0.2
        energy_cost *= AgentActions._night_multiplier(environment, agent.position, radius=1)
        energy_cost *= max(0.2, 1.0 + getattr(agent, 'research_bonuses', {}).get('craft_energy_mult', 0.0))

        if agent.energy < energy_cost:
            return ActionResult('combine', False, -0.2, 0.0, {'reason': 'insufficient_energy'})

        # Навыки агента для проверки рецептов
        agent_skills = {s: agent.skills.get(s) for s in ['gathering', 'crafting', 'hunting', 'cooking', 'communication', 'survival']}

        # Целевой крафт: если указан _craft_target_kind
        target_kind = getattr(agent, '_craft_target_kind', None)
        if target_kind:
            setattr(agent, '_craft_target_kind', None)

        # ── Хелпер: попробовать собрать ингредиенты для конкретного рецепта ──
        def _try_gather_for_recipe(tokens_key):
            inv_ids = list(agent.inventory or [])
            tool_ids = list(getattr(agent, 'tools', []) or [])
            selected_items = []
            selected_obj_ids = []
            selected_tool_ids = []
            for tok in tokens_key:
                tok = str(tok)
                if tok.startswith('obj:'):
                    need = tok.split(':', 1)[1]
                    found = None
                    for oid in inv_ids:
                        o = environment.objects.get(oid)
                        if o and getattr(o, 'type', None) == need:
                            found = oid
                            break
                    if found is None:
                        return None
                    inv_ids.remove(found)
                    selected_items.append(environment.objects[found])
                    selected_obj_ids.append(found)
                elif tok.startswith('tool:'):
                    need = tok.split(':', 1)[1]
                    found = None
                    for tid in tool_ids:
                        t = environment.tools.get(tid)
                        if t and getattr(t, 'kind', None) == need:
                            found = tid
                            break
                    if found is None:
                        return None
                    tool_ids.remove(found)
                    selected_items.append(environment.tools[found])
                    selected_tool_ids.append(found)
            return {'items': selected_items, 'objects': selected_obj_ids, 'tools': selected_tool_ids}

        # ── Хелпер: потратить компоненты ──
        def _consume_components(comp_info):
            for oid in comp_info.get('objects', []):
                try:
                    agent.remove_from_inventory(oid)
                except Exception:
                    pass
            for tid in comp_info.get('tools', []):
                try:
                    if tid in (getattr(agent, 'tools', []) or []):
                        agent.tools.remove(tid)
                except Exception:
                    pass
                try:
                    environment.remove_tool(tid)
                except Exception:
                    pass

        crafted_tool = None
        crafted_components = None
        crafted_object_result = None

        # 1) Целевой крафт по kind
        if target_kind:
            tokens_key = ToolFactory.RECIPE_BY_KIND.get(target_kind)
            if tokens_key:
                comp_info = _try_gather_for_recipe(tokens_key)
                if comp_info:
                    # Сначала проверяем — рецепт создаёт объект?
                    obj_result = ToolFactory._try_create_named_object(comp_info['items'], agent_skills)
                    if obj_result:
                        crafted_object_result = obj_result
                        crafted_components = comp_info
                    else:
                        tool = ToolFactory.create_tool_from_objects(
                            comp_info['items'], agent.id,
                            f"tool_{environment.timestep}_{random.randint(1000, 9999)}",
                            environment.timestep, agent_skills=agent_skills)
                        if tool:
                            crafted_tool = tool
                            crafted_components = comp_info

        # 2) Admin custom recipes
        if crafted_tool is None and crafted_object_result is None:
            try:
                recipes = ToolFactory.get_custom_recipes()
            except Exception:
                recipes = []
            if recipes:
                try:
                    random.shuffle(recipes)
                except Exception:
                    pass
                for r in recipes:
                    try:
                        comps = r.get('components') or []
                        if not isinstance(comps, list) or len(comps) < 2:
                            continue
                        comp_info = _try_gather_for_recipe(tuple(sorted(str(c) for c in comps)))
                        if not comp_info:
                            continue
                        tool = ToolFactory.create_tool_from_objects(
                            comp_info['items'], agent.id,
                            f"tool_{environment.timestep}_{random.randint(1000, 9999)}",
                            environment.timestep, agent_skills=agent_skills)
                        if tool:
                            crafted_tool = tool
                            crafted_components = comp_info
                            break
                    except Exception:
                        continue

        # 3) Перебор NAMED_RECIPES (автоматический поиск подходящего)
        if crafted_tool is None and crafted_object_result is None:
            recipe_keys = list(ToolFactory.NAMED_RECIPES.keys())
            random.shuffle(recipe_keys)
            for tokens_key in recipe_keys:
                comp_info = _try_gather_for_recipe(tokens_key)
                if not comp_info:
                    continue
                # Объектный рецепт?
                obj_result = ToolFactory._try_create_named_object(comp_info['items'], agent_skills)
                if obj_result:
                    crafted_object_result = obj_result
                    crafted_components = comp_info
                    break
                # Инструмент?
                tool = ToolFactory.create_tool_from_objects(
                    comp_info['items'], agent.id,
                    f"tool_{environment.timestep}_{random.randint(1000, 9999)}",
                    environment.timestep, agent_skills=agent_skills)
                if tool:
                    crafted_tool = tool
                    crafted_components = comp_info
                    break

        # 4) Fallback: random 2 objects (legacy emergent tools)
        obj_ids = None
        if crafted_tool is None and crafted_object_result is None and len(agent.inventory) >= 2:
            obj_ids = random.sample(agent.inventory, 2)
            obj1 = environment.objects.get(obj_ids[0])
            obj2 = environment.objects.get(obj_ids[1])
            if obj1 and obj2:
                tool = ToolFactory.create_tool_from_objects(
                    [obj1, obj2], agent.id,
                    f"tool_{environment.timestep}_{random.randint(1000, 9999)}",
                    environment.timestep, agent_skills=agent_skills)
                if tool:
                    crafted_tool = tool

        # ── Обработка результата: рецепт создающий объект ──
        if crafted_object_result and crafted_components:
            _consume_components(crafted_components)
            agent.energy -= energy_cost
            ts = getattr(environment, 'timestep', 0)
            result_type = crafted_object_result['object_type']
            result_kind = crafted_object_result['kind']

            if crafted_object_result.get('type') == 'placed':
                # Размещаемый объект (печь, горн и т.п.)
                placed_id = f"{result_type}_{agent.id}_{ts}"
                placed_obj = ObjectFactory.create_object(result_type, agent.position, placed_id, ts)
                setattr(placed_obj, 'placed_by', agent.id)
                setattr(placed_obj, 'permanent', True)
                environment.add_object(placed_obj)
                try:
                    agent.life_log.add(ts, 'craft', f'Построил {result_kind} 🏗️', icon='🏗️')
                except Exception:
                    pass
                return ActionResult('combine', True, 1.5, energy_cost,
                                    {'placed_id': placed_id, 'kind': result_kind})
            else:
                # Создаёт объект в инвентарь (rope, metal_ingot и т.п.)
                qty = crafted_object_result.get('quantity', 1)
                new_id = f"{result_type}_{ts}_{random.randint(1000, 9999)}"
                new_obj = ObjectFactory.create_object(result_type, agent.position, new_id, ts)
                new_obj.quantity = qty
                environment.objects[new_id] = new_obj
                if len(agent.inventory) < getattr(agent, 'inventory_capacity', 5):
                    agent.add_to_inventory(new_id)
                try:
                    agent.life_log.add(ts, 'craft', f'Создал {result_kind} ⚒️', icon='⚒️')
                except Exception:
                    pass
                return ActionResult('combine', True, 0.8, energy_cost,
                                    {'object_id': new_id, 'kind': result_kind})

        # ── Нет результата ──
        if crafted_tool is None:
            agent.energy -= energy_cost * 0.5
            return ActionResult('combine', False, -0.1, energy_cost * 0.5,
                                {'reason': 'ineffective_combination'})

        # ── Успешное создание инструмента ──
        if crafted_components:
            _consume_components(crafted_components)
        elif obj_ids:
            agent.remove_from_inventory(obj_ids[0])
            agent.remove_from_inventory(obj_ids[1])

        agent.add_tool(crafted_tool.id)
        environment.add_tool(crafted_tool)

        # Пассивные бонусы (leather_bag → inventory_bonus и т.п.)
        recipe_key = ToolFactory.RECIPE_BY_KIND.get(crafted_tool.kind)
        if recipe_key:
            recipe = ToolFactory.NAMED_RECIPES.get(recipe_key, {})
            passive = recipe.get('passive_bonus', {})
            if passive:
                for bk, bv in passive.items():
                    if bk == 'inventory_bonus':
                        agent.inventory_capacity = getattr(agent, 'inventory_capacity', 5) + int(bv)
                    elif bk == 'damage_reduction':
                        _rb = getattr(agent, 'research_bonuses', {})
                        _rb['damage_reduction'] = _rb.get('damage_reduction', 0.0) + float(bv)
                        agent.research_bonuses = _rb

        discovery_type = environment.tool_library.register_tool(crafted_tool)
        if discovery_type == 'new_discovery':
            reward = 1.0
            if crafted_tool.id not in agent.discoveries_made:
                agent.discoveries_made.append(crafted_tool.id)
        else:
            reward = 0.2

        agent.energy -= energy_cost

        ts = getattr(environment, 'timestep', 0)
        kind_label = crafted_tool.kind or crafted_tool.get_tool_type()
        try:
            agent.life_log.add(ts, 'craft', f'Создал {kind_label} 🔨', icon='🔨')
        except Exception:
            pass

        return ActionResult(
            'combine', True, reward, energy_cost,
            {'tool_id': crafted_tool.id, 'tool_type': crafted_tool.get_tool_type(),
             'kind': crafted_tool.kind, 'discovery_type': discovery_type}
        )
    
    @staticmethod
    def execute_attack(agent: Agent, environment: Environment) -> ActionResult:
        """Выполняет атаку (охоту)"""
        energy_cost = 0.15
        energy_cost *= AgentActions._night_multiplier(environment, agent.position, radius=1)
        
        if agent.energy < energy_cost:
            return ActionResult('attack', False, -0.2, 0.0, {'reason': 'insufficient_energy'})
        
        # Получаем локальную среду
        local_env = environment.get_local_environment(agent.position, agent.perception_radius)
        
        # Поиск животных в локальной среде
        animals = [obj for obj in local_env.get('objects', []) if obj.type == 'animal']
        
        if not animals:
            return ActionResult('attack', False, -0.1, energy_cost, {'reason': 'no_prey_found'})
        
        # Выбор цели
        target = random.choice(animals)
        
        # Проверка наличия подходящего инструмента
        tool_id = agent.get_best_tool_for_action('attack', environment)
        tool = environment.tools.get(tool_id) if tool_id else None
        
        # Расчет вероятности успеха
        base_success_prob = 0.3
        if tool:
            base_success_prob *= tool.calculate_effectiveness('attack')
        
        success_prob = (base_success_prob + agent.genes.strength * 0.2) \
            * AgentActions._performance_mult(agent)

        _hit = random.random() < success_prob
        if _hit:
            # Успешная охота
            environment.remove_object(target.id)
            agent.add_to_inventory(target.id)

            # Дроп кожи (leather) при охоте
            try:
                ts = getattr(environment, 'timestep', 0)
                leather_id = f"leather_{ts}_{random.randint(1000, 9999)}"
                leather_drop = ObjectFactory.create_object('leather', agent.position, leather_id, ts)
                leather_drop.quantity = 1
                environment.objects[leather_id] = leather_drop
                if len(agent.inventory) < getattr(agent, 'inventory_capacity', 5):
                    agent.add_to_inventory(leather_id)
            except Exception:
                pass

            # Использование инструмента
            if tool:
                tool.use('attack')
                if tool.is_broken():
                    agent.remove_tool(tool.id)
                    environment.remove_tool(tool.id)

            reward = 1.5
            result_data = {'success': True, 'prey_id': target.id, 'tool_used': tool_id}
        else:
            # Неудачная охота
            reward = -0.3
            result_data = {'success': False, 'prey_id': target.id, 'tool_used': tool_id}

        agent.energy -= energy_cost

        return ActionResult('attack', _hit, reward, energy_cost, result_data)
    
    @staticmethod
    def execute_break(agent: Agent, environment: Environment) -> ActionResult:
        """Выполняет разрушение объектов (добыча ресурсов)"""
        energy_cost = 0.12
        energy_cost *= AgentActions._night_multiplier(environment, agent.position, radius=1)
        
        if agent.energy < energy_cost:
            return ActionResult('break', False, -0.2, 0.0, {'reason': 'insufficient_energy'})
        
        # Поиск объектов, которые можно разбить
        local_env = environment.get_local_environment(agent.position, agent.perception_radius)
        breakable_objects = [
            obj for obj in local_env.get('objects', []) 
            if obj.type in ['stone', 'wood', 'bone'] and obj.hardness > 0.5
        ]
        
        if not breakable_objects:
            return ActionResult('break', False, -0.1, energy_cost, {'reason': 'no_breakable_objects'})
        
        # Выбор цели
        target = random.choice(breakable_objects)
        
        # Проверка наличия подходящего инструмента
        tool_id = agent.get_best_tool_for_action('break', environment)
        tool = environment.tools.get(tool_id) if tool_id else None
        
        # Расчет вероятности успеха
        base_success_prob = 0.4
        if tool:
            base_success_prob *= tool.calculate_effectiveness('break')
        
        success_prob = base_success_prob + agent.genes.strength * 0.15

        _hit = random.random() < success_prob
        if _hit:
            # Успешное разрушение
            new_resources = AgentActions._create_resources_from_break(target, environment)

            environment.remove_object(target.id)

            # Использование инструмента
            if tool:
                tool.use('break')
                if tool.is_broken():
                    agent.remove_tool(tool.id)
                    environment.remove_tool(tool.id)

            reward = len(new_resources) * 0.3
            result_data = {
                'success': True,
                'broken_object': target.id,
                'new_resources': new_resources,
                'tool_used': tool_id
            }
        else:
            # Неудачная попытка
            reward = -0.2
            result_data = {'success': False, 'target_id': target.id, 'tool_used': tool_id}

        agent.energy -= energy_cost

        return ActionResult('break', _hit, reward, energy_cost, result_data)
    
    @staticmethod
    def _create_resources_from_break(broken_obj: Object, environment: Environment) -> List[str]:
        """Создает новые ресурсы из разрушенного объекта"""
        new_resources = []
        
        if broken_obj.type == 'stone':
            # Камень раскалывается на более мелкие камни
            for i in range(2):
                new_obj = ObjectFactory.create_object(
                    'stone', broken_obj.position, 
                    f"fragment_{environment.timestep}_{i}", 
                    environment.timestep, environment.season
                )
                new_obj.hardness *= 0.8  # Осколки менее твердые
                new_obj.weight *= 0.5
                environment.add_object(new_obj)
                new_resources.append(new_obj.id)
        
        elif broken_obj.type == 'wood':
            # Дерево раскалывается на палки и волокна
            for i in range(2):
                obj_type = random.choice(['wood', 'fiber'])
                new_obj = ObjectFactory.create_object(
                    obj_type, broken_obj.position,
                    f"fragment_{environment.timestep}_{i}",
                    environment.timestep, environment.season
                )
                environment.add_object(new_obj)
                new_resources.append(new_obj.id)
        
        elif broken_obj.type == 'bone':
            # Кость раскалывается на осколки костей
            for i in range(2):
                new_obj = ObjectFactory.create_object(
                    'bone', broken_obj.position,
                    f"fragment_{environment.timestep}_{i}",
                    environment.timestep, environment.season
                )
                new_obj.hardness *= 0.7
                new_obj.weight *= 0.4
                environment.add_object(new_obj)
                new_resources.append(new_obj.id)
        
        return new_resources
    
    @staticmethod
    def execute_rest(agent: Agent, environment: Environment) -> ActionResult:
        """Выполняет отдых"""
        # Восстановление энергии
        energy_recovery = 0.1 * (1 + agent.genes.strength * 0.3)

        # Ночью отдых эффективнее
        night_mult = AgentActions._night_multiplier(environment, agent.position, radius=1)
        if night_mult > 1.0:
            energy_recovery *= 1.2
        agent.energy = min(1.0, agent.energy + energy_recovery)
        
        # Небольшое восстановление здоровья
        if agent.health < 1.0:
            agent.health = min(1.0, agent.health + 0.01)
        
        # Награда за отдых (особенно при низкой энергии)
        reward = energy_recovery * 0.5
        
        return ActionResult(
            'rest',
            True,
            reward,
            0.0,
            {'energy_recovery': energy_recovery}
        )

    @staticmethod
    def execute_sleep(agent: Agent, environment: Environment) -> ActionResult:
        """Сон (лучше отдыха): восстанавливает энергию/здоровье и снижает сонливость."""
        # Во сне агент почти не тратит энергии, но "теряет время" через reward
        sleepiness_before = float(getattr(agent, 'sleepiness', 0.0))

        # Ночью сон эффективнее
        is_daytime = getattr(environment, 'is_daytime', True)
        eff = 1.25 if not is_daytime else 1.0

        energy_recovery = 0.16 * eff * (1 + agent.genes.strength * 0.2)
        agent.energy = min(1.0, agent.energy + energy_recovery)

        # Здоровье восстанавливается лучше чем при rest
        agent.health = min(1.0, agent.health + 0.02 * eff)

        # Сонливость снижается (sleep_efficiency увеличивает восстановление)
        _sleep_eff = 1.0 + getattr(agent, 'research_bonuses', {}).get('sleep_efficiency', 0.0)
        agent.sleepiness = max(0.0, sleepiness_before - 0.45 * eff * _sleep_eff)

        reward = 0.04
        return ActionResult(
            'sleep',
            True,
            reward,
            0.0,
            {
                'energy_recovery': energy_recovery,
                'sleepiness_before': sleepiness_before,
                'sleepiness_after': float(agent.sleepiness),
            }
        )


    @staticmethod
    def execute_light_fire(agent: Agent, environment: Environment) -> ActionResult:
        """Разводит костёр из 3 дерева на текущей позиции."""
        energy_cost = 0.15
        if agent.energy < energy_cost:
            return ActionResult('light_fire', False, -0.1, 0.0, {'reason': 'insufficient_energy'})

        # Нельзя на воде
        if environment.is_water(agent.position):
            return ActionResult('light_fire', False, -0.1, 0.0, {'reason': 'on_water'})

        # Уже есть костёр здесь?
        cell = environment.get_objects_at_position(agent.position)
        if any(o.type == 'campfire' for o in cell):
            return ActionResult('light_fire', False, -0.05, 0.0, {'reason': 'fire_exists'})

        # Найти 3 дерева в инвентаре
        wood_ids = [
            oid for oid in agent.inventory
            if (o := environment.objects.get(oid)) and o.type == 'wood'
        ]
        if len(wood_ids) < 3:
            return ActionResult('light_fire', False, -0.1, 0.0, {'reason': 'not_enough_wood'})

        agent.energy -= energy_cost

        # Убрать 3 дерева из инвентаря
        for oid in wood_ids[:3]:
            agent.remove_from_inventory(oid)
            environment.remove_object(oid)

        # Создать костёр
        ts = getattr(environment, 'timestep', 0)
        fire_id = f"campfire_{agent.id}_{ts}"
        fire = ObjectFactory.create_object('campfire', agent.position, fire_id, ts)
        setattr(fire, 'fuel_ticks', 500)
        environment.add_object(fire)

        # Дневник
        try:
            agent.life_log.add(ts, 'craft', 'Развёл костёр 🔥', icon='🔥')
        except Exception:
            pass

        return ActionResult('light_fire', True, 1.5, energy_cost, {'fire_id': fire_id})

    @staticmethod
    def execute_plant_berry(agent: Agent, environment: Environment) -> ActionResult:
        """Сажает ягодный куст из ягоды в инвентаре."""
        energy_cost = 0.05
        if agent.energy < energy_cost:
            return ActionResult('plant_berry', False, -0.05, 0.0, {'reason': 'insufficient_energy'})

        if environment.is_water(agent.position):
            return ActionResult('plant_berry', False, -0.05, 0.0, {'reason': 'on_water'})

        cell = environment.get_objects_at_position(agent.position)
        if any(o.type == 'berry_bush' for o in cell):
            return ActionResult('plant_berry', False, -0.02, 0.0, {'reason': 'bush_exists'})

        berry_ids = [
            oid for oid in agent.inventory
            if (o := environment.objects.get(oid)) and o.type == 'berry'
        ]
        if not berry_ids:
            return ActionResult('plant_berry', False, -0.05, 0.0, {'reason': 'no_berry'})

        agent.energy -= energy_cost
        # Из стака ягод тратим 1 штуку, не удаляя весь объект
        berry_obj = environment.objects.get(berry_ids[0])
        if berry_obj and berry_obj.quantity > 1:
            berry_obj.quantity -= 1
        else:
            agent.remove_from_inventory(berry_ids[0])
            environment.remove_object(berry_ids[0])

        ts = getattr(environment, 'timestep', 0)
        bush_id = f"berry_bush_{agent.id}_{ts}"
        bush = ObjectFactory.create_object('berry_bush', agent.position, bush_id, ts)
        setattr(bush, 'planted_at', ts)
        setattr(bush, 'ripe', False)
        environment.add_object(bush)

        try:
            agent.life_log.add(ts, 'craft', 'Посадил ягодный куст 🌿', icon='🌿')
        except Exception:
            pass

        return ActionResult('plant_berry', True, 0.8, energy_cost, {'bush_id': bush_id})

    @staticmethod
    def execute_build_shelter(agent: Agent, environment: Environment) -> ActionResult:
        """Строит убежище из 5 дерева + 3 камня. Требует crafting lv5 (>= 0.4)."""
        import uuid as _uuid
        energy_cost = 0.25
        energy_cost *= max(0.2, 1.0 + getattr(agent, 'research_bonuses', {}).get('craft_energy_mult', 0.0))
        if agent.energy < energy_cost:
            return ActionResult('build_shelter', False, -0.1, 0.0, {'reason': 'insufficient_energy'})
        if environment.is_water(agent.position):
            return ActionResult('build_shelter', False, -0.1, 0.0, {'reason': 'on_water'})
        cell = environment.get_objects_at_position(agent.position)
        if any(o.type == 'shelter' for o in cell):
            return ActionResult('build_shelter', False, -0.05, 0.0, {'reason': 'shelter_exists'})

        wood_ids  = [oid for oid in agent.inventory
                     if (o := environment.objects.get(oid)) and o.type == 'wood']
        stone_ids = [oid for oid in agent.inventory
                     if (o := environment.objects.get(oid)) and o.type == 'stone']
        if len(wood_ids) < 5 or len(stone_ids) < 3:
            return ActionResult('build_shelter', False, -0.1, 0.0, {'reason': 'not_enough_materials'})

        agent.energy -= energy_cost
        for oid in wood_ids[:5]:
            agent.remove_from_inventory(oid)
            environment.remove_object(oid)
        for oid in stone_ids[:3]:
            agent.remove_from_inventory(oid)
            environment.remove_object(oid)

        ts = getattr(environment, 'timestep', 0)
        sh_id = f"shelter_{agent.id}_{ts}"
        shelter = ObjectFactory.create_object('shelter', agent.position, sh_id, ts)
        setattr(shelter, 'shelter_owner_id', agent.id)
        setattr(shelter, 'shelter_owner_name', getattr(agent, 'display_name', agent.id))
        setattr(shelter, 'permanent', True)
        environment.add_object(shelter)

        try:
            agent.life_log.add(ts, 'craft', 'Построил убежище 🏠', icon='🏠')
        except Exception:
            pass
        return ActionResult('build_shelter', True, 3.0, energy_cost, {'shelter_id': sh_id})

    @staticmethod
    def execute_build(agent: Agent, environment: Environment) -> ActionResult:
        """Строит здание из BUILDING_RECIPES. Тип здания берётся из agent._build_target."""
        import uuid as _uuid

        target = getattr(agent, '_build_target', None)

        # Если цель не задана — выбираем первое доступное по материалам
        if not target:
            for btype, recipe in BUILDING_RECIPES.items():
                skill_name, skill_min = recipe['skill']
                if agent.skills.get(skill_name) < skill_min:
                    continue
                # Проверяем материалы
                ok = True
                for mat, cnt in recipe['materials'].items():
                    have = sum(1 for oid in agent.inventory
                               if (o := environment.objects.get(oid)) and o.type == mat)
                    if have < cnt:
                        ok = False
                        break
                if ok:
                    target = btype
                    break

        if not target or target not in BUILDING_RECIPES:
            return ActionResult('build', False, -0.05, 0.0, {'reason': 'no_target'})

        recipe = BUILDING_RECIPES[target]
        skill_name, skill_min = recipe['skill']

        # Проверка скилла
        if agent.skills.get(skill_name) < skill_min:
            return ActionResult('build', False, -0.05, 0.0, {'reason': 'insufficient_skill'})

        # Проверка энергии
        energy_cost = recipe['energy_cost']
        energy_cost *= max(0.2, 1.0 + getattr(agent, 'research_bonuses', {}).get('craft_energy_mult', 0.0))
        if agent.energy < energy_cost:
            return ActionResult('build', False, -0.1, 0.0, {'reason': 'insufficient_energy'})

        # Нельзя строить на воде
        if environment.is_water(agent.position):
            return ActionResult('build', False, -0.1, 0.0, {'reason': 'on_water'})

        # Нельзя ставить здание того же типа в той же клетке
        cell = environment.get_objects_at_position(agent.position)
        if any(o.type == target for o in cell):
            return ActionResult('build', False, -0.05, 0.0, {'reason': 'building_exists'})

        # Проверка и списание материалов
        mat_ids = {}  # mat_type -> [obj_ids]
        for mat, cnt in recipe['materials'].items():
            ids = [oid for oid in agent.inventory
                   if (o := environment.objects.get(oid)) and o.type == mat]
            if len(ids) < cnt:
                return ActionResult('build', False, -0.1, 0.0, {'reason': 'not_enough_materials'})
            mat_ids[mat] = ids[:cnt]

        # Списать энергию и материалы
        agent.energy -= energy_cost
        for mat, ids in mat_ids.items():
            for oid in ids:
                agent.remove_from_inventory(oid)
                environment.remove_object(oid)

        # Создать здание
        ts = getattr(environment, 'timestep', 0)
        b_id = f"{target}_{agent.id}_{ts}"
        building = ObjectFactory.create_object(target, agent.position, b_id, ts)
        setattr(building, 'building_owner_id', agent.id)
        setattr(building, 'building_owner_name', getattr(agent, 'display_name', agent.id))
        setattr(building, 'building_type', target)
        setattr(building, 'building_level', 1)
        setattr(building, 'permanent', True)
        # Для огорода: счётчик производства
        if target == 'garden':
            setattr(building, 'produce_timer', 0)
            setattr(building, 'stored_produce', 0)
        # Для сушилки: счётчик
        if target == 'drying_rack':
            setattr(building, 'dry_timer', 0)
        # Для торгового поста: хранилище
        if target == 'trading_post':
            setattr(building, 'stored_items', [])
        environment.add_object(building)

        emoji = recipe['emoji']
        label = recipe['label']
        try:
            agent.life_log.add(ts, 'craft', f'Построил {label} {emoji}', icon=emoji)
        except Exception:
            pass

        # Сброс таргета
        setattr(agent, '_build_target', None)
        setattr(agent, 'pending_build_target', None)

        return ActionResult('build', True, 3.0, energy_cost,
                            {'building_id': b_id, 'building_type': target})

    @staticmethod
    def execute_upgrade(agent: Agent, environment: Environment) -> ActionResult:
        """Улучшает здание до следующего уровня. Стоимость: L2 = 1.5x базы, L3 = 2x базы + metal_ingot."""
        import uuid as _uuid
        import math

        # Ищем здание рядом (радиус 1) принадлежащее агенту
        target_type = getattr(agent, '_upgrade_target', None)
        building = None
        ax, ay = agent.position
        for dx in range(-1, 2):
            for dy in range(-1, 2):
                for obj in environment.get_objects_at_position((ax + dx, ay + dy)):
                    if getattr(obj, 'building_owner_id', None) != agent.id:
                        continue
                    bt = getattr(obj, 'building_type', None)
                    if bt and bt in BUILDING_RECIPES:
                        if target_type and bt != target_type:
                            continue
                        lvl = getattr(obj, 'building_level', 1)
                        if lvl < 3:
                            building = obj
                            break
                if building:
                    break
            if building:
                break

        if not building:
            return ActionResult('upgrade', False, -0.05, 0.0, {'reason': 'no_upgradeable_building'})

        bt = getattr(building, 'building_type')
        lvl = getattr(building, 'building_level', 1)
        recipe = BUILDING_RECIPES[bt]
        next_lvl = lvl + 1

        # Множитель материалов по уровню
        mat_mult = {2: 1.5, 3: 2.0}.get(next_lvl, 1.5)
        energy_cost = recipe['energy_cost'] * mat_mult
        energy_cost *= max(0.2, 1.0 + getattr(agent, 'research_bonuses', {}).get('craft_energy_mult', 0.0))

        if agent.energy < energy_cost:
            return ActionResult('upgrade', False, -0.1, 0.0, {'reason': 'insufficient_energy'})

        # Проверка скилла (на 0.1 выше за уровень)
        skill_name, skill_min = recipe['skill']
        required_skill = skill_min + (next_lvl - 1) * 0.1
        if agent.skills.get(skill_name) < required_skill:
            return ActionResult('upgrade', False, -0.05, 0.0, {'reason': 'insufficient_skill'})

        # Проверка материалов
        mat_ids = {}
        for mat, cnt in recipe['materials'].items():
            need = math.ceil(cnt * mat_mult)
            ids = [oid for oid in agent.inventory
                   if (o := environment.objects.get(oid)) and o.type == mat]
            if len(ids) < need:
                return ActionResult('upgrade', False, -0.1, 0.0,
                                    {'reason': 'not_enough_materials'})
            mat_ids[mat] = ids[:need]

        # L3 дополнительно требует metal_ingot
        if next_lvl == 3:
            ingot_ids = [oid for oid in agent.inventory
                         if (o := environment.objects.get(oid)) and o.type == 'metal_ingot']
            if not ingot_ids:
                return ActionResult('upgrade', False, -0.1, 0.0,
                                    {'reason': 'need_metal_ingot'})
            mat_ids['metal_ingot'] = ingot_ids[:1]

        # Списать
        agent.energy -= energy_cost
        for mat, ids in mat_ids.items():
            for oid in ids:
                agent.remove_from_inventory(oid)
                environment.remove_object(oid)

        # Повысить уровень
        setattr(building, 'building_level', next_lvl)
        # Восстановить прочность при апгрейде
        building.durability = min(1.0, building.durability + 0.5)

        emoji = recipe['emoji']
        label = recipe['label']
        ts = getattr(environment, 'timestep', 0)
        try:
            agent.life_log.add(ts, 'craft',
                               f'Улучшил {label} до ур.{next_lvl} {emoji}⬆', icon='⬆️')
        except Exception:
            pass

        setattr(agent, '_upgrade_target', None)
        return ActionResult('upgrade', True, 4.0, energy_cost,
                            {'building_type': bt, 'new_level': next_lvl})

    @staticmethod
    def execute_repair_building(agent: Agent, environment: Environment) -> ActionResult:
        """Ремонтирует ближайшее повреждённое здание. Нужно 2 wood или 2 stone."""
        building = None
        ax, ay = agent.position
        for dx in range(-1, 2):
            for dy in range(-1, 2):
                for obj in environment.get_objects_at_position((ax + dx, ay + dy)):
                    if getattr(obj, 'building_owner_id', None) != agent.id:
                        continue
                    bt = getattr(obj, 'building_type', None)
                    if bt and bt in BUILDING_RECIPES and getattr(obj, 'durability', 1.0) < 0.7:
                        building = obj
                        break
                if building:
                    break
            if building:
                break

        if not building:
            return ActionResult('repair_building', False, -0.05, 0.0,
                                {'reason': 'no_damaged_building'})

        # Нужно 2 wood или 2 stone
        wood_ids = [oid for oid in agent.inventory
                    if (o := environment.objects.get(oid)) and o.type == 'wood']
        stone_ids = [oid for oid in agent.inventory
                     if (o := environment.objects.get(oid)) and o.type == 'stone']
        if len(wood_ids) >= 2:
            use_ids = wood_ids[:2]
        elif len(stone_ids) >= 2:
            use_ids = stone_ids[:2]
        else:
            return ActionResult('repair_building', False, -0.05, 0.0,
                                {'reason': 'no_repair_materials'})

        energy_cost = 0.08
        if agent.energy < energy_cost:
            return ActionResult('repair_building', False, -0.05, 0.0,
                                {'reason': 'insufficient_energy'})

        agent.energy -= energy_cost
        for oid in use_ids:
            agent.remove_from_inventory(oid)
            environment.remove_object(oid)

        # Восстановить 0.3 прочности
        building.durability = min(1.0, building.durability + 0.3)

        bt = getattr(building, 'building_type', '?')
        label = BUILDING_RECIPES.get(bt, {}).get('label', bt)
        ts = getattr(environment, 'timestep', 0)
        try:
            agent.life_log.add(ts, 'craft',
                               f'Починил {label} 🔧', icon='🔧')
        except Exception:
            pass

        return ActionResult('repair_building', True, 1.0, energy_cost,
                            {'building_type': bt, 'durability': building.durability})

    # ── Торговля (Trading Post) ─────────────────────────────────────────

    @staticmethod
    def execute_deposit(agent: Agent, environment: Environment) -> ActionResult:
        """Кладёт предмет из инвентаря на торговый пост. Макс 10 предметов на посте."""
        energy_cost = 0.02
        if agent.energy < energy_cost:
            return ActionResult('deposit', False, -0.05, 0.0, {'reason': 'insufficient_energy'})

        # Найти свой торговый пост рядом (радиус 2)
        post = None
        ax, ay = agent.position
        for dx in range(-2, 3):
            for dy in range(-2, 3):
                for obj in environment.get_objects_at_position((ax + dx, ay + dy)):
                    if (getattr(obj, 'building_type', None) == 'trading_post'
                            and getattr(obj, 'building_owner_id', None) == agent.id):
                        post = obj
                        break
                if post:
                    break
            if post:
                break

        if not post:
            return ActionResult('deposit', False, -0.05, 0.0, {'reason': 'no_trading_post'})

        stored = getattr(post, 'stored_items', [])
        if len(stored) >= 10:
            return ActionResult('deposit', False, -0.02, 0.0, {'reason': 'post_full'})

        # Выбираем предмет для депозита: дубликаты или излишки
        # Приоритет: cooked_food > plant > berry > stone > wood > herb > fiber
        _deposit_priority = ['cooked_food', 'plant', 'berry', 'stone', 'wood', 'herb', 'fiber',
                             'bone', 'clay', 'mushroom', 'fish', 'leather', 'rope', 'ore', 'metal_ingot']
        deposited = None
        for ptype in _deposit_priority:
            ids = [oid for oid in agent.inventory
                   if (o := environment.objects.get(oid)) and o.type == ptype]
            if len(ids) >= 2:  # только если есть дубликат
                deposited = ids[0]
                break

        if not deposited:
            return ActionResult('deposit', False, -0.02, 0.0, {'reason': 'nothing_to_deposit'})

        obj = environment.objects.get(deposited)
        obj_type = obj.type if obj else '?'

        agent.remove_from_inventory(deposited)
        stored.append(deposited)
        setattr(post, 'stored_items', stored)
        # Перемещаем объект на позицию поста (но не в grid — он "внутри" поста)
        if obj:
            obj.position = post.position

        agent.energy -= energy_cost

        ts = getattr(environment, 'timestep', 0)
        try:
            agent.life_log.add(ts, 'trade', f'Положил {obj_type} на торговый пост', icon='🏪')
        except Exception:
            pass

        return ActionResult('deposit', True, 0.3, energy_cost,
                            {'item_type': obj_type, 'post_items': len(stored)})

    @staticmethod
    def execute_collect_trade(agent: Agent, environment: Environment) -> ActionResult:
        """Берёт предмет с чужого торгового поста. Повышает trust с владельцем."""
        energy_cost = 0.02
        if agent.energy < energy_cost:
            return ActionResult('collect_trade', False, -0.05, 0.0, {'reason': 'insufficient_energy'})

        if len(agent.inventory) >= agent.inventory_capacity:
            return ActionResult('collect_trade', False, -0.02, 0.0, {'reason': 'inventory_full'})

        # Найти чужой торговый пост рядом с предметами
        post = None
        ax, ay = agent.position
        for dx in range(-2, 3):
            for dy in range(-2, 3):
                for obj in environment.get_objects_at_position((ax + dx, ay + dy)):
                    if (getattr(obj, 'building_type', None) == 'trading_post'
                            and getattr(obj, 'building_owner_id', None) != agent.id
                            and len(getattr(obj, 'stored_items', [])) > 0):
                        post = obj
                        break
                if post:
                    break
            if post:
                break

        if not post:
            return ActionResult('collect_trade', False, -0.02, 0.0, {'reason': 'no_trade_available'})

        stored = getattr(post, 'stored_items', [])
        item_id = stored.pop(0)  # FIFO
        setattr(post, 'stored_items', stored)

        item_obj = environment.objects.get(item_id)
        item_type = item_obj.type if item_obj else '?'

        agent.add_to_inventory(item_id)
        if item_obj:
            item_obj.position = agent.position

        agent.energy -= energy_cost

        # Повышаем trust между агентом и владельцем поста
        owner_id = getattr(post, 'building_owner_id', None)
        if owner_id and hasattr(agent, 'social'):
            agent.social.add_interaction(owner_id, 0.08)

        ts = getattr(environment, 'timestep', 0)
        try:
            agent.life_log.add(ts, 'trade',
                               f'Взял {item_type} с торгового поста', icon='🤝')
        except Exception:
            pass

        return ActionResult('collect_trade', True, 0.4, energy_cost,
                            {'item_type': item_type, 'owner_id': owner_id})

    @staticmethod
    def execute_treat(agent: Agent, environment: Environment) -> ActionResult:
        """Лечит себя травами (plant или berry). Требует survival lv4 (>= 0.3)."""
        energy_cost = 0.03
        if agent.energy < energy_cost:
            return ActionResult('treat', False, -0.05, 0.0, {'reason': 'insufficient_energy'})

        herb_ids = [oid for oid in agent.inventory
                    if (o := environment.objects.get(oid)) and o.type in ('plant', 'berry')]
        if not herb_ids:
            return ActionResult('treat', False, -0.05, 0.0, {'reason': 'no_herbs'})

        herb = environment.objects.get(herb_ids[0])
        if herb and getattr(herb, 'quantity', 1) > 1:
            herb.quantity -= 1
        else:
            agent.remove_from_inventory(herb_ids[0])
            environment.remove_object(herb_ids[0])

        agent.energy -= energy_cost
        agent.health  = min(1.0, agent.health + 0.15)
        agent.hunger  = min(1.0, agent.hunger + 0.05)

        ts = getattr(environment, 'timestep', 0)
        try:
            agent.life_log.add(ts, 'craft', 'Вылечил себя травами 🌿', icon='🌿')
        except Exception:
            pass
        return ActionResult('treat', True, 0.5, energy_cost, {})

    @staticmethod
    def execute_share(agent: Agent, environment: Environment,
                      other_agents: List[Agent]) -> ActionResult:
        """Поделиться ягодами с соседним агентом."""
        energy_cost = 0.02
        if agent.energy < energy_cost:
            return ActionResult('share', False, -0.05, 0.0, {'reason': 'insufficient_energy'})

        if not other_agents:
            return ActionResult('share', False, 0.0, 0.0, {'reason': 'no_agents_nearby'})

        # Ищем ягоды в инвентаре
        berry_ids = [
            oid for oid in agent.inventory
            if (bo := environment.objects.get(oid)) and bo.type == 'berry' and bo.quantity > 0
        ]
        if not berry_ids:
            return ActionResult('share', False, 0.0, 0.0, {'reason': 'no_berries'})

        berry_obj = environment.objects.get(berry_ids[0])
        if not berry_obj or berry_obj.quantity <= 0:
            return ActionResult('share', False, 0.0, 0.0, {'reason': 'invalid_berry'})

        # Делимся с самым голодным соседом
        target = max(other_agents, key=lambda a: getattr(a, 'hunger', 0.0))

        # Отдаём половину, минимум 1
        share_count = max(1, berry_obj.quantity // 2)

        import uuid as _uuid
        gift_id = f"berry_gift_{_uuid.uuid4().hex[:8]}"
        gift = ObjectFactory.create_object('berry', agent.position, gift_id, berry_obj.created_at)
        gift.quantity = share_count
        environment.objects[gift_id] = gift

        if len(target.inventory) < target.inventory_capacity:
            target.add_to_inventory(gift_id)
        else:
            # Инвентарь полный — просто уменьшаем стак без передачи
            del environment.objects[gift_id]
            return ActionResult('share', False, 0.0, 0.0, {'reason': 'target_inventory_full'})

        berry_obj.quantity -= share_count
        if berry_obj.quantity <= 0:
            agent.remove_from_inventory(berry_ids[0])
            environment.remove_object(berry_ids[0])

        agent.energy -= energy_cost

        ts = getattr(environment, 'timestep', 0)
        try:
            agent.life_log.add(ts, 'social', f'Поделился {share_count} ягодами 🍓', icon='🤝')
        except Exception:
            pass

        return ActionResult('share', True, 0.3, energy_cost,
                            {'target_id': target.id, 'amount': share_count})


    # ══════════════════════════════════════════════════════════════════════
    # Новые действия
    # ══════════════════════════════════════════════════════════════════════

    @staticmethod
    def execute_cook(agent: Agent, environment: Environment) -> ActionResult:
        """Готовит еду у костра/печи. Убирает токсичность, повышает питательность."""
        energy_cost = 0.08
        if agent.energy < energy_cost:
            return ActionResult('cook', False, -0.05, 0.0, {'reason': 'insufficient_energy'})

        # Нужна еда в инвентаре
        food_ids = [oid for oid in agent.inventory
                    if (o := environment.objects.get(oid)) and o.type in ('berry', 'plant', 'mushroom', 'fish')]
        if not food_ids:
            return ActionResult('cook', False, -0.05, 0.0, {'reason': 'no_food_to_cook'})

        # Нужен костёр или печь рядом (радиус 2)
        ax, ay = agent.position
        has_heat = False
        for dx in range(-2, 3):
            for dy in range(-2, 3):
                for o in environment.get_objects_at_position((ax + dx, ay + dy)):
                    if (o.type == 'campfire' and getattr(o, 'fuel_ticks', 0) > 0) or o.type == 'clay_oven':
                        has_heat = True
                        break
                if has_heat:
                    break
            if has_heat:
                break
        if not has_heat:
            return ActionResult('cook', False, -0.05, 0.0, {'reason': 'no_heat_source'})

        # Готовим первый подходящий предмет
        food_obj = environment.objects.get(food_ids[0])
        if not food_obj:
            return ActionResult('cook', False, -0.05, 0.0, {'reason': 'food_gone'})

        if getattr(food_obj, 'quantity', 1) > 1:
            food_obj.quantity -= 1
        else:
            agent.remove_from_inventory(food_ids[0])
            environment.remove_object(food_ids[0])

        # Создаём cooked_food
        ts = getattr(environment, 'timestep', 0)
        cooked_id = f"cooked_{ts}_{random.randint(1000, 9999)}"
        cooked = ObjectFactory.create_object('cooked_food', agent.position, cooked_id, ts)
        cooked.quantity = 1
        environment.objects[cooked_id] = cooked
        if len(agent.inventory) < getattr(agent, 'inventory_capacity', 5):
            agent.add_to_inventory(cooked_id)

        agent.energy -= energy_cost
        try:
            agent.life_log.add(ts, 'craft', 'Приготовил еду 🍳', icon='🍳')
        except Exception:
            pass
        return ActionResult('cook', True, 0.4, energy_cost, {'cooked_id': cooked_id})

    @staticmethod
    def execute_fish(agent: Agent, environment: Environment) -> ActionResult:
        """Ловит рыбу удочкой рядом с водой."""
        energy_cost = 0.12
        if agent.energy < energy_cost:
            return ActionResult('fish', False, -0.05, 0.0, {'reason': 'insufficient_energy'})

        # Проверяем наличие удочки
        has_rod = False
        rod_tool = None
        for tid in (getattr(agent, 'tools', []) or []):
            t = environment.tools.get(tid)
            if t and getattr(t, 'kind', None) == 'fishing_rod' and not t.is_broken():
                has_rod = True
                rod_tool = t
                break
        if not has_rod:
            return ActionResult('fish', False, -0.05, 0.0, {'reason': 'no_fishing_rod'})

        # Нужна вода рядом (радиус 1)
        ax, ay = agent.position
        near_water = False
        for dx in range(-1, 2):
            for dy in range(-1, 2):
                if environment.is_water((ax + dx, ay + dy)):
                    near_water = True
                    break
            if near_water:
                break
        if not near_water:
            return ActionResult('fish', False, -0.05, 0.0, {'reason': 'no_water_nearby'})

        # Шанс успеха зависит от навыка gathering
        skill_val = agent.skills.get('gathering')
        success_chance = 0.4 + 0.4 * min(1.0, skill_val)
        success_chance *= AgentActions._performance_mult(agent)

        agent.energy -= energy_cost
        rod_tool.use('gather')

        if random.random() > success_chance:
            return ActionResult('fish', False, 0.0, energy_cost, {'reason': 'fish_escaped'})

        ts = getattr(environment, 'timestep', 0)
        fish_id = f"fish_{ts}_{random.randint(1000, 9999)}"
        fish_obj = ObjectFactory.create_object('fish', agent.position, fish_id, ts)
        fish_obj.quantity = 1
        environment.objects[fish_id] = fish_obj
        if len(agent.inventory) < getattr(agent, 'inventory_capacity', 5):
            agent.add_to_inventory(fish_id)

        try:
            agent.life_log.add(ts, 'craft', 'Поймал рыбу 🐟', icon='🐟')
        except Exception:
            pass
        return ActionResult('fish', True, 0.5, energy_cost, {'fish_id': fish_id})

    @staticmethod
    def execute_smelt(agent: Agent, environment: Environment) -> ActionResult:
        """Выплавляет metal_ingot из ore рядом с stone_furnace + костром."""
        energy_cost = 0.2
        energy_cost *= max(0.2, 1.0 + getattr(agent, 'research_bonuses', {}).get('craft_energy_mult', 0.0))
        if agent.energy < energy_cost:
            return ActionResult('smelt', False, -0.05, 0.0, {'reason': 'insufficient_energy'})

        # Руда в инвентаре
        ore_ids = [oid for oid in agent.inventory
                   if (o := environment.objects.get(oid)) and o.type == 'ore']
        if not ore_ids:
            return ActionResult('smelt', False, -0.05, 0.0, {'reason': 'no_ore'})

        # stone_furnace + campfire рядом (радиус 2)
        ax, ay = agent.position
        has_furnace = False
        has_fire = False
        for dx in range(-2, 3):
            for dy in range(-2, 3):
                for o in environment.get_objects_at_position((ax + dx, ay + dy)):
                    if o.type == 'stone_furnace':
                        has_furnace = True
                    if o.type == 'campfire' and getattr(o, 'fuel_ticks', 0) > 0:
                        has_fire = True
                if has_furnace and has_fire:
                    break
            if has_furnace and has_fire:
                break
        if not (has_furnace and has_fire):
            return ActionResult('smelt', False, -0.05, 0.0, {'reason': 'no_furnace_or_fire'})

        # Потребляем руду
        ore_obj = environment.objects.get(ore_ids[0])
        if ore_obj and getattr(ore_obj, 'quantity', 1) > 1:
            ore_obj.quantity -= 1
        else:
            agent.remove_from_inventory(ore_ids[0])
            environment.remove_object(ore_ids[0])

        agent.energy -= energy_cost

        ts = getattr(environment, 'timestep', 0)
        ingot_id = f"metal_ingot_{ts}_{random.randint(1000, 9999)}"
        ingot = ObjectFactory.create_object('metal_ingot', agent.position, ingot_id, ts)
        ingot.quantity = 1
        environment.objects[ingot_id] = ingot
        if len(agent.inventory) < getattr(agent, 'inventory_capacity', 5):
            agent.add_to_inventory(ingot_id)

        try:
            agent.life_log.add(ts, 'craft', 'Выплавил слиток ⚙️', icon='⚙️')
        except Exception:
            pass
        return ActionResult('smelt', True, 0.8, energy_cost, {'ingot_id': ingot_id})

    @staticmethod
    def execute_repair(agent: Agent, environment: Environment) -> ActionResult:
        """Ремонтирует самый изношенный инструмент, расходуя 1 материал."""
        energy_cost = 0.1
        if agent.energy < energy_cost:
            return ActionResult('repair', False, -0.05, 0.0, {'reason': 'insufficient_energy'})

        # Ищем инструмент с durability < 60
        worst_tool = None
        worst_dur = 100.0
        for tid in (getattr(agent, 'tools', []) or []):
            t = environment.tools.get(tid)
            if t and t.durability_left < 60 and t.durability_left < worst_dur:
                worst_tool = t
                worst_dur = t.durability_left

        if worst_tool is None:
            return ActionResult('repair', False, -0.05, 0.0, {'reason': 'no_damaged_tools'})

        # Определяем нужный материал: metal → ore/metal_ingot, остальные → wood или stone
        need_types = ['wood', 'stone', 'bone', 'fiber', 'ore', 'metal_ingot']
        mat_id = None
        for oid in agent.inventory:
            o = environment.objects.get(oid)
            if o and o.type in need_types:
                mat_id = oid
                break
        if mat_id is None:
            return ActionResult('repair', False, -0.05, 0.0, {'reason': 'no_repair_material'})

        # Потребляем материал
        mat_obj = environment.objects.get(mat_id)
        if mat_obj and getattr(mat_obj, 'quantity', 1) > 1:
            mat_obj.quantity -= 1
        else:
            agent.remove_from_inventory(mat_id)
            environment.remove_object(mat_id)

        repair_amount = 25.0 + 10.0 * agent.skills.get('crafting')
        worst_tool.repair(repair_amount)
        agent.energy -= energy_cost

        ts = getattr(environment, 'timestep', 0)
        try:
            agent.life_log.add(ts, 'craft', f'Починил {worst_tool.kind or "инструмент"} 🔧', icon='🔧')
        except Exception:
            pass
        return ActionResult('repair', True, 0.3, energy_cost,
                            {'tool_id': worst_tool.id, 'new_durability': worst_tool.durability_left})

    @staticmethod
    def execute_tan_hide(agent: Agent, environment: Environment) -> ActionResult:
        """Выделка кожи: bone + plant у костра → leather."""
        energy_cost = 0.1
        if agent.energy < energy_cost:
            return ActionResult('tan_hide', False, -0.05, 0.0, {'reason': 'insufficient_energy'})

        bone_id = None
        plant_id = None
        for oid in agent.inventory:
            o = environment.objects.get(oid)
            if not o:
                continue
            if o.type == 'bone' and bone_id is None:
                bone_id = oid
            elif o.type == 'plant' and plant_id is None:
                plant_id = oid
        if bone_id is None or plant_id is None:
            return ActionResult('tan_hide', False, -0.05, 0.0, {'reason': 'missing_materials'})

        # Нужен костёр рядом (радиус 2)
        ax, ay = agent.position
        has_fire = False
        for dx in range(-2, 3):
            for dy in range(-2, 3):
                for o in environment.get_objects_at_position((ax + dx, ay + dy)):
                    if o.type == 'campfire' and getattr(o, 'fuel_ticks', 0) > 0:
                        has_fire = True
                        break
                if has_fire:
                    break
            if has_fire:
                break
        if not has_fire:
            return ActionResult('tan_hide', False, -0.05, 0.0, {'reason': 'no_campfire'})

        # Потребляем bone + plant
        for mid in [bone_id, plant_id]:
            m = environment.objects.get(mid)
            if m and getattr(m, 'quantity', 1) > 1:
                m.quantity -= 1
            else:
                agent.remove_from_inventory(mid)
                environment.remove_object(mid)

        agent.energy -= energy_cost

        ts = getattr(environment, 'timestep', 0)
        leather_id = f"leather_{ts}_{random.randint(1000, 9999)}"
        leather = ObjectFactory.create_object('leather', agent.position, leather_id, ts)
        leather.quantity = 1
        environment.objects[leather_id] = leather
        if len(agent.inventory) < getattr(agent, 'inventory_capacity', 5):
            agent.add_to_inventory(leather_id)

        try:
            agent.life_log.add(ts, 'craft', 'Выделал кожу 🧶', icon='🧶')
        except Exception:
            pass
        return ActionResult('tan_hide', True, 0.4, energy_cost, {'leather_id': leather_id})


class ActionExecutor:
    """Основной класс для выполнения действий агента"""

    def __init__(self):
        self.actions = {
            'move': AgentActions.execute_move,
            'gather': AgentActions.execute_gather,
            'consume': AgentActions.execute_consume,
            'drink': AgentActions.execute_drink,
            'communicate': AgentActions.execute_communicate,
            'mate': AgentActions.execute_mate,
            'care': AgentActions.execute_care,
            'combine': AgentActions.execute_combine,
            'attack': AgentActions.execute_attack,
            'break': AgentActions.execute_break,
            'rest': AgentActions.execute_rest,
            'sleep': AgentActions.execute_sleep,
            'light_fire': AgentActions.execute_light_fire,
            'plant_berry': AgentActions.execute_plant_berry,
            'share': AgentActions.execute_share,
            'build_shelter': AgentActions.execute_build_shelter,
            'build': AgentActions.execute_build,
            'upgrade': AgentActions.execute_upgrade,
            'repair_building': AgentActions.execute_repair_building,
            'deposit': AgentActions.execute_deposit,
            'collect_trade': AgentActions.execute_collect_trade,
            'treat': AgentActions.execute_treat,
            'cook': AgentActions.execute_cook,
            'fish': AgentActions.execute_fish,
            'smelt': AgentActions.execute_smelt,
            'repair': AgentActions.execute_repair,
            'tan_hide': AgentActions.execute_tan_hide,
        }
    
    def execute_action(self, agent: Agent, environment: Environment, 
                      action: str, **kwargs) -> ActionResult:
        """Выполняет указанное действие"""
        if action not in self.actions:
            return ActionResult(action, False, -0.5, 0.0, {'reason': 'unknown_action'})
        
        return self.actions[action](agent, environment, **kwargs)
    
    def get_available_actions(self, agent: Agent, environment: Environment) -> List[str]:
        """Возвращает доступные действия для агента"""
        actions = ['rest']

        # Сон всегда доступен (это "углублённый" отдых)
        actions.append('sleep')
        
        if agent.energy > 0.05:
            actions.append('move')
            
            # Проверяем локальную среду
            local_env = environment.get_local_environment(agent.position, agent.perception_radius)
            
            if local_env.get('objects'):
                actions.append('gather')
            
            if agent.inventory:
                actions.append('consume')

            # Вода: если стоим на источнике
            cell_objects = environment.get_objects_at_position(agent.position)
            if any(o.type == 'water' for o in cell_objects):
                actions.append('drink')
            
            if len(agent.inventory) >= 2:
                actions.append('combine')
            
            # Проверяем наличие животных для охоты
            has_animals = any(obj.type == 'animal' for obj in local_env.get('objects', []))
            if has_animals:
                actions.append('attack')
            
            # Проверяем наличие объектов для разрушения
            has_breakable = any(
                obj.type in ['stone', 'wood', 'bone'] and obj.hardness > 0.5
                for obj in local_env.get('objects', [])
            )
            if has_breakable:
                actions.append('break')

            # Костёр: нужно 3 дерева и клетка не на воде и нет уже костра
            if not environment.is_water(agent.position):
                wood_count = sum(
                    1 for oid in agent.inventory
                    if (wo := environment.objects.get(oid)) and wo.type == 'wood'
                )
                cell_here = environment.get_objects_at_position(agent.position)
                if wood_count >= 3 and not any(o.type == 'campfire' for o in cell_here):
                    actions.append('light_fire')

                # Посадить ягоду: нужна хотя бы 1 ягода и нет куста здесь
                has_berry_inv = any(
                    (bo := environment.objects.get(oid)) and bo.type == 'berry'
                    for oid in agent.inventory
                )
                if has_berry_inv and not any(o.type == 'berry_bush' for o in cell_here):
                    actions.append('plant_berry')

        return actions
