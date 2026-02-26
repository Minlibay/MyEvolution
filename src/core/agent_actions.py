"""
Agent actions module - defines agent actions and their execution
"""

from typing import Dict, List, Tuple, Any, Optional
import random

from .agent import Agent
from .objects import Object, ObjectFactory
from .tools import Tool, ToolFactory
from .environment import Environment


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
        return 1.25 if not local_env.get('is_daytime', True) else 1.0
    
    @staticmethod
    def execute_move(agent: Agent, environment: Environment, 
                    target_position: Optional[Tuple[int, int]] = None) -> ActionResult:
        """Выполняет движение агента"""
        # Энергетическая стоимость движения
        energy_cost = 0.05 * (1 + len(agent.inventory) * 0.1)
        energy_cost *= AgentActions._night_multiplier(environment, agent.position, radius=1)
        
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
        
        valid_directions = []
        for dx, dy in directions:
            new_x = agent.position[0] + dx
            new_y = agent.position[1] + dy
            
            if (0 <= new_x < environment.width and 
                0 <= new_y < environment.height):
                valid_directions.append((dx, dy))
        
        if not valid_directions:
            return None
        
        # Выбор направления с учетом исследования
        if random.random() < agent.exploration_rate:
            # Исследование - выбираем случайное направление
            dx, dy = random.choice(valid_directions)
        else:
            # Эксплуатация - движемся к ресурсам
            dx, dy = AgentActions._choose_direction_towards_resources(
                agent, environment, valid_directions
            )
        
        return (agent.position[0] + dx, agent.position[1] + dy)
    
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
        energy_cost = 0.1
        energy_cost *= AgentActions._night_multiplier(environment, agent.position, radius=1)
        
        if agent.energy < energy_cost:
            return ActionResult('gather', False, -0.2, 0.0, {'reason': 'insufficient_energy'})
        
        # Получаем локальную среду
        local_env = environment.get_local_environment(agent.position, agent.perception_radius)
        
        # Поиск объектов в текущей клетке
        cell_objects = environment.get_objects_at_position(agent.position)
        gathered_objects = []
        
        for obj in cell_objects[:]:  # Копия для безопасного удаления
            if len(agent.inventory) >= agent.inventory_capacity:
                break
            
            # Вероятность успешного сбора зависит от веса объекта и силы агента
            success_prob = 1.0 - obj.weight * 0.5 + agent.genes.strength * 0.3
            
            if random.random() < success_prob:
                gathered_objects.append(obj.id)
                agent.add_to_inventory(obj.id)
                environment.detach_object_from_world(obj.id)
                
                # Обновление статистической памяти
                memory_key = f"object_{obj.type}"
                agent.statistical_memory.update_statistic(memory_key, 1.0)
        
        agent.energy -= energy_cost
        
        # Награда за собранные объекты
        reward = len(gathered_objects) * 0.05
        
        return ActionResult(
            'gather',
            len(gathered_objects) > 0,
            reward,
            energy_cost,
            {'gathered_objects': gathered_objects}
        )
    
    @staticmethod
    def execute_consume(agent: Agent, environment: Environment) -> ActionResult:
        """Выполняет потребление объекта"""
        if not agent.inventory:
            return ActionResult('consume', False, -0.1, 0.0, {'reason': 'empty_inventory'})
        
        # Выбираем объект для потребления (самый питательный)
        best_obj_id = None
        best_energy_value = 0.0
        
        for obj_id in agent.inventory:
            obj = environment.objects.get(obj_id)
            if obj and obj.is_edible():
                energy_value = obj.get_energy_value()
                if energy_value > best_energy_value:
                    best_energy_value = energy_value
                    best_obj_id = obj_id
        
        if best_obj_id is None:
            return ActionResult('consume', False, -0.1, 0.0, {'reason': 'no_edible_objects'})
        
        obj = environment.objects[best_obj_id]
        
        # Потребление объекта
        agent.remove_from_inventory(best_obj_id)
        environment.remove_object(best_obj_id)
        
        # Восстановление энергии и уменьшение голода
        energy_gain = obj.get_energy_value()
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
        """Пьёт воду из источника в текущей клетке."""
        cell_objects = environment.get_objects_at_position(agent.position)
        has_water = any(o.type == 'water' for o in cell_objects)
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
        conception_chance = 0.15 + 0.15 * min(male.genes.social_tendency, female.genes.social_tendency)
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
        """Выполняет комбинирование объектов"""
        if len(agent.inventory) < 2:
            return ActionResult('combine', False, -0.1, 0.0, {'reason': 'insufficient_objects'})
        
        energy_cost = 0.2
        energy_cost *= AgentActions._night_multiplier(environment, agent.position, radius=1)
        
        if agent.energy < energy_cost:
            return ActionResult('combine', False, -0.2, 0.0, {'reason': 'insufficient_energy'})
        
        # Выбор двух случайных объектов для комбинирования
        obj_ids = random.sample(agent.inventory, 2)
        obj1 = environment.objects.get(obj_ids[0])
        obj2 = environment.objects.get(obj_ids[1])
        
        if not obj1 or not obj2:
            return ActionResult('combine', False, -0.1, 0.0, {'reason': 'objects_not_found'})
        
        # Попытка создания инструмента
        tool = ToolFactory.create_tool_from_objects(
            [obj1, obj2], agent.id, f"tool_{environment.timestep}_{random.randint(1000, 9999)}", 
            environment.timestep
        )
        
        if tool is None:
            # Неудачная комбинация
            agent.energy -= energy_cost * 0.5
            return ActionResult(
                'combine', 
                False, 
                -0.1, 
                energy_cost * 0.5,
                {'reason': 'ineffective_combination', 'attempted_objects': obj_ids}
            )
        
        # Успешное создание инструмента
        agent.remove_from_inventory(obj_ids[0])
        agent.remove_from_inventory(obj_ids[1])
        agent.add_tool(tool.id)
        environment.add_tool(tool)
        
        # Проверка на новое открытие
        discovery_type = environment.tool_library.register_tool(tool)
        
        if discovery_type == 'new_discovery':
            reward = 1.0
            if tool.id not in agent.discoveries_made:
                agent.discoveries_made.append(tool.id)
        else:
            reward = 0.2
        
        agent.energy -= energy_cost
        
        return ActionResult(
            'combine',
            True,
            reward,
            energy_cost,
            {
                'tool_id': tool.id,
                'tool_type': tool.get_tool_type(),
                'discovery_type': discovery_type,
                'components': obj_ids
            }
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
        
        success_prob = base_success_prob + agent.genes.strength * 0.2
        
        if random.random() < success_prob:
            # Успешная охота
            environment.remove_object(target.id)
            agent.add_to_inventory(target.id)
            
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
        
        return ActionResult('attack', success_prob > 0.5, reward, energy_cost, result_data)
    
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
        
        if random.random() < success_prob:
            # Успешное разрушение
            # Создаем новые ресурсы из разрушенного объекта
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
        
        return ActionResult('break', success_prob > 0.5, reward, energy_cost, result_data)
    
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

        # Сонливость снижается
        agent.sleepiness = max(0.0, sleepiness_before - 0.45 * eff)

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
            'sleep': AgentActions.execute_sleep
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
        
        return actions
