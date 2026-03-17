"""
Core simulation module - main simulation controller
"""

from typing import Dict, List, Any, Optional
import random
import time

from .environment import Environment, EnvironmentConfig
from .agent import Agent, AgentFactory, generate_thought, ACTION_TO_SKILL, Skills
from .agent_actions import ActionExecutor, ActionResult, BUILDING_RECIPES
from .objects import ObjectFactory
from .tools import ToolLibrary
from ..learning.q_learning import LearningManager
from ..evolution.genetics import EvolutionManager
from ..utils.metrics import MetricsCalculator, SimulationMetrics


class SimulationState:
    """Состояние симуляции"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.timestep = 0

        water_cfg = (self.config.get('simulation', {}).get('world', {}) or {})
        self.drowning_grace_steps = int(water_cfg.get('drowning_grace_steps', 90) or 90)
        self.drowning_base_risk = float(water_cfg.get('drowning_base_risk', 0.0015) or 0.0015)
        self.drowning_risk_gain = float(water_cfg.get('drowning_risk_gain', 0.00035) or 0.00035)
        self.drowning_risk_cap = float(water_cfg.get('drowning_risk_cap', 0.04) or 0.04)
        duration = None
        try:
            duration = self.config.get('experiment', {}).get('duration', None)
        except Exception:
            duration = None

        # duration <= 0 or missing => infinite simulation
        if duration is None or (isinstance(duration, (int, float)) and duration <= 0):
            self.max_steps = None
        else:
            self.max_steps = int(duration)
        
        # Инициализация компонентов
        self._initialize_environment()
        self._initialize_agents()
        self._initialize_systems()
        
        # Метрики
        self.metrics_calculator = MetricsCalculator()
        self.metrics_history: List[SimulationMetrics] = []
        
        # События
        self.events_log = []
        
        # Статистика
        self.total_births = 0
        self.total_deaths = 0
        self.total_discoveries = 0
        
        # Мёртвые агенты (надгробия на карте)
        self.dead_agents: List[Dict[str, Any]] = []
    
    def _initialize_environment(self):
        """Инициализирует среду"""
        world_config = self.config['simulation']['world']
        env_config = EnvironmentConfig(
            width=world_config['width'],
            height=world_config['height'],
            seed=world_config['seed'],
            disable_random_water_lakes=bool(world_config.get('disable_random_water_lakes', False)),
            disable_random_initial_resources=bool(world_config.get('disable_random_initial_resources', False)),
        )
        
        self.environment = Environment(env_config)
    
    def _initialize_agents(self):
        """Инициализирует агентов"""
        agents_config = self.config['simulation']['agents']
        initial_population = agents_config['initial_population']
        
        self.agents: Dict[str, Agent] = {}
        
        # Создаем начальную популяцию
        empty_positions = self.environment.get_empty_positions(initial_population)
        
        for i in range(initial_population):
            if i < len(empty_positions):
                position = empty_positions[i]
            else:
                position = (random.randint(0, self.environment.width - 1),
                           random.randint(0, self.environment.height - 1))
            
            agent = AgentFactory.create_random_agent(f"agent_{i}", position)
            agent.birth_time = 0
            agent.display_name = agent.invent_name()
            self.agents[agent.id] = agent
    
    def _initialize_systems(self):
        """Инициализирует системы симуляции"""
        # Система действий
        self.action_executor = ActionExecutor()
        
        # Система обучения
        learning_config = self.config['simulation']['learning']
        self.learning_manager = LearningManager()
        
        for agent in self.agents.values():
            self.learning_manager.register_agent(agent)
        
        # Система эволюции
        evolution_config = self.config['simulation']['evolution']
        agents_config = self.config['simulation']['agents']
        self.evolution_manager = EvolutionManager(
            population_size=agents_config['max_population'],
            mutation_rate=evolution_config['mutation_rate'],
            selection_pressure=evolution_config['selection_pressure']
        )
        
        self.evolution_manager.reproduction_interval = evolution_config['reproduction_interval']
    
    def step(self) -> bool:
        """Выполняет один шаг симуляции"""
        if self.max_steps is not None and self.timestep >= self.max_steps:
            return False
        
        # Обновление среды
        self.environment.update(self.timestep)

        # Обновление костров и ягодных кустов
        self._update_campfires_and_bushes()

        # Закон Ферхюльста: коэффициент ёмкости среды (раз в тик, хранится на environment)
        _n = len(self.agents)
        _food = sum(
            int(getattr(o, 'quantity', 1))
            for o in self.environment.objects.values()
            if getattr(o, 'nutrition', 0.0) > 0.3 and getattr(o, 'toxicity', 0.0) < 0.4
        )
        _K = max(float(_n + 5), _food / 3.0)
        self.environment._logistic_factor = max(0.2, 1.0 - _n / _K)

        # Обработка агентов
        self._process_agents()

        # Эволюционные процессы
        if self.evolution_manager.should_evolve(self.timestep):
            self._process_evolution()

        # Сбор метрик
        self._collect_metrics()

        # Обновление времени
        self.timestep += 1
        
        return True
    
    def _process_agents(self):
        """Обрабатывает действия всех агентов"""
        # Случайный порядок обработки
        agent_ids = list(self.agents.keys())
        random.shuffle(agent_ids)
        
        for agent_id in agent_ids:
            agent = self.agents[agent_id]

            action = "rest"
            
            # Проверка жизнеспособности
            if not agent.is_alive():
                self._handle_agent_death(agent)
                continue
            
            # Восприятие среды
            local_env = agent.perceive(self.environment)

            # Находим других агентов в радиусе коммуникации
            communication_radius = 2
            nearby_agents = []
            for other in self.agents.values():
                if other.id == agent.id:
                    continue
                dx = other.position[0] - agent.position[0]
                dy = other.position[1] - agent.position[1]
                if abs(dx) <= communication_radius and abs(dy) <= communication_radius:
                    nearby_agents.append(other)

            # Находим ближайших агентов (для mate/care)
            close_radius = 1
            close_agents = []
            for other in self.agents.values():
                if other.id == agent.id:
                    continue
                dx = other.position[0] - agent.position[0]
                dy = other.position[1] - agent.position[1]
                if abs(dx) <= close_radius and abs(dy) <= close_radius:
                    close_agents.append(other)

            # Кандидаты для размножения (взрослые, противоположный пол, female не беременна, условия норм)
            mate_candidates = []
            try:
                agent_sex = getattr(agent, 'sex', 'unknown')
                agent_is_child = bool(getattr(agent, 'is_child', lambda: False)())
                agent_pregnant = bool(getattr(agent, 'pregnant', False)) if agent_sex == 'female' else False
                agent_repro_ok = bool(getattr(agent, 'can_reproduce', lambda: False)())
            except Exception:
                agent_sex = getattr(agent, 'sex', 'unknown')
                agent_is_child = False
                agent_pregnant = False
                agent_repro_ok = False

            if close_agents and (not agent_is_child) and (not agent_pregnant) and agent_repro_ok:
                for other in close_agents:
                    try:
                        if bool(getattr(other, 'is_child', lambda: False)()):
                            continue
                        other_sex = getattr(other, 'sex', 'unknown')
                        if {agent_sex, other_sex} != {'male', 'female'}:
                            continue
                        # Ensure female partner isn't already pregnant
                        if other_sex == 'female' and bool(getattr(other, 'pregnant', False)):
                            continue
                        if not bool(getattr(other, 'can_reproduce', lambda: False)()):
                            continue
                        mate_candidates.append(other)
                    except Exception:
                        continue
            
            # Получение доступных действий
            available_actions = self.action_executor.get_available_actions(agent, self.environment)

            if nearby_agents:
                available_actions.append('communicate')

            # Поделиться ягодами: нужны ягоды в инвентаре и соседи рядом
            if nearby_agents:
                has_berries_to_share = any(
                    (bo := self.environment.objects.get(oid)) and bo.type == 'berry' and bo.quantity > 1
                    for oid in agent.inventory
                )
                if has_berries_to_share:
                    available_actions.append('share')

            # Лечение травами: survival lv4+ (>= 0.3) + есть plant/berry + здоровье < 0.8
            _inv = getattr(agent, 'inventory', []) or []
            _has_herb = any(
                (ho := self.environment.objects.get(oid)) and ho.type in ('plant', 'berry')
                for oid in _inv
            )
            if agent.skills.get('survival') >= 0.3 and _has_herb and agent.health < 0.8:
                available_actions.append('treat')

            # Убежище: crafting lv5+ (>= 0.4) + 5 wood + 3 stone + не на воде
            _wood_cnt  = sum(1 for oid in _inv if (wo := self.environment.objects.get(oid)) and wo.type == 'wood')
            _stone_cnt = sum(1 for oid in _inv if (so := self.environment.objects.get(oid)) and so.type == 'stone')
            _no_shelter = not any(o.type == 'shelter'
                                  for o in self.environment.get_objects_at_position(agent.position))
            if (agent.skills.get('crafting') >= 0.4 and _wood_cnt >= 5 and _stone_cnt >= 3
                    and not self.environment.is_water(agent.position) and _no_shelter):
                available_actions.append('build_shelter')

            # build (settlement buildings): проверяем все рецепты зданий
            if not self.environment.is_water(agent.position):
                _cell_types = set(o.type for o in self.environment.get_objects_at_position(agent.position))
                for _btype, _brecipe in BUILDING_RECIPES.items():
                    if _btype in _cell_types:
                        continue  # уже есть такое здание тут
                    _bskill, _bmin = _brecipe['skill']
                    if agent.skills.get(_bskill) < _bmin:
                        continue
                    _bok = True
                    for _bmat, _bcnt in _brecipe['materials'].items():
                        _bhave = sum(1 for oid in _inv
                                     if (bo := self.environment.objects.get(oid)) and bo.type == _bmat)
                        if _bhave < _bcnt:
                            _bok = False
                            break
                    if _bok:
                        available_actions.append('build')
                        break  # достаточно одного доступного рецепта

            # upgrade: улучшение здания рядом (lvl < 3) + материалы
            _has_upgradeable = False
            for _dx in range(-1, 2):
                for _dy in range(-1, 2):
                    for _uo in self.environment.get_objects_at_position(
                            (agent.position[0] + _dx, agent.position[1] + _dy)):
                        if (getattr(_uo, 'building_owner_id', None) == agent.id
                                and getattr(_uo, 'building_level', 1) < 3
                                and getattr(_uo, 'building_type', None) in BUILDING_RECIPES):
                            _has_upgradeable = True
                            break
                    if _has_upgradeable:
                        break
                if _has_upgradeable:
                    break
            if _has_upgradeable and agent.skills.get('crafting') >= 0.3:
                # Проверяем наличие хоть каких-то материалов
                _has_mats = (_wood_cnt >= 2 or _stone_cnt >= 2)
                if _has_mats:
                    available_actions.append('upgrade')

            # repair_building: починка повреждённого здания рядом (durability < 0.7)
            _has_damaged_bld = False
            for _dx in range(-1, 2):
                for _dy in range(-1, 2):
                    for _ro in self.environment.get_objects_at_position(
                            (agent.position[0] + _dx, agent.position[1] + _dy)):
                        if (getattr(_ro, 'building_owner_id', None) == agent.id
                                and getattr(_ro, 'building_type', None) in BUILDING_RECIPES
                                and getattr(_ro, 'durability', 1.0) < 0.7):
                            _has_damaged_bld = True
                            break
                    if _has_damaged_bld:
                        break
                if _has_damaged_bld:
                    break
            if _has_damaged_bld and (_wood_cnt >= 2 or _stone_cnt >= 2):
                available_actions.append('repair_building')

            # mate only when a valid close partner exists
            if mate_candidates:
                available_actions.append('mate')

            # care when close child exists
            if any(getattr(a, 'is_child', lambda: False)() for a in close_agents):
                available_actions.append('care')

            # ── Торговля: deposit/collect_trade ──
            _ax, _ay = agent.position
            _has_own_post = False
            _has_foreign_post = False
            for _dx in range(-2, 3):
                for _dy in range(-2, 3):
                    for _to in self.environment.get_objects_at_position((_ax + _dx, _ay + _dy)):
                        if getattr(_to, 'building_type', None) == 'trading_post':
                            if getattr(_to, 'building_owner_id', None) == agent.id:
                                _has_own_post = True
                            elif len(getattr(_to, 'stored_items', [])) > 0:
                                _has_foreign_post = True
                    if _has_own_post and _has_foreign_post:
                        break
                if _has_own_post and _has_foreign_post:
                    break
            # deposit: свой пост + есть дубликаты в инвентаре
            if _has_own_post and len(_inv) >= 2:
                available_actions.append('deposit')
            # collect_trade: чужой пост с предметами + есть место в инвентаре
            if _has_foreign_post and len(_inv) < getattr(agent, 'inventory_capacity', 5):
                available_actions.append('collect_trade')

            # ── Новые действия: доступность ──

            # cook: cooking lv2+ (>= 0.1) или clay_oven рядом, + еда в инвентаре
            _has_cookable = any(
                (fo := self.environment.objects.get(oid)) and fo.type in ('berry', 'plant', 'mushroom', 'fish')
                for oid in _inv
            )
            if _has_cookable:
                _ax, _ay = agent.position
                _heat_near = False
                for _dx in range(-2, 3):
                    for _dy in range(-2, 3):
                        for _o in self.environment.get_objects_at_position((_ax + _dx, _ay + _dy)):
                            if (_o.type == 'campfire' and getattr(_o, 'fuel_ticks', 0) > 0) or _o.type == 'clay_oven':
                                _heat_near = True
                                break
                        if _heat_near:
                            break
                    if _heat_near:
                        break
                if _heat_near and agent.skills.get('cooking') >= 0.1:
                    available_actions.append('cook')

            # fish: gathering lv3+ (>= 0.2) + fishing_rod + рядом вода
            _has_rod = any(
                (t := self.environment.tools.get(tid)) and getattr(t, 'kind', None) == 'fishing_rod' and not t.is_broken()
                for tid in (getattr(agent, 'tools', []) or [])
            )
            if _has_rod and agent.skills.get('gathering') >= 0.2:
                _ax, _ay = agent.position
                _water_near = False
                for _dx in range(-1, 2):
                    for _dy in range(-1, 2):
                        if self.environment.is_water((_ax + _dx, _ay + _dy)):
                            _water_near = True
                            break
                    if _water_near:
                        break
                if _water_near:
                    available_actions.append('fish')

            # smelt: crafting lv5+ (>= 0.4) + ore + stone_furnace + campfire рядом
            _has_ore = any(
                (oo := self.environment.objects.get(oid)) and oo.type == 'ore'
                for oid in _inv
            )
            if _has_ore and agent.skills.get('crafting') >= 0.4:
                _ax, _ay = agent.position
                _has_furn = False
                _has_camp = False
                for _dx in range(-2, 3):
                    for _dy in range(-2, 3):
                        for _o in self.environment.get_objects_at_position((_ax + _dx, _ay + _dy)):
                            if _o.type == 'stone_furnace':
                                _has_furn = True
                            if _o.type == 'campfire' and getattr(_o, 'fuel_ticks', 0) > 0:
                                _has_camp = True
                        if _has_furn and _has_camp:
                            break
                    if _has_furn and _has_camp:
                        break
                if _has_furn and _has_camp:
                    available_actions.append('smelt')

            # repair: crafting lv3+ (>= 0.2) + повреждённый инструмент + материал
            _has_damaged = any(
                (t := self.environment.tools.get(tid)) and t.durability_left < 60
                for tid in (getattr(agent, 'tools', []) or [])
            )
            _has_mat = any(
                (mo := self.environment.objects.get(oid)) and mo.type in ('wood', 'stone', 'bone', 'fiber', 'ore', 'metal_ingot')
                for oid in _inv
            )
            if _has_damaged and _has_mat and agent.skills.get('crafting') >= 0.2:
                available_actions.append('repair')

            # tan_hide: crafting lv2+ (>= 0.1) + bone + plant + campfire рядом
            _has_bone_inv = any((bo := self.environment.objects.get(oid)) and bo.type == 'bone' for oid in _inv)
            _has_plant_inv = any((po := self.environment.objects.get(oid)) and po.type == 'plant' for oid in _inv)
            if _has_bone_inv and _has_plant_inv and agent.skills.get('crafting') >= 0.1:
                _ax, _ay = agent.position
                _fire_near = False
                for _dx in range(-2, 3):
                    for _dy in range(-2, 3):
                        for _o in self.environment.get_objects_at_position((_ax + _dx, _ay + _dy)):
                            if _o.type == 'campfire' and getattr(_o, 'fuel_ticks', 0) > 0:
                                _fire_near = True
                                break
                        if _fire_near:
                            break
                    if _fire_near:
                        break
                if _fire_near:
                    available_actions.append('tan_hide')
            
            # Простой инстинкт ухода: если рядом ребёнок и он голоден/без энергии — приоритет care
            hungry_child = None
            for other in close_agents:
                if getattr(other, 'is_child', lambda: False)() and (other.hunger > 0.65 or other.energy < 0.35):
                    hungry_child = other
                    break

            # Инстинкт голода: искать/добывать/есть еду
            hungry_self = agent.hunger > 0.6
            has_food_in_inventory = False
            if hungry_self:
                for obj_id in agent.inventory:
                    obj = self.environment.objects.get(obj_id)
                    if obj is not None and obj.is_edible():
                        has_food_in_inventory = True
                        break

            food_visible_here = False
            if hungry_self:
                for obj in local_env.get('perceived_objects', []) or []:
                    try:
                        if obj.is_edible():
                            food_visible_here = True
                            break
                    except Exception:
                        continue

            # Инстинкт жажды: искать/пить воду (+ колодец рядом)
            thirsty_self = getattr(agent, 'thirst', 0.0) > 0.6
            water_here = False
            if thirsty_self:
                try:
                    water_here = any(o.type == 'water' for o in self.environment.get_objects_at_position(agent.position))
                    # Колодец рядом тоже считается источником воды
                    if not water_here:
                        _ax, _ay = agent.position
                        for _dx in range(-2, 3):
                            for _dy in range(-2, 3):
                                for _wo in self.environment.get_objects_at_position((_ax + _dx, _ay + _dy)):
                                    if getattr(_wo, 'building_type', None) == 'well':
                                        water_here = True
                                        break
                                if water_here:
                                    break
                            if water_here:
                                break
                except Exception:
                    water_here = False

            # Инстинкт "творчества": пробовать создавать инструменты
            can_toolmake = (
                len(getattr(agent, 'inventory', []) or []) >= 2
                and getattr(agent, 'energy', 0.0) > 0.35
                and getattr(agent, 'hunger', 0.0) < 0.65
                and getattr(agent, 'thirst', 0.0) < 0.65
                and getattr(agent, 'sleepiness', 0.0) < 0.8
            )

            # Инстинкт подготовки к инструментам: собирать материалы (stone/wood/bone/fiber)
            inventory = getattr(agent, 'inventory', []) or []
            inventory_capacity = getattr(agent, 'inventory_capacity', 0) or 0
            has_inventory_space = len(inventory) < inventory_capacity

            tool_material_in_inventory = 0
            for obj_id in inventory:
                obj = self.environment.objects.get(obj_id)
                if obj is not None and obj.type in ('stone', 'wood', 'bone', 'fiber'):
                    tool_material_in_inventory += 1

            wants_tool_materials = (
                has_inventory_space
                and tool_material_in_inventory < 3
                and getattr(agent, 'energy', 0.0) > 0.25
                and getattr(agent, 'hunger', 0.0) < 0.65
                and getattr(agent, 'thirst', 0.0) < 0.65
            )

            # Сколько дерева в инвентаре (для костра)
            wood_in_inventory = sum(
                1 for oid in inventory
                if (wo := self.environment.objects.get(oid)) and wo.type == 'wood'
            )
            wood_visible_here = any(
                getattr(o, 'type', None) == 'wood'
                for o in local_env.get('perceived_objects', []) or []
            )

            tool_material_visible_here = False
            if wants_tool_materials:
                for obj in local_env.get('perceived_objects', []) or []:
                    try:
                        if getattr(obj, 'type', None) in ('stone', 'wood', 'bone', 'fiber'):
                            tool_material_visible_here = True
                            break
                    except Exception:
                        continue

            # Выбор действия (personality-influenced)
            decision_maker = self.learning_manager.get_decision_maker(agent_id)
            pers = getattr(agent, 'personality', None)
            if decision_maker:
                if hungry_child and 'care' in available_actions:
                    # Empathetic agents always care; others sometimes skip
                    care_prob = 0.5 + 0.5 * (pers.empathy if pers else 0.5)
                    if random.random() < care_prob:
                        action = 'care'
                    else:
                        action = decision_maker.select_action(local_env, available_actions)
                # Размножение: когда всё спокойно и рядом подходящий партнёр
                elif mate_candidates and 'mate' in available_actions:
                    if not (getattr(agent, 'hunger', 0.0) > 0.75 or getattr(agent, 'thirst', 0.0) > 0.75 or getattr(agent, 'sleepiness', 0.0) > 0.85):
                        base = 0.002
                        base += 0.010 * float(getattr(agent.genes, 'social_tendency', 0.5))
                        base += 0.006 * (pers.sociability if pers else 0.5)
                        if random.random() < base:
                            action = 'mate'
                        else:
                            action = decision_maker.select_action(local_env, available_actions)
                    else:
                        action = decision_maker.select_action(local_env, available_actions)
                # Сон: ночью при высокой сонливости
                elif (not getattr(self.environment, 'is_daytime', True)) and getattr(agent, 'sleepiness', 0.0) > 0.65 and 'sleep' in available_actions:
                    if not (getattr(agent, 'hunger', 0.0) > 0.95 or getattr(agent, 'thirst', 0.0) > 0.95):
                        action = 'sleep'
                    else:
                        action = decision_maker.select_action(local_env, available_actions)
                # Personality-driven spontaneous actions (when not in urgent need)
                elif not (hungry_self or thirsty_self) and pers:
                    chosen = False
                    is_night_now = not getattr(self.environment, 'is_daytime', True)

                    # ── Убежище: высочайший приоритет (строим как только можем) ─────
                    if not chosen and 'build_shelter' in available_actions:
                        if random.random() < 0.90:
                            action = 'build_shelter'
                            chosen = True

                    # ── Лечение: при низком здоровье ────────────────────────────────
                    if not chosen and 'treat' in available_actions and agent.health < 0.5:
                        if random.random() < 0.80:
                            action = 'treat'
                            chosen = True

                    # ── Костёр: высокий приоритет когда уже есть 3 дерева ──────────
                    if not chosen and 'light_fire' in available_actions:
                        prob_fire = (0.75 if is_night_now else 0.40) + 0.2 * getattr(pers, 'industriousness', 0.5)
                        if random.random() < prob_fire:
                            action = 'light_fire'
                            chosen = True

                    # Добор дерева до 3 штук для костра (если уже начали)
                    if not chosen and 0 < wood_in_inventory < 3 and wood_visible_here and 'gather' in available_actions:
                        if random.random() < 0.55 + 0.3 * getattr(pers, 'industriousness', 0.5):
                            action = 'gather'
                            chosen = True

                    # Social agents spontaneously communicate
                    if not chosen and 'communicate' in available_actions and nearby_agents:
                        if random.random() < 0.03 * pers.sociability:
                            action = 'communicate'
                            chosen = True
                    # Эмпатичные/социальные агенты делятся ягодами
                    if not chosen and 'share' in available_actions:
                        prob_share = 0.02 + 0.06 * getattr(pers, 'empathy', 0.5) + 0.02 * pers.sociability
                        if random.random() < prob_share:
                            action = 'share'
                            chosen = True
                    # Curious agents explore more
                    if not chosen and 'move' in available_actions:
                        if random.random() < 0.02 * pers.curiosity:
                            action = 'move'
                            chosen = True
                    # Industrious agents gather materials proactively
                    if not chosen and wants_tool_materials and tool_material_visible_here and 'gather' in available_actions:
                        if random.random() < 0.3 + 0.5 * pers.industriousness:
                            action = 'gather'
                            chosen = True
                    # Curious/patient agents try crafting more
                    if not chosen and can_toolmake and 'combine' in available_actions:
                        prob = 0.01 + 0.06 * getattr(agent.genes, 'intelligence', 0.5) + 0.05 * pers.curiosity + 0.03 * pers.patience
                        if random.random() < prob:
                            action = 'combine'
                            chosen = True
                    # Brave agents attack when possible
                    if not chosen and 'attack' in available_actions:
                        if random.random() < 0.04 * pers.bravery:
                            action = 'attack'
                            chosen = True
                    # Plant berries occasionally
                    if not chosen and 'plant_berry' in available_actions:
                        prob_plant = 0.02 + 0.04 * getattr(pers, 'industriousness', 0.5)
                        if random.random() < prob_plant:
                            action = 'plant_berry'
                            chosen = True

                    # ── Инстинкты для новых действий ──

                    # Готовка: если есть токсичная еда (mushroom) или просто голоден
                    if not chosen and 'cook' in available_actions:
                        _has_toxic = any(
                            (fo := self.environment.objects.get(oid)) and fo.type in ('mushroom', 'fish')
                            for oid in agent.inventory
                        )
                        prob_cook = (0.70 if _has_toxic else 0.15) + 0.15 * pers.patience
                        if random.random() < prob_cook:
                            action = 'cook'
                            chosen = True

                    # Рыбалка: при голоде и пустом инвентаре
                    if not chosen and 'fish' in available_actions:
                        prob_fish = 0.10 + 0.20 * pers.patience + (0.30 if agent.hunger > 0.4 else 0.0)
                        if random.random() < prob_fish:
                            action = 'fish'
                            chosen = True

                    # Плавка: если есть руда и рядом горн
                    if not chosen and 'smelt' in available_actions:
                        prob_smelt = 0.40 + 0.20 * pers.industriousness
                        if random.random() < prob_smelt:
                            action = 'smelt'
                            chosen = True

                    # Ремонт: если инструмент сильно повреждён
                    if not chosen and 'repair' in available_actions:
                        prob_repair = 0.50 + 0.20 * pers.patience
                        if random.random() < prob_repair:
                            action = 'repair'
                            chosen = True

                    # Выделка кожи: если есть кость и растение у костра
                    if not chosen and 'tan_hide' in available_actions:
                        prob_tan = 0.25 + 0.20 * pers.industriousness
                        if random.random() < prob_tan:
                            action = 'tan_hide'
                            chosen = True

                    # Ремонт зданий: высший приоритет для терпеливых
                    if not chosen and 'repair_building' in available_actions:
                        prob_rb = 0.70 + 0.20 * pers.patience
                        if random.random() < prob_rb:
                            action = 'repair_building'
                            chosen = True

                    # Улучшение зданий: трудолюбивые
                    if not chosen and 'upgrade' in available_actions:
                        prob_up = 0.20 + 0.25 * pers.industriousness + 0.15 * pers.patience
                        if random.random() < prob_up:
                            action = 'upgrade'
                            chosen = True

                    # Торговля: социальные/эмпатичные агенты кладут на пост
                    if not chosen and 'deposit' in available_actions:
                        prob_dep = 0.15 + 0.25 * pers.empathy + 0.15 * pers.sociability
                        if random.random() < prob_dep:
                            action = 'deposit'
                            chosen = True

                    # Сбор с чужого поста: любопытные/социальные
                    if not chosen and 'collect_trade' in available_actions:
                        prob_col = 0.30 + 0.20 * pers.curiosity + 0.15 * pers.sociability
                        if random.random() < prob_col:
                            action = 'collect_trade'
                            chosen = True

                    # Строительство зданий: трудолюбивые и терпеливые агенты
                    if not chosen and 'build' in available_actions:
                        prob_build = 0.35 + 0.25 * pers.industriousness + 0.10 * pers.patience
                        if random.random() < prob_build:
                            action = 'build'
                            chosen = True

                    if not chosen:
                        action = decision_maker.select_action(local_env, available_actions)
                # Ремонт зданий: высший приоритет
                elif 'repair_building' in available_actions:
                    action = 'repair_building'
                # Убежище: строим при первой возможности
                elif 'build_shelter' in available_actions:
                    action = 'build_shelter'
                # Строительство зданий
                elif 'build' in available_actions:
                    action = 'build'
                # Улучшение зданий
                elif 'upgrade' in available_actions:
                    action = 'upgrade'
                # Лечение: при низком здоровье
                elif 'treat' in available_actions and agent.health < 0.5:
                    action = 'treat'
                # Костёр: если есть 3 дерева — жечь немедленно
                elif 'light_fire' in available_actions:
                    action = 'light_fire'
                # Добор дерева до 3 штук для костра
                elif 0 < wood_in_inventory < 3 and wood_visible_here and 'gather' in available_actions:
                    action = 'gather'
                # Сбор материалов для инструментов
                elif wants_tool_materials and tool_material_visible_here and 'gather' in available_actions:
                    action = 'gather'
                # Создание инструментов
                elif can_toolmake and 'combine' in available_actions:
                    prob = 0.01 + 0.06 * getattr(agent.genes, 'intelligence', 0.5) + 0.04 * getattr(agent.genes, 'exploration_bias', 0.5)
                    if random.random() < prob:
                        action = 'combine'
                    else:
                        action = decision_maker.select_action(local_env, available_actions)
                elif thirsty_self and water_here and 'drink' in available_actions:
                    action = 'drink'
                elif hungry_self and has_food_in_inventory and 'consume' in available_actions:
                    action = 'consume'
                elif hungry_self and food_visible_here and 'gather' in available_actions:
                    action = 'gather'
                elif (thirsty_self or hungry_self or wants_tool_materials) and 'move' in available_actions:
                    action = 'move'
                else:
                    action = decision_maker.select_action(local_env, available_actions)
            else:
                action = "rest"
            
            # Тяга к костру ночью / при сонливости
            if 'move' in available_actions and not (hungry_self or thirsty_self):
                is_night = not getattr(self.environment, 'is_daytime', True)
                is_sleepy = getattr(agent, 'sleepiness', 0.0) > 0.4
                if is_night or is_sleepy:
                    for obj in local_env.get('perceived_objects', []) or []:
                        if getattr(obj, 'type', '') == 'campfire' and \
                                getattr(obj, 'fuel_ticks', 0) > 0:
                            action = 'move'
                            setattr(agent, '_move_target', obj.position)
                            break

            # Притяжение к поселениям: социальные агенты ходят в гости
            if ('move' in available_actions and action == 'move'
                    and pers and random.random() < 0.02 * pers.sociability):
                # Ищем ближайшее чужое здание (trading_post, well, garden)
                _visit_target = None
                _visit_dist = 999
                for _vobj in (local_env.get('perceived_objects', []) or []):
                    _vbt = getattr(_vobj, 'building_type', None)
                    if _vbt in ('trading_post', 'well', 'garden'):
                        _vo = getattr(_vobj, 'building_owner_id', None)
                        if _vo and _vo != agent.id:
                            _vd = abs(_vobj.position[0] - agent.position[0]) + abs(_vobj.position[1] - agent.position[1])
                            if _vd < _visit_dist:
                                _visit_dist = _vd
                                _visit_target = _vobj.position
                if _visit_target:
                    setattr(agent, '_move_target', _visit_target)

            # Команда владельца — переопределяет выбранное действие
            _pending = getattr(agent, 'pending_command', None)
            _pending_ticks = int(getattr(agent, 'pending_command_ticks', 0))
            if _pending and _pending_ticks > 0:
                _new_ticks = _pending_ticks - 1
                # Используем команду если действие доступно или всегда разрешено (rest/sleep/move)
                if _pending in available_actions or _pending in ('rest', 'sleep', 'move'):
                    action = _pending
                    # Передаём целевой крафт/сбор
                    if _pending == 'combine':
                        _ct = getattr(agent, 'pending_craft_target', None)
                        if _ct:
                            setattr(agent, '_craft_target_kind', _ct)
                    if _pending == 'gather':
                        _gt = getattr(agent, 'pending_gather_target', None)
                        if _gt:
                            setattr(agent, '_gather_target_type', _gt)
                    if _pending == 'build':
                        _bt = getattr(agent, 'pending_build_target', None)
                        if _bt:
                            setattr(agent, '_build_target', _bt)
                    if _pending == 'upgrade':
                        _ut = getattr(agent, 'pending_upgrade_target', None)
                        if _ut:
                            setattr(agent, '_upgrade_target', _ut)
                setattr(agent, 'pending_command_ticks', _new_ticks)
                if _new_ticks <= 0:
                    setattr(agent, 'pending_command', None)
                    setattr(agent, 'pending_command_ticks', 0)
                    setattr(agent, 'pending_craft_target', None)
                    setattr(agent, 'pending_gather_target', None)
                    setattr(agent, 'pending_build_target', None)

            # Исполнение действия
            if action in ('communicate', 'mate', 'care', 'share'):
                result = self.action_executor.execute_action(
                    agent,
                    self.environment,
                    action,
                    other_agents=(mate_candidates if action == 'mate' else close_agents) if action in ('mate', 'care') else nearby_agents,
                )
            else:
                result = self.action_executor.execute_action(agent, self.environment, action)
            
            # Обучение
            if decision_maker:
                new_state = decision_maker.state_encoder.encode_state(agent, local_env)
                previous_state = getattr(agent, '_last_state', new_state)
                
                decision_maker.learn_from_experience(
                    previous_state, action, result, new_state, available_actions
                )
                
                agent._last_state = new_state
            
            # Обновление состояния агента
            # Передаем агенту информацию о времени суток для физиологии
            setattr(agent, 'is_daytime', getattr(self.environment, 'is_daytime', True))
            agent.update_physiology()

            # Тепло от ближайшего костра (радиус 3)
            try:
                ax, ay = agent.position
                for _dx in range(-3, 4):
                    for _dy in range(-3, 4):
                        _fire_objs = self.environment.get_objects_at_position((ax + _dx, ay + _dy))
                        if any(o.type == 'campfire' and getattr(o, 'fuel_ticks', 0) > 0
                               for o in _fire_objs):
                            agent.sleepiness = max(0.0, agent.sleepiness - 0.002)
                            agent.energy = min(1.0, agent.energy + 0.001)
                            break
                    else:
                        continue
                    break
            except Exception:
                pass

            # Бонус от своего убежища (зависит от shelter_bonus_mult / shelter_radius)
            try:
                _rb = getattr(agent, 'research_bonuses', {})
                _sh_radius = int(2 + _rb.get('shelter_radius', 0))
                _sh_bonus  = 0.002 * (1.0 + _rb.get('shelter_bonus_mult', 0.0))
                ax, ay = agent.position
                _agent_id = agent.id
                for _dx in range(-_sh_radius, _sh_radius + 1):
                    for _dy in range(-_sh_radius, _sh_radius + 1):
                        _sh_objs = self.environment.get_objects_at_position((ax + _dx, ay + _dy))
                        if any(o.type == 'shelter' and getattr(o, 'shelter_owner_id', None) == _agent_id
                               for o in _sh_objs):
                            agent.energy = min(1.0, agent.energy + _sh_bonus)
                            agent.sleepiness = max(0.0, agent.sleepiness - _sh_bonus)
                            break
                    else:
                        continue
                    break
            except Exception:
                pass

            # ── Бонусы от зданий (Settlement System) ────────────────────────────
            try:
                ax, ay = agent.position
                _agent_id = agent.id
                _bld_radius = 3  # радиус действия зданий
                # Альянсы: собираем ID союзников (trust > 0.5)
                _ally_ids = set()
                if hasattr(agent, 'social'):
                    for _aid, _trust in agent.social.relationships.items():
                        if _trust > 0.5:
                            _ally_ids.add(_aid)
                _my_buildings = []
                for _dx in range(-_bld_radius, _bld_radius + 1):
                    for _dy in range(-_bld_radius, _bld_radius + 1):
                        for _bo in self.environment.get_objects_at_position((ax + _dx, ay + _dy)):
                            _bo_owner = getattr(_bo, 'building_owner_id', None)
                            if _bo_owner == _agent_id or _bo_owner in _ally_ids:
                                _my_buildings.append(_bo)

                for _bld in _my_buildings:
                    _bt = getattr(_bld, 'building_type', None)
                    _blvl = getattr(_bld, 'building_level', 1)
                    # Повреждённые здания (< 30% прочности) не дают бонусов
                    if getattr(_bld, 'durability', 1.0) < 0.3:
                        continue

                    # Склад: L1 +5, L2 +8, L3 +12 к вместимости
                    if _bt == 'storage_hut':
                        _s_bonus = {1: 5, 2: 8, 3: 12}.get(_blvl, 5)
                        _prev = getattr(agent, '_storage_bonus_val', 0)
                        if _prev != _s_bonus:
                            agent.inventory_capacity += (_s_bonus - _prev)
                            setattr(agent, '_storage_bonus_val', _s_bonus)
                        if not getattr(agent, '_storage_bonus_applied', False):
                            agent.inventory_capacity += _s_bonus
                            setattr(agent, '_storage_bonus_applied', True)
                            setattr(agent, '_storage_bonus_val', _s_bonus)

                    # Мастерская: L1 флаг, L2 -15% энергии, L3 -25% энергии
                    if _bt == 'workshop':
                        setattr(agent, '_workshop_nearby', True)
                        setattr(agent, '_workshop_level', _blvl)

                    # Огород: L1 каждые 50, L2 каждые 35, L3 каждые 25 тиков + herbs
                    if _bt == 'garden':
                        _interval = {1: 50, 2: 35, 3: 25}.get(_blvl, 50)
                        _max_store = {1: 5, 2: 8, 3: 12}.get(_blvl, 5)
                        _pt = getattr(_bld, 'produce_timer', 0) + 1
                        _stored = getattr(_bld, 'stored_produce', 0)
                        if _pt >= _interval and _stored < _max_store:
                            import uuid as _uuid_g
                            # L3 иногда производит herb вместо plant
                            _prod_type = 'herb' if (_blvl >= 3 and random.random() < 0.3) else 'plant'
                            _food_id = f"garden_{_prod_type}_{_uuid_g.uuid4().hex[:8]}"
                            _food = ObjectFactory.create_object(
                                _prod_type, _bld.position, _food_id,
                                getattr(self.environment, 'timestep', 0))
                            self.environment.add_object(_food)
                            setattr(_bld, 'stored_produce', _stored + 1)
                            _pt = 0
                        setattr(_bld, 'produce_timer', _pt)

                    # Колодец: L1 radius 2, L2 radius 3, L3 radius 3 + -0.002 thirst/tick
                    if _bt == 'well' and _blvl >= 3:
                        agent.thirst = max(0.0, agent.thirst - 0.002)

                    # Дозорная башня: L1 +3, L2 +4, L3 +5 к восприятию
                    if _bt == 'watchtower':
                        _w_bonus = {1: 3, 2: 4, 3: 5}.get(_blvl, 3)
                        if not getattr(agent, '_watchtower_bonus_applied', False):
                            agent.perception_radius += _w_bonus
                            setattr(agent, '_watchtower_bonus_applied', True)
                            setattr(agent, '_watchtower_bonus_val', _w_bonus)
                        elif getattr(agent, '_watchtower_bonus_val', 3) != _w_bonus:
                            _prev_w = getattr(agent, '_watchtower_bonus_val', 3)
                            agent.perception_radius += (_w_bonus - _prev_w)
                            setattr(agent, '_watchtower_bonus_val', _w_bonus)

                    # Сушилка: L1 каждые 30, L2 каждые 20, L3 каждые 15 тиков
                    if _bt == 'drying_rack':
                        _d_interval = {1: 30, 2: 20, 3: 15}.get(_blvl, 30)
                        _dt = getattr(_bld, 'dry_timer', 0) + 1
                        if _dt >= _d_interval:
                            for _fid in list(agent.inventory):
                                _fobj = self.environment.objects.get(_fid)
                                _dry_types = ('fish', 'mushroom') if _blvl >= 3 else ('fish',)
                                if _fobj and _fobj.type in _dry_types:
                                    agent.remove_from_inventory(_fid)
                                    self.environment.remove_object(_fid)
                                    import uuid as _uuid_d
                                    _cf_id = f"dried_{_uuid_d.uuid4().hex[:8]}"
                                    _cf = ObjectFactory.create_object(
                                        'cooked_food', agent.position, _cf_id,
                                        getattr(self.environment, 'timestep', 0))
                                    self.environment.objects[_cf_id] = _cf
                                    agent.add_to_inventory(_cf_id)
                                    break
                            _dt = 0
                        setattr(_bld, 'dry_timer', _dt)

                # Сбросить бонусы если нет зданий рядом
                _bld_types = set(getattr(b, 'building_type', None) for b in _my_buildings)
                if 'storage_hut' not in _bld_types and getattr(agent, '_storage_bonus_applied', False):
                    agent.inventory_capacity = max(5, agent.inventory_capacity - 5)
                    setattr(agent, '_storage_bonus_applied', False)
                if 'watchtower' not in _bld_types and getattr(agent, '_watchtower_bonus_applied', False):
                    agent.perception_radius = max(1, agent.perception_radius - 3)
                    setattr(agent, '_watchtower_bonus_applied', False)
                if 'workshop' not in _bld_types:
                    setattr(agent, '_workshop_nearby', False)
            except Exception:
                pass

            # ── Термодинамика ─────────────────────────────────────────────────────
            try:
                env_temp = self.environment.get_local_temperature(agent.position)

                # 1-й закон: метаболизм и действия генерируют тепло тела
                metabolic_heat = 0.05   # °C/тик базовое тепло жизнедеятельности
                action_heat = {
                    'move': 0.04, 'gather': 0.03, 'attack': 0.06,
                    'combine': 0.04, 'build_shelter': 0.05, 'build': 0.05,
                    'upgrade': 0.05, 'repair_building': 0.04,
                    'deposit': 0.01, 'collect_trade': 0.01,
                    'sleep': -0.02, 'rest': -0.01,
                }.get(getattr(agent, 'last_action', ''), 0.02)

                # Теплопередача: Закон Ньютона об охлаждении
                # ΔT = -λ*(T_body - T_env)*(1 - insulation) + Q_metabolic
                LAMBDA = 0.003
                insulation = 0.0
                _rb = getattr(agent, 'research_bonuses', {})
                _sh_r = 2 + int(_rb.get('shelter_radius', 0))
                ax, ay = agent.position
                for _dx in range(-_sh_r, _sh_r + 1):
                    for _dy in range(-_sh_r, _sh_r + 1):
                        _sh_objs = self.environment.get_objects_at_position((ax + _dx, ay + _dy))
                        if any(o.type == 'shelter' and getattr(o, 'shelter_owner_id', None) == agent.id for o in _sh_objs):
                            insulation = max(insulation, 0.50)
                            break
                    else:
                        continue
                    break
                insulation = min(insulation + _rb.get('insulation', 0.0), 0.80)

                # Wind-chill: ветер снижает теплоизоляцию
                _wind_s = float(getattr(self.environment, 'wind_speed', 0.0))
                insulation *= max(0.3, 1.0 - _wind_s * 0.6)

                heat_loss = LAMBDA * (agent.body_temp - env_temp) * (1.0 - insulation)
                agent.body_temp = max(20.0, min(45.0,
                    agent.body_temp - heat_loss + metabolic_heat + action_heat))

                # Эффекты на физиологию
                _dmg_red = min(0.80, getattr(agent, 'research_bonuses', {}).get('damage_reduction', 0.0))
                if agent.body_temp < 35.0:
                    sev = (35.0 - agent.body_temp) / 3.0
                    agent.hunger   = min(1.0, agent.hunger   + 0.006 * sev)
                    agent.thirst   = min(1.0, agent.thirst   + 0.002 * sev)
                    if agent.body_temp < 33.0:
                        agent.health = max(0.0, agent.health - 0.012 * sev * (1.0 - _dmg_red))
                elif agent.body_temp > 39.0:
                    sev = (agent.body_temp - 39.0) / 3.0
                    agent.thirst = min(1.0, agent.thirst + 0.009 * sev)
                    if agent.body_temp > 40.5:
                        agent.health = max(0.0, agent.health - 0.008 * sev * (1.0 - _dmg_red))
            except Exception:
                pass

            try:
                in_water = bool(self.environment.is_water(agent.position))
            except Exception:
                in_water = False

            if in_water:
                prev_ticks = int(getattr(agent, 'water_ticks', 0) or 0)
                ticks = prev_ticks + 1
                setattr(agent, 'water_ticks', ticks)
                setattr(agent, 'is_swimming', True)

                if ticks > self.drowning_grace_steps:
                    extra = ticks - self.drowning_grace_steps
                    risk = self.drowning_base_risk + self.drowning_risk_gain * float(extra)
                    if risk > self.drowning_risk_cap:
                        risk = self.drowning_risk_cap

                    if random.random() < risk:
                        agent.health = 0.0
                        setattr(agent, 'drowned', True)
            else:
                setattr(agent, 'water_ticks', 0)
                setattr(agent, 'is_swimming', False)

            agent.last_action_time = self.timestep

            # Беременность: отсчёт и роды
            if getattr(agent, 'pregnant', False):
                remaining = int(getattr(agent, 'pregnancy_remaining', 0))
                if remaining > 0:
                    setattr(agent, 'pregnancy_remaining', remaining - 1)
                if int(getattr(agent, 'pregnancy_remaining', 0)) == 0:
                    father_id = getattr(agent, 'pregnancy_father_id', None)
                    father = self.agents.get(father_id) if father_id else None
                    if father is not None:
                        child_id = f"child_{self.timestep}_{random.randint(1000, 9999)}"
                        child = AgentFactory.create_offspring(father, agent, child_id, self.timestep)
                        # Clamp within world bounds
                        try:
                            cx, cy = child.position
                            cx = max(0, min(int(self.environment.width) - 1, int(cx)))
                            cy = max(0, min(int(self.environment.height) - 1, int(cy)))
                            child.position = (cx, cy)
                        except Exception:
                            pass
                        child.age = 0
                        child.birth_time = self.timestep
                        setattr(child, 'mother_id', agent.id)
                        setattr(child, 'father_id', father.id)
                        setattr(child, 'sex', 'male' if random.random() < 0.5 else 'female')

                        # Имя придумывают родители
                        name_source = father if random.random() < 0.5 else agent
                        setattr(child, 'display_name', name_source.invent_name())

                        # Family bonds: parent <-> child + inherit personality & skills
                        try:
                            from .agent import Personality
                            child.social.add_family(agent.id)
                            child.social.add_family(father.id)
                            agent.social.add_family(child.id)
                            father.social.add_family(child.id)
                            child.personality = Personality.inherit(agent.personality, father.personality)
                            child.skills = Skills.inherit(agent.skills, father.skills)
                        except Exception:
                            pass

                        # Записи в дневник родителей
                        try:
                            child_name = getattr(child, 'display_name', child.id)
                            agent.life_log.add(self.timestep, 'birth', f'Родился ребёнок: {child_name}')
                            father.life_log.add(self.timestep, 'birth', f'Родился ребёнок: {child_name}')
                        except Exception:
                            pass

                        self.agents[child.id] = child
                        self.learning_manager.register_agent(child)
                        self.total_births += 1

                        self.events_log.append({
                            'timestamp': self.timestep,
                            'type': 'birth',
                            'mother_id': agent.id,
                            'father_id': father.id,
                            'child_id': child.id,
                            'child_sex': getattr(child, 'sex', 'unknown'),
                            'child_name': getattr(child, 'display_name', child.id),
                        })

                    setattr(agent, 'pregnant', False)
                    setattr(agent, 'pregnancy_father_id', None)
                    setattr(agent, 'pregnancy_remaining', 0)
                    setattr(agent, '_just_gave_birth', True)  # флаг для эмоций
            
            # Логирование события
            self._log_agent_action(agent, action, result)

            if action == 'communicate' and result.data:
                self.events_log.append({
                    'timestamp': self.timestep,
                    'type': 'communication',
                    'speaker_id': agent.id,
                    'listener_id': result.data.get('listener_id'),
                    'token': result.data.get('token'),
                    'meaning': result.data.get('meaning'),
                    'success': result.success,
                })

            if action == 'mate' and result.data:
                self.events_log.append({
                    'timestamp': self.timestep,
                    'type': 'mate',
                    'success': result.success,
                    'mother_id': result.data.get('mother_id'),
                    'father_id': result.data.get('father_id'),
                    'pregnancy_remaining': result.data.get('pregnancy_remaining'),
                })

            if action == 'care' and result.data:
                self.events_log.append({
                    'timestamp': self.timestep,
                    'type': 'care',
                    'success': result.success,
                    'parent_id': agent.id,
                    'child_id': result.data.get('child_id'),
                    'token': result.data.get('token'),
                    'meaning': result.data.get('meaning'),
                })

            if action == 'share' and result.success and result.data:
                self.events_log.append({
                    'timestamp': self.timestep,
                    'type': 'social',
                    'actor_id': agent.id,
                    'target_id': result.data.get('target_id'),
                    'amount': result.data.get('amount'),
                })

            # ── Emotions, social, thoughts ──────────────────────────
            try:
                emo = agent.emotional_state
                soc = agent.social
                pers = agent.personality

                # Decay emotions & relationships each tick
                emo.decay()
                soc.decay()

                # Need-driven emotions
                if agent.hunger > 0.75:
                    emo.add('fear', 0.06)
                    emo.add('anger', 0.03)
                if agent.thirst > 0.75:
                    emo.add('fear', 0.08)
                if agent.health < 0.4:
                    emo.add('fear', 0.10)
                if agent.energy > 0.6 and agent.hunger < 0.3 and agent.thirst < 0.3:
                    emo.add('contentment', 0.08)
                    emo.add('happiness', 0.04)

                # Loneliness: no close agents for a while
                if not close_agents:
                    emo.add('loneliness', 0.02 * pers.sociability)
                else:
                    emo.add('loneliness', -0.05)
                    # Met a friend?
                    for other in close_agents:
                        trust = soc.get_trust(other.id)
                        if trust > 0.3:
                            emo.add('happiness', 0.03)

                # Action-based emotions
                if result.success:
                    if action == 'consume':
                        emo.add('happiness', 0.12)
                        emo.add('contentment', 0.08)
                    elif action == 'drink':
                        emo.add('contentment', 0.10)
                    elif action == 'combine':
                        emo.add('pride', 0.20)
                        emo.add('curiosity', 0.10)
                        emo.add('happiness', 0.10)
                    elif action == 'communicate':
                        emo.add('happiness', 0.06)
                        listener_id = result.data.get('listener_id') if result.data else None
                        if listener_id:
                            soc.add_interaction(listener_id, 0.05)
                            # Слушатель тоже получает XP — общение двустороннее (Меткалф)
                            if result.success:
                                _listener_ag = self.agents.get(listener_id)
                                if _listener_ag and hasattr(_listener_ag, 'social'):
                                    _l_n = sum(1 for t in _listener_ag.social.relationships.values() if t > 0.3)
                                    _l_xp = (0.004 + 0.003 * getattr(_listener_ag.genes, 'intelligence', 0.5)) * 0.5
                                    _l_mult = 1.0 + min(1.5, (_l_n / 8.0) ** 2)
                                    _listener_ag.skills.add_xp('communication', _l_xp * _l_mult)
                    elif action == 'mate':
                        emo.add('happiness', 0.15)
                        partner_id = result.data.get('father_id') or result.data.get('mother_id') if result.data else None
                        if partner_id and partner_id != agent.id:
                            soc.add_interaction(partner_id, 0.15)
                            soc.add_family(partner_id)
                    elif action == 'care':
                        emo.add('contentment', 0.10)
                        emo.add('happiness', 0.06)
                        child_id = result.data.get('child_id') if result.data else None
                        if child_id:
                            soc.add_interaction(child_id, 0.08)
                    elif action == 'gather':
                        emo.add('contentment', 0.03)
                    elif action == 'sleep':
                        emo.add('contentment', 0.05)
                    elif action == 'attack':
                        emo.add('pride', 0.08)
                        emo.add('anger', -0.05)
                    elif action == 'collect_trade':
                        emo.add('happiness', 0.05)
                        _trade_owner = result.data.get('owner_id') if result.data else None
                        if _trade_owner:
                            soc.add_interaction(_trade_owner, 0.08)
                            # Владелец тоже получает trust
                            _owner_ag = self.agents.get(_trade_owner)
                            if _owner_ag and hasattr(_owner_ag, 'social'):
                                _owner_ag.social.add_interaction(agent.id, 0.08)
                    elif action == 'deposit':
                        emo.add('contentment', 0.04)
                    elif action in ('build', 'upgrade'):
                        emo.add('pride', 0.06)
                        emo.add('contentment', 0.04)
                else:
                    if action == 'combine':
                        emo.add('anger', 0.06)
                    elif action == 'attack':
                        emo.add('fear', 0.05)
                        emo.add('anger', 0.08)
                    elif action in ('gather', 'consume', 'drink'):
                        emo.add('anger', 0.03)

                # Curiosity from exploration
                if action == 'move':
                    emo.add('curiosity', 0.02 * pers.curiosity)

                # Birth: family bonds + emotions (happiness при рождении ребёнка)
                if getattr(agent, '_just_gave_birth', False):
                    emo.add('happiness', 0.25)
                    emo.add('pride', 0.15)
                    setattr(agent, '_just_gave_birth', False)

                # Generate thought for UI (every 5 ticks to avoid spam)
                if self.timestep % 5 == 0:
                    thought = generate_thought(agent)
                    agent.current_thought = thought

                # Update mood label
                agent.last_mood = emo.mood_ru()

                # ── Skills XP + разблокировка ──────────────────────────
                skill_name = ACTION_TO_SKILL.get(action)
                if skill_name and result.success:
                    xp = 0.004 + 0.003 * getattr(agent.genes, 'intelligence', 0.5)
                    _rb = getattr(agent, 'research_bonuses', {})
                    _xp_mult = 1.0 + _rb.get('xp_mult_all', 0.0) + _rb.get(f'xp_mult_{skill_name}', 0.0)
                    # Закон Меткалфа: ценность коммуникации растёт с n² связей
                    if skill_name == 'communication' and hasattr(agent, 'social'):
                        n_conn = sum(1 for t in agent.social.relationships.values() if t > 0.3)
                        _xp_mult *= 1.0 + min(1.5, (n_conn / 8.0) ** 2)
                    # Закон Йеркса-Додсона: обучение эффективнее при умеренном возбуждении (~0.35)
                    _arousal = (
                        getattr(agent, 'hunger', 0.0) +
                        getattr(agent, 'thirst', 0.0) +
                        getattr(agent, 'sleepiness', 0.0)
                    ) / 3.0
                    _xp_mult *= max(0.5, 1.0 - 2.0 * (_arousal - 0.35) ** 2)
                    val_before = agent.skills.get(skill_name)
                    agent.skills.add_xp(skill_name, xp * _xp_mult)
                    val_after = agent.skills.get(skill_name)
                    # Уведомление о разблокировке нового умения
                    _SKILL_UNLOCKS = {
                        'survival': [
                            (0.2, '🌿 Выживание lv3 — разблокировано: сбор трав'),
                            (0.3, '🌿 Выживание lv4 — разблокировано: лечение травами'),
                        ],
                        'crafting': [
                            (0.1, '🧶 Крафт lv2 — разблокировано: выделка кожи'),
                            (0.2, '🔧 Крафт lv3 — разблокировано: ремонт инструментов, рецепты Tier 2'),
                            (0.4, '🏠 Крафт lv5 — разблокировано: убежище, плавка, рецепты Tier 3'),
                            (0.6, '⚔️ Крафт lv7 — разблокировано: металлические орудия, рецепты Tier 4'),
                        ],
                        'cooking': [
                            (0.1, '🍳 Кулинария lv2 — разблокировано: готовка еды'),
                        ],
                        'gathering': [
                            (0.2, '🐟 Собирательство lv3 — разблокировано: рыбалка'),
                        ],
                    }
                    for threshold, msg in _SKILL_UNLOCKS.get(skill_name, []):
                        if val_before < threshold <= val_after:
                            try:
                                agent.life_log.add(self.timestep, 'skill_up', msg, icon='⬆️')
                            except Exception:
                                pass

                # Track visited cells for explorer achievement
                if action == 'move' and result.success:
                    agent.track_visit(agent.position)

                # ── Life log (история жизни с эмоциональной привязкой) ──
                log = agent.life_log
                ach = agent.achievements
                ts = self.timestep

                # Обновляем эмоциональный контекст дневника
                log.set_emotional_context(
                    getattr(agent, 'last_mood', None),
                    emo.dominant() if hasattr(emo, 'dominant') else None,
                )

                # ── Еда / питьё ──
                if action == 'consume' and result.success:
                    log.add(ts, 'eat', 'Утолил голод — поел.')
                if action == 'drink' and result.success:
                    log.add(ts, 'drink', 'Утолил жажду — попил воды.')

                # ── Сбор ──
                if action == 'gather' and result.success:
                    if ach.unlock('first_gather', ts):
                        log.add(ts, 'achievement', 'Первая добыча! Собрал первый объект.')
                    elif ts % 50 == 0:
                        log.add(ts, 'gather', 'Нашёл и собрал ресурсы.', icon='🌿')

                # ── Крафт ──
                if action == 'combine' and result.success:
                    if ach.unlock('first_craft', ts):
                        log.add(ts, 'achievement', 'Изобретатель! Создал первый инструмент.')
                    else:
                        log.add(ts, 'craft', 'Создал новый инструмент.')

                # ── Бой ──
                if action == 'attack':
                    if result.success:
                        log.add(ts, 'fight', 'Одержал победу в схватке!')
                    else:
                        log.add(ts, 'fight', 'Проиграл бой... Получил раны.')

                # ── Общение ──
                if action == 'communicate' and result.success:
                    listener_id = result.data.get('listener_id') if result.data else None
                    if listener_id:
                        name = self._agent_display_name(listener_id)
                        trust = soc.get_trust(listener_id)
                        if trust > 0.5:
                            log.add(ts, 'social', f'Пообщался с другом {name}.')
                        elif trust > 0.0:
                            log.add(ts, 'social', f'Поговорил с {name}.')

                # ── Любовь / размножение ──
                if action == 'mate' and result.success:
                    partner_id = (result.data.get('father_id') or result.data.get('mother_id')) if result.data else None
                    name = self._agent_display_name(partner_id) if partner_id else '?'
                    log.add(ts, 'love', f'Нашёл пару: {name}.')
                    if getattr(agent, 'pregnant', False):
                        if ach.unlock('first_child', ts):
                            log.add(ts, 'achievement', 'Родитель! Скоро появится ребёнок.')

                # ── Забота ──
                if action == 'care' and result.success:
                    child_id = result.data.get('child_id') if result.data else None
                    name = self._agent_display_name(child_id) if child_id else 'ребёнок'
                    log.add(ts, 'family', f'Позаботился о {name}.')

                # ── Сон ──
                if action == 'sleep' and result.success:
                    log.add(ts, 'sleep', 'Лёг отдохнуть и набраться сил.')

                # ── Опасность: здоровье/голод/жажда ──
                if agent.health < 0.25 and not getattr(agent, '_log_danger_hp', False):
                    log.add(ts, 'danger', 'Здоровье критически низкое! Нужна помощь...')
                    agent._log_danger_hp = True
                elif agent.health >= 0.4:
                    agent._log_danger_hp = False

                if agent.hunger > 0.85 and not getattr(agent, '_log_danger_hunger', False):
                    log.add(ts, 'danger', 'Ужасно голоден... Нужно найти еду!')
                    agent._log_danger_hunger = True
                elif agent.hunger < 0.5:
                    agent._log_danger_hunger = False

                if getattr(agent, 'thirst', 0) > 0.85 and not getattr(agent, '_log_danger_thirst', False):
                    log.add(ts, 'danger', 'Мучает жажда... Нужна вода!')
                    agent._log_danger_thirst = True
                elif getattr(agent, 'thirst', 0) < 0.5:
                    agent._log_danger_thirst = False

                # ── Плавание ──
                if getattr(agent, 'is_swimming', False):
                    wt = int(getattr(agent, 'water_ticks', 0))
                    if wt == 3:
                        log.add(ts, 'swim', 'Начал плыть по воде...')
                    elif wt > self.drowning_grace_steps:
                        log.add(ts, 'danger', 'Тону! Не могу выбраться из воды!')

                # ── Смена настроения (значительная) ──
                prev_mood = getattr(agent, '_prev_mood_for_log', None)
                cur_mood = getattr(agent, 'last_mood', None)
                if prev_mood and cur_mood and prev_mood != cur_mood and ts % 10 == 0:
                    log.add(ts, 'mood', f'Настроение изменилось: {cur_mood}.')
                agent._prev_mood_for_log = cur_mood

                # ── Новая дружба ──
                for other in close_agents:
                    trust = soc.get_trust(other.id)
                    prev_key = f'_log_friend_{other.id}'
                    if trust > 0.4 and not getattr(agent, prev_key, False):
                        name = self._agent_display_name(other.id)
                        log.add(ts, 'social', f'Подружился с {name}!')
                        setattr(agent, prev_key, True)

                # ── Прокачка навыка ──
                if skill_name and result.success:
                    sk_obj = agent.skills
                    new_lv = sk_obj.level(skill_name)
                    prev_lv_key = f'_log_sk_lv_{skill_name}'
                    prev_lv = getattr(agent, prev_lv_key, 1)
                    if new_lv > prev_lv:
                        from .agent import SKILL_RU
                        log.add(ts, 'skill_up', f'Навык «{SKILL_RU.get(skill_name, skill_name)}» повысился до lv{new_lv}!')
                        setattr(agent, prev_lv_key, new_lv)

                # ── Долгожитель ──
                if agent.age >= 5000:
                    if ach.unlock('elder', ts):
                        log.add(ts, 'achievement', 'Долгожитель! Прожил 5000 тиков.')

                # ── Мастерство навыков ──
                for sk, ach_id in [('hunting', 'master_hunter'), ('crafting', 'master_crafter'),
                                    ('gathering', 'master_gatherer'), ('survival', 'survivor'),
                                    ('communication', 'communicator')]:
                    if agent.skills.level(sk) >= 7:
                        if ach.unlock(ach_id, ts):
                            from .agent import SKILL_RU
                            log.add(ts, 'achievement', f'Мастерство: {SKILL_RU.get(sk, sk)} lv7!')

                # ── Социальные достижения ──
                friends_count = len([1 for _, t in agent.social.relationships.items() if t > 0.3])
                if friends_count >= 5:
                    if ach.unlock('social_butterfly', ts):
                        log.add(ts, 'achievement', 'Душа компании! 5+ друзей.')

                family_count = len(agent.social.family)
                if family_count >= 3:
                    if ach.unlock('family_person', ts):
                        log.add(ts, 'achievement', 'Семьянин! 3+ членов семьи.')

                # ── Путешественник ──
                if agent.visited_cells >= 100:
                    if ach.unlock('explorer', ts):
                        log.add(ts, 'achievement', 'Путешественник! Посетил 100+ клеток.')

            except Exception:
                pass
    
    def _agent_display_name(self, agent_id: str) -> str:
        """Возвращает display_name агента по id, или сам id."""
        a = self.agents.get(agent_id)
        if a:
            return getattr(a, 'display_name', a.id)
        return str(agent_id)[:12]

    def _handle_agent_death(self, agent: Agent):
        """Обрабатывает смерть агента"""
        cause = 'unknown'
        if bool(getattr(agent, 'drowned', False)):
            cause = 'drowning'
        elif agent.hunger >= 1.0:
            cause = 'starvation'
        elif getattr(agent, 'thirst', 0.0) >= 1.0:
            cause = 'dehydration'
        elif agent.health <= 0:
            cause = 'health_collapse'
        elif agent.age >= agent.max_age:
            cause = 'old_age'
        elif agent.energy <= 0.0:
            cause = 'exhaustion'

        self.events_log.append({
            'timestamp': self.timestep,
            'type': 'agent_death',
            'agent_id': agent.id,
            'age': agent.age,
            'cause': cause,
            'hunger': agent.hunger,
            'thirst': float(getattr(agent, 'thirst', 0.0)),
            'sleepiness': float(getattr(agent, 'sleepiness', 0.0)),
            'energy': agent.energy,
            'health': agent.health,
            'max_age': agent.max_age,
        })
        
        # Запись смерти в дневник самого агента
        cause_ru = {'drowning': 'утонул', 'starvation': 'от голода', 'dehydration': 'от жажды',
                     'health_collapse': 'от ран', 'old_age': 'от старости', 'exhaustion': 'от истощения'}
        try:
            agent.life_log.add(self.timestep, 'death',
                               f'Умер {cause_ru.get(cause, "по неизвестной причине")}. Прожил {agent.age} тиков.')
        except Exception:
            pass

        # Grief: nearby agents feel sadness, especially family
        agent_name = getattr(agent, 'display_name', agent.id)
        try:
            for other in list(self.agents.values()):
                if other.id == agent.id:
                    continue
                dx = abs(other.position[0] - agent.position[0])
                dy = abs(other.position[1] - agent.position[1])
                if dx <= 3 and dy <= 3:
                    grief_amt = 0.08
                    is_family = agent.id in other.social.family
                    is_friend = other.social.get_trust(agent.id) > 0.2
                    if is_family:
                        grief_amt = 0.35
                        other.life_log.add(self.timestep, 'death',
                                           f'Потерял близкого: {agent_name} умер... Горе.')
                    elif is_friend:
                        grief_amt = 0.15
                        other.life_log.add(self.timestep, 'death',
                                           f'Знакомый {agent_name} погиб... Печально.')
                    other.emotional_state.add('grief', grief_amt)
                    other.emotional_state.add('fear', grief_amt * 0.3)
        except Exception:
            pass

        # Сохраняем мёртвого агента для отображения надгробия на карте
        cause_ru_map = {
            'drowning': 'утонул',
            'starvation': 'голод',
            'dehydration': 'жажда',
            'health_collapse': 'здоровье',
            'old_age': 'старость',
            'exhaustion': 'истощение',
            'unknown': 'неизвестно',
        }
        self.dead_agents.append({
            'id': agent.id,
            'name': getattr(agent, 'display_name', agent.id),
            'sex': getattr(agent, 'sex', 'unknown'),
            'owner_username': getattr(agent, 'owner_username', None),
            'owner_uid': getattr(agent, 'owner_uid', None),
            'x': int(agent.position[0]),
            'y': int(agent.position[1]),
            'age': int(agent.age),
            'cause': cause,
            'cause_ru': cause_ru_map.get(cause, cause),
            'died_at': int(self.timestep),
            'personality_ru': agent.personality.describe_ru() if hasattr(agent, 'personality') and hasattr(agent.personality, 'describe_ru') else None,
            'personality': agent.personality.to_dict() if hasattr(agent, 'personality') and hasattr(agent.personality, 'to_dict') else None,
            # Навыки для сохранения в dynasty_skills
            'skills': {s: agent.skills.get(s) for s in ['gathering','crafting','hunting','cooking','communication','survival']}
            if hasattr(agent, 'skills') else {},
        })
        # Ограничиваем количество надгробий (не больше 50)
        if len(self.dead_agents) > 50:
            self.dead_agents = self.dead_agents[-50:]

        # Удаляем агента из систем
        self.learning_manager.unregister_agent(agent.id)
        del self.agents[agent.id]
        
        self.total_deaths += 1
    
    def _process_evolution(self):
        """Обрабатывает эволюционные процессы"""
        alive_agents = list(self.agents.values())

        if len(alive_agents) < 2:
            return

        ga_births_before = self.evolution_manager.genetic_algorithm.total_births

        # Эволюция популяции
        new_agents = self.evolution_manager.evolve(
            alive_agents, self.environment, self.timestep
        )

        # Обновление агентов
        self.agents.clear()
        for agent in new_agents:
            self.agents[agent.id] = agent
            # Регистрируем в системе обучения, если новый агент
            if agent.id not in [dm.agent_id for dm in self.learning_manager.learners.values()]:
                self.learning_manager.register_agent(agent)

        # Прибавляем только дельту GA-рождений, не перезаписывая естественные
        self.total_births += self.evolution_manager.genetic_algorithm.total_births - ga_births_before
    
    def _update_campfires_and_bushes(self):
        """Обновляет состояние костров (топливо, пожар) и ягодных кустов (созревание)."""
        ts = self.timestep
        to_remove = []

        for obj_id, obj in list(self.environment.objects.items()):

            # ── Костёр ──────────────────────────────────────────────────────
            if obj.type == 'campfire':
                fuel = int(getattr(obj, 'fuel_ticks', 500))
                fuel -= 1
                if fuel <= 0:
                    to_remove.append(obj_id)
                    continue
                setattr(obj, 'fuel_ticks', fuel)

                # Распространение огня: раз в ~250 тиков на костёр
                if random.random() < 0.004:
                    x, y = obj.position
                    for dx in (-1, 0, 1):
                        for dy in (-1, 0, 1):
                            if dx == 0 and dy == 0:
                                continue
                            npos = (x + dx, y + dy)
                            if self.environment.is_water(npos):
                                continue
                            neighbors = self.environment.get_objects_at_position(npos)
                            for n in neighbors:
                                if n.type in ('wood', 'plant', 'berry_bush') and \
                                        getattr(n, 'flammability', 0) > 0.4 and \
                                        random.random() < getattr(n, 'flammability', 0) * 0.1:
                                    self.environment.detach_object_from_world(n.id)
                                    new_id = f"fire_{ts}_{dx}_{dy}_{obj_id[:6]}"
                                    try:
                                        new_fire = ObjectFactory.create_object('campfire', npos, new_id, ts)
                                        setattr(new_fire, 'fuel_ticks', 300)
                                        self.environment.add_object(new_fire)
                                    except Exception:
                                        pass
                                    break

            # ── Ягодный куст ────────────────────────────────────────────────
            elif obj.type == 'berry_bush':
                if not getattr(obj, 'ripe', False):
                    planted_at = int(getattr(obj, 'planted_at', ts))
                    if ts - planted_at >= 100:
                        setattr(obj, 'ripe', True)
                        obj.nutrition = 0.8  # теперь edible

        # ── Деградация зданий ────────────────────────────────────────────
        for obj_id, obj in list(self.environment.objects.items()):
            bt = getattr(obj, 'building_type', None)
            if bt and bt in BUILDING_RECIPES:
                dur = getattr(obj, 'durability', 1.0)
                # Деградация: 0.0005 за тик → ~2000 тиков до разрушения
                dur -= 0.0005
                obj.durability = dur
                if dur <= 0:
                    to_remove.append(obj_id)
                    # Уведомление в лог владельца
                    try:
                        _oid = getattr(obj, 'building_owner_id', None)
                        if _oid and _oid in self.agents:
                            _label = BUILDING_RECIPES[bt].get('label', bt)
                            self.agents[_oid].life_log.add(
                                ts, 'alert',
                                f'⚠️ {_label} разрушен(а)!', icon='💥')
                    except Exception:
                        pass
                elif dur < 0.3:
                    # Предупреждение (раз в 100 тиков)
                    if ts % 100 == 0:
                        try:
                            _oid = getattr(obj, 'building_owner_id', None)
                            if _oid and _oid in self.agents:
                                _label = BUILDING_RECIPES[bt].get('label', bt)
                                _pct = int(dur * 100)
                                self.agents[_oid].life_log.add(
                                    ts, 'alert',
                                    f'⚠️ {_label} повреждён ({_pct}%)!', icon='🔧')
                        except Exception:
                            pass

        for obj_id in to_remove:
            try:
                self.environment.remove_object(obj_id)
            except Exception:
                pass

    def _collect_metrics(self):
        """Собирает метрики симуляции"""
        agents_list = list(self.agents.values())
        
        metrics = self.metrics_calculator.calculate_metrics(
            agents_list, self.environment, self.timestep
        )
        
        # Обновляем общие счетчики
        metrics.total_births = self.total_births
        metrics.total_deaths = self.total_deaths
        
        self.metrics_history.append(metrics)
    
    def _log_agent_action(self, agent: Agent, action: str, result: ActionResult):
        """Логирует действие агента"""
        setattr(agent, 'last_action', action)
        setattr(agent, 'last_action_success', result.success)
        setattr(agent, 'last_action_reward', result.reward)
        self.events_log.append({
            'timestamp': self.timestep,
            'type': 'agent_action',
            'agent_id': agent.id,
            'action': action,
            'success': result.success,
            'reward': result.reward,
            'energy_cost': result.energy_cost,
            'position': agent.position
        })
    
    def get_statistics(self) -> Dict[str, Any]:
        """Возвращает текущую статистику"""
        stats = self.environment.get_statistics()
        
        # Добавляем статистику агентов
        if self.agents:
            agents_list = list(self.agents.values())
            stats.update({
                'alive_agents': len(agents_list),
                'average_age': sum(agent.age for agent in agents_list) / len(agents_list),
                'average_health': sum(agent.health for agent in agents_list) / len(agents_list),
                'average_energy': sum(agent.energy for agent in agents_list) / len(agents_list),
                'total_discoveries': sum(len(agent.discoveries_made) for agent in agents_list)
            })
        else:
            stats.update({
                'alive_agents': 0,
                'average_age': 0,
                'average_health': 0,
                'average_energy': 0,
                'total_discoveries': 0
            })
        
        # Эволюционная статистика
        if hasattr(self, 'evolution_manager'):
            evolution_stats = self.evolution_manager.get_evolution_summary()
            stats.update(evolution_stats)
        
        # Статистика обучения
        if hasattr(self, 'learning_manager'):
            learning_stats = self.learning_manager.get_global_stats()
            stats.update(learning_stats)
        
        return stats
    
    def is_finished(self) -> bool:
        """Проверяет, завершена ли симуляция"""
        if len(self.agents) == 0:
            return True
        if self.max_steps is None:
            return False
        return self.timestep >= self.max_steps
    
    def get_final_report(self) -> Dict[str, Any]:
        """Генерирует финальный отчет"""
        if not self.metrics_history:
            return {'error': 'No metrics data available'}
        
        # Анализ метрик
        from ..utils.metrics import MetricsAnalyzer
        analyzer = MetricsAnalyzer(self.metrics_history)
        
        report = {
            'simulation_info': {
                'duration': self.timestep,
                'max_steps': self.max_steps,
                'final_population': len(self.agents),
                'total_births': self.total_births,
                'total_deaths': self.total_deaths,
                'config': self.config
            },
            'technology_analysis': analyzer.analyze_technology_progression(),
            'evolution_analysis': analyzer.analyze_evolution_patterns(),
            'cultural_analysis': analyzer.analyze_cultural_transmission(),
            'insights': analyzer.generate_insights(),
            'final_metrics': self.metrics_history[-1].to_dict() if self.metrics_history else {},
            'summary_statistics': self.metrics_calculator.calculate_summary_statistics()
        }
        
        return report


class Simulation:
    """Основной класс симуляции"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.state: Optional[SimulationState] = None
        self.start_time = None
        self.end_time = None
    
    def initialize(self):
        """Инициализирует симуляцию"""
        self.state = SimulationState(self.config)
        self.start_time = time.time()
        
        print(f"Симуляция инициализирована:")
        print(f"- Размер мира: {self.config['simulation']['world']['width']}x{self.config['simulation']['world']['height']}")
        print(f"- Начальная популяция: {self.config['simulation']['agents']['initial_population']}")
        print(f"- Максимальная популяция: {self.config['simulation']['agents']['max_population']}")
        print(f"- Длительность: {self.config['experiment']['duration']} шагов")
    
    def run(self, max_steps: Optional[int] = None, verbose: bool = True) -> SimulationState:
        """Запускает симуляцию"""
        if not self.state:
            self.initialize()
        
        if max_steps:
            self.state.max_steps = max_steps
        
        step_count = 0
        
        try:
            while not self.state.is_finished():
                if not self.state.step():
                    break
                
                step_count += 1
                
                if verbose and step_count % 100 == 0:
                    stats = self.state.get_statistics()
                    print(f"Шаг {self.state.timestep}: "
                          f"Агентов: {stats['alive_agents']}, "
                          f"Открытий: {stats['total_discoveries']}, "
                          f"Инструментов: {stats.get('total_tools', 0)}")
        
        except KeyboardInterrupt:
            print(f"\nСимуляция прервана на шаге {self.state.timestep}")
        
        except Exception as e:
            print(f"Ошибка в симуляции: {e}")
            raise
        
        finally:
            self.end_time = time.time()
        
        if verbose:
            self._print_summary()
        
        return self.state
    
    def _print_summary(self):
        """Выводит сводку результатов"""
        if not self.state:
            return
        
        duration = self.end_time - self.start_time if self.end_time else 0
        stats = self.state.get_statistics()
        
        print("\n" + "="*50)
        print("СИМУЛЯЦИЯ ЗАВЕРШЕНА")
        print("="*50)
        print(f"Длительность: {duration:.2f} секунд")
        print(f"Выполнено шагов: {self.state.timestep}")
        print(f"Выжившие агенты: {stats.get('alive_agents', 0)}")
        print(f"Всего рождено: {stats.get('total_births', 0)}")
        print(f"Всего умерло: {stats.get('total_deaths', 0)}")
        print(f"Общие открытия: {stats.get('total_discoveries', 0)}")
        print(f"Всего инструментов: {stats.get('total_tools', 0)}")
        print(f"Уникальных типов инструментов: {stats.get('unique_tool_types', 0)}")
        print(f"Поколений: {stats.get('generation', 0)}")
        
        if stats.get('genetic_diversity', 0) > 0:
            print(f"Генетическое разнообразие: {stats['genetic_diversity']:.3f}")
        
        print("="*50)
    
    def save_state(self, filepath: str):
        """Сохраняет состояние симуляции"""
        if not self.state:
            raise ValueError("No simulation state to save")
        
        import pickle
        
        save_data = {
            'state': self.state,
            'config': self.config,
            'start_time': self.start_time,
            'end_time': self.end_time
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(save_data, f)
    
    @classmethod
    def load_state(cls, filepath: str) -> 'Simulation':
        """Загружает состояние симуляции"""
        import pickle
        
        with open(filepath, 'rb') as f:
            save_data = pickle.load(f)
        
        simulation = cls(save_data['config'])
        simulation.state = save_data['state']
        simulation.start_time = save_data['start_time']
        simulation.end_time = save_data['end_time']
        
        return simulation
