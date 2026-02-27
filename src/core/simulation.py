"""
Core simulation module - main simulation controller
"""

from typing import Dict, List, Any, Optional
import random
import time

from .environment import Environment, EnvironmentConfig
from .agent import Agent, AgentFactory
from .agent_actions import ActionExecutor, ActionResult
from .tools import ToolLibrary
from ..learning.q_learning import LearningManager
from ..evolution.genetics import EvolutionManager
from ..utils.metrics import MetricsCalculator, SimulationMetrics


class SimulationState:
    """Состояние симуляции"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.timestep = 0
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

            # mate only when a valid close partner exists
            if mate_candidates:
                available_actions.append('mate')

            # care when close child exists
            if any(getattr(a, 'is_child', lambda: False)() for a in close_agents):
                available_actions.append('care')
            
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

            # Инстинкт жажды: искать/пить воду
            thirsty_self = getattr(agent, 'thirst', 0.0) > 0.6
            water_here = False
            if thirsty_self:
                try:
                    water_here = any(o.type == 'water' for o in self.environment.get_objects_at_position(agent.position))
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
                and tool_material_in_inventory < 2
                and getattr(agent, 'energy', 0.0) > 0.25
                and getattr(agent, 'hunger', 0.0) < 0.65
                and getattr(agent, 'thirst', 0.0) < 0.65
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

            # Выбор действия
            decision_maker = self.learning_manager.get_decision_maker(agent_id)
            if decision_maker:
                if hungry_child and 'care' in available_actions:
                    action = 'care'
                # Размножение: когда всё спокойно и рядом подходящий партнёр
                elif mate_candidates and 'mate' in available_actions:
                    # Не пытаемся размножаться при критических нуждах
                    if not (getattr(agent, 'hunger', 0.0) > 0.75 or getattr(agent, 'thirst', 0.0) > 0.75 or getattr(agent, 'sleepiness', 0.0) > 0.85):
                        base = 0.002
                        base += 0.010 * float(getattr(agent.genes, 'social_tendency', 0.5))
                        base += 0.006 * float(getattr(agent.genes, 'exploration_bias', 0.5))
                        if random.random() < base:
                            action = 'mate'
                        else:
                            action = decision_maker.select_action(local_env, available_actions)
                    else:
                        action = decision_maker.select_action(local_env, available_actions)
                # Сон: ночью при высокой сонливости
                elif (not getattr(self.environment, 'is_daytime', True)) and getattr(agent, 'sleepiness', 0.0) > 0.65 and 'sleep' in available_actions:
                    # Не спим, если критический голод/жажда
                    if not (getattr(agent, 'hunger', 0.0) > 0.85 or getattr(agent, 'thirst', 0.0) > 0.85):
                        action = 'sleep'
                    else:
                        action = decision_maker.select_action(local_env, available_actions)
                # Сбор материалов для инструментов
                elif wants_tool_materials and tool_material_visible_here and 'gather' in available_actions:
                    action = 'gather'
                # Создание инструментов: периодически пробуем combine, когда всё спокойно
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
                action = "rest"  # Действие по умолчанию
            
            # Исполнение действия
            if action in ('communicate', 'mate', 'care'):
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

                        self.agents[child.id] = child
                        self.learning_manager.register_agent(child)

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
    
    def _handle_agent_death(self, agent: Agent):
        """Обрабатывает смерть агента"""
        cause = 'unknown'
        if agent.hunger >= 1.0:
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
        
        # Удаляем агента из систем
        self.learning_manager.unregister_agent(agent.id)
        del self.agents[agent.id]
        
        self.total_deaths += 1
    
    def _process_evolution(self):
        """Обрабатывает эволюционные процессы"""
        alive_agents = list(self.agents.values())
        
        if len(alive_agents) < 2:
            return
        
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
        
        # Обновляем счетчики
        self.total_births = self.evolution_manager.genetic_algorithm.total_births
    
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
