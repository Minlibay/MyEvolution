"""
Environment module - defines the 2D grid environment
"""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Any, Optional
import random
import numpy as np

from .objects import Object, ObjectFactory
from .tools import Tool, ToolLibrary


@dataclass
class EnvironmentConfig:
    """Конфигурация среды"""
    width: int = 100
    height: int = 100
    seed: int = 42
    resource_spawn_rate: float = 0.1
    max_objects_per_cell: int = 10
    season_length: int = 1000
    initial_resource_density: float = 0.05

    # Вода (источники)
    lake_count: int = 2
    lake_radius: int = 6

    # Сутки (1 шаг = 1 сек)
    day_length_seconds: int = 24 * 60 * 60
    day_start_hour: int = 6
    night_start_hour: int = 18


class Environment:
    """Моделирование 2D среды"""
    
    def __init__(self, config: EnvironmentConfig):
        self.config = config
        self.width = config.width
        self.height = config.height
        self.seed = config.seed
        
        # Инициализация генератора случайных чисел
        self.random = random.Random(config.seed)
        self.np_random = np.random.RandomState(config.seed)
        
        # Пространство
        self.grid: Dict[Tuple[int, int], List[str]] = {}

        # Рельеф / биомы
        self.terrain = np.zeros((self.width, self.height), dtype=np.int8)  # 0=land, 1=water
        
        # Объекты в среде
        self.objects: Dict[str, Object] = {}
        self.tools: Dict[str, Tool] = {}
        
        # Параметры среды
        self.season: int = 0  # 0-3 (весна, лето, осень, зима)
        self.temperature: float = 20.0  # °C
        self.resource_abundance: float = 1.0
        
        # Библиотека инструментов
        self.tool_library = ToolLibrary()
        
        # Счетчик времени
        self.timestep: int = 0

        # Сутки
        self.day_count: int = 0
        self.time_seconds: int = 0
        self.hour: int = 0
        self.minute: int = 0
        self.is_daytime: bool = True
        
        # Инициализация начальных ресурсов
        self._initialize_water_sources()
        self._initialize_resources()

    def _initialize_water_sources(self):
        """Создает несколько "озер" как постоянные источники воды."""
        count = max(0, int(getattr(self.config, 'lake_count', 0)))
        radius = max(1, int(getattr(self.config, 'lake_radius', 6)))
        for i in range(count):
            cx = self.random.randint(0, self.width - 1)
            cy = self.random.randint(0, self.height - 1)
            for dx in range(-radius, radius + 1):
                for dy in range(-radius, radius + 1):
                    if dx * dx + dy * dy > radius * radius:
                        continue
                    x = cx + dx
                    y = cy + dy
                    if 0 <= x < self.width and 0 <= y < self.height:
                        self.terrain[x, y] = 1

        # Создаем водные объекты на водных клетках
        for x in range(self.width):
            for y in range(self.height):
                if self.terrain[x, y] == 1:
                    self._ensure_water_object_at((x, y))

    def _ensure_water_object_at(self, position: Tuple[int, int]):
        cell_ids = self.grid.get(position, [])
        for obj_id in cell_ids:
            obj = self.objects.get(obj_id)
            if obj is not None and obj.type == 'water':
                return

        if len(cell_ids) >= self.config.max_objects_per_cell:
            return

        obj_id = f"water_{self.timestep}_{self.random.randint(1000, 9999)}"
        obj = ObjectFactory.create_object('water', position, obj_id, self.timestep, self.season)
        self.objects[obj.id] = obj
        if position not in self.grid:
            self.grid[position] = []
        self.grid[position].append(obj.id)

    def is_water(self, position: Tuple[int, int]) -> bool:
        x, y = position
        if x < 0 or y < 0 or x >= self.width or y >= self.height:
            return False
        return self.terrain[x, y] == 1

    def is_near_water(self, position: Tuple[int, int], radius: int = 3) -> bool:
        x0, y0 = position
        for dx in range(-radius, radius + 1):
            for dy in range(-radius, radius + 1):
                x = x0 + dx
                y = y0 + dy
                if 0 <= x < self.width and 0 <= y < self.height and self.terrain[x, y] == 1:
                    return True
        return False
    
    def _initialize_resources(self):
        """Создает начальные ресурсы в среде"""
        num_initial_resources = int(
            self.width * self.height * self.config.initial_resource_density
        )
        
        for _ in range(num_initial_resources):
            self._spawn_random_resource(self.timestep)
    
    def update(self, timestep: int):
        """Обновляет состояние среды"""
        self.timestep = timestep

        # Обновление времени суток
        self._update_time_of_day(timestep)
        
        # Обновление сезона
        if timestep % self.config.season_length == 0 and timestep > 0:
            self.season = (self.season + 1) % 4
            self._update_seasonal_parameters()
        
        # Стохастическое появление ресурсов
        if self.random.random() < self.config.resource_spawn_rate:
            self._spawn_resources(timestep)
        
        # Естественная деградация объектов
        self._degrade_objects()
        
        # Обновление температуры
        self._update_temperature(timestep)

    def _update_time_of_day(self, timestep: int):
        """Обновляет сутки и время (1 шаг = 1 сек)."""
        day_len = max(1, int(self.config.day_length_seconds))
        self.day_count = int(timestep // day_len)
        self.time_seconds = int(timestep % day_len)

        self.hour = int(self.time_seconds // 3600)
        self.minute = int((self.time_seconds % 3600) // 60)

        self.is_daytime = (self.config.day_start_hour <= self.hour < self.config.night_start_hour)
    
    def _update_seasonal_parameters(self):
        """Обновляет параметры среды в зависимости от сезона"""
        season_effects = {
            0: {'resource_abundance': 1.2, 'temperature': 15.0},  # Весна
            1: {'resource_abundance': 1.5, 'temperature': 25.0},  # Лето
            2: {'resource_abundance': 1.0, 'temperature': 10.0},  # Осень
            3: {'resource_abundance': 0.5, 'temperature': -5.0}, # Зима
        }
        
        effects = season_effects[self.season]
        self.resource_abundance = effects['resource_abundance']
        self.temperature = effects['temperature']
    
    def _update_temperature(self, timestep: int):
        """Обновляет температуру с суточными колебаниями"""
        # Суточные колебания температуры
        day_len = max(1, int(self.config.day_length_seconds))
        day_progress = (timestep % day_len) / float(day_len)
        daily_variation = 5.0 * np.sin(2 * np.pi * day_progress)
        
        self.temperature += daily_variation * 0.1
    
    def _spawn_resources(self, timestep: int):
        """Создает новые ресурсы"""
        # Количество ресурсов зависит от сезона и обилия
        num_resources = self.np_random.poisson(
            self.resource_abundance * self.config.resource_spawn_rate * 2
        )
        
        for _ in range(num_resources):
            self._spawn_random_resource(timestep)
    
    def _spawn_random_resource(self, timestamp: int):
        """Создает один случайный ресурс"""
        # Случайная позиция (только суша)
        for _ in range(10):
            x = self.random.randint(0, self.width - 1)
            y = self.random.randint(0, self.height - 1)
            if self.terrain[x, y] == 0:
                break
        position = (x, y)

        if self.is_water(position):
            return
        
        # Проверка лимита объектов в клетке
        cell_objects = self.grid.get(position, [])
        if len(cell_objects) >= self.config.max_objects_per_cell:
            return
        
        # Выбор типа объекта (у воды больше еды)
        if self.is_near_water(position, radius=3):
            weights = {
                'plant': 0.35,
                'berry': 0.35,
                'wood': 0.15,
                'stone': 0.1,
                'bone': 0.03,
                'fiber': 0.02,
            }
            obj_type = self.random.choices(list(weights.keys()), weights=list(weights.values()))[0]
        else:
            obj_type = ObjectFactory.get_random_object_type(self.season)
        
        # Создание объекта
        obj_id = f"obj_{timestamp}_{self.random.randint(1000, 9999)}"
        obj = ObjectFactory.create_object(
            obj_type, position, obj_id, timestamp, self.season
        )
        
        # Добавление в среду
        self.objects[obj.id] = obj
        if position not in self.grid:
            self.grid[position] = []
        self.grid[position].append(obj.id)
    
    def _degrade_objects(self):
        """Деградация объектов со временем"""
        objects_to_remove = []
        
        for obj_id, obj in self.objects.items():
            if obj.type == 'water':
                continue
            # Скорость деградации зависит от сезона
            degradation_rate = 0.001
            if self.season == 3:  # Зима
                degradation_rate *= 2.0
            
            obj.degrade(degradation_rate)
            
            # Удаление полностью деградированных объектов
            if obj.quantity <= 0:
                objects_to_remove.append(obj_id)
        
        # Удаление объектов
        for obj_id in objects_to_remove:
            self.remove_object(obj_id)
    
    def get_local_environment(self, position: Tuple[int, int], 
                            radius: int = 2) -> Dict[str, Any]:
        """Возвращает локальную окружающую среду для агента"""
        local_objects = []
        local_tools = []
        
        for dx in range(-radius, radius + 1):
            for dy in range(-radius, radius + 1):
                x, y = position[0] + dx, position[1] + dy
                if 0 <= x < self.width and 0 <= y < self.height:
                    cell_objects = self.grid.get((x, y), [])
                    
                    for obj_id in cell_objects:
                        obj = self.objects.get(obj_id)
                        if obj:
                            local_objects.append(obj)
        
        # Поиск инструментов в локальной области
        for tool in self.tools.values():
            # Инструменты могут быть "в руках" у агентов или на земле
            # Пока simplistically считаем все инструменты доступными
            local_tools.append(tool)
        
        return {
            'objects': local_objects,
            'tools': local_tools,
            'position': position,
            'terrain': 'water' if self.is_water(position) else 'land',
            'season': self.season,
            'temperature': self.temperature,
            'resource_abundance': self.resource_abundance,
            'day': self.day_count,
            'hour': self.hour,
            'minute': self.minute,
            'is_daytime': self.is_daytime,
        }
    
    def add_object(self, obj: Object) -> bool:
        """Добавляет объект в среду"""
        if obj.position[0] < 0 or obj.position[0] >= self.width:
            return False
        if obj.position[1] < 0 or obj.position[1] >= self.height:
            return False
        
        # Проверка лимита объектов в клетке
        cell_objects = self.grid.get(obj.position, [])
        if len(cell_objects) >= self.config.max_objects_per_cell:
            return False
        
        self.objects[obj.id] = obj
        if obj.position not in self.grid:
            self.grid[obj.position] = []
        self.grid[obj.position].append(obj.id)
        
        return True
    
    def remove_object(self, obj_id: str) -> bool:
        """Удаляет объект из среды"""
        if obj_id not in self.objects:
            return False
        
        obj = self.objects[obj_id]
        position = obj.position
        
        # Удаление из сетки
        if position in self.grid:
            if obj_id in self.grid[position]:
                self.grid[position].remove(obj_id)
            
            # Удаление пустой клетки
            if not self.grid[position]:
                del self.grid[position]
        
        # Удаление из словаря объектов
        del self.objects[obj_id]
        
        return True

    def detach_object_from_world(self, obj_id: str) -> bool:
        """Убирает объект с карты (в инвентарь/вне мира), но не удаляет из self.objects."""
        obj = self.objects.get(obj_id)
        if obj is None:
            return False

        position = obj.position
        if position in self.grid and obj_id in self.grid[position]:
            self.grid[position].remove(obj_id)
            if not self.grid[position]:
                del self.grid[position]

        # Перемещаем объект за границы мира, чтобы он не визуализировался как лежащий на земле.
        obj.position = (-1, -1)
        return True
    
    def add_tool(self, tool: Tool):
        """Добавляет инструмент в среду"""
        self.tools[tool.id] = tool
        
        # Регистрация в библиотеке инструментов
        discovery_type = self.tool_library.register_tool(tool)
        
        return discovery_type
    
    def remove_tool(self, tool_id: str) -> bool:
        """Удаляет инструмент из среды"""
        if tool_id in self.tools:
            del self.tools[tool_id]
            return True
        return False
    
    def move_object(self, obj_id: str, new_position: Tuple[int, int]) -> bool:
        """Перемещает объект в новую позицию"""
        if obj_id not in self.objects:
            return False
        
        obj = self.objects[obj_id]
        old_position = obj.position
        
        # Проверка новой позиции
        if (new_position[0] < 0 or new_position[0] >= self.width or
            new_position[1] < 0 or new_position[1] >= self.height):
            return False
        
        # Проверка лимита объектов в новой клетке
        cell_objects = self.grid.get(new_position, [])
        if len(cell_objects) >= self.config.max_objects_per_cell:
            return False
        
        # Удаление из старой позиции
        if old_position in self.grid and obj_id in self.grid[old_position]:
            self.grid[old_position].remove(obj_id)
            if not self.grid[old_position]:
                del self.grid[old_position]
        
        # Добавление в новую позицию
        obj.position = new_position
        if new_position not in self.grid:
            self.grid[new_position] = []
        self.grid[new_position].append(obj_id)
        
        return True
    
    def get_objects_at_position(self, position: Tuple[int, int]) -> List[Object]:
        """Возвращает объекты в указанной позиции"""
        object_ids = self.grid.get(position, [])
        return [self.objects[obj_id] for obj_id in object_ids if obj_id in self.objects]
    
    def get_empty_positions(self, count: int) -> List[Tuple[int, int]]:
        """Возвращает список случайных пустых позиций"""
        empty_positions = []
        attempts = 0
        
        while len(empty_positions) < count and attempts < count * 10:
            x = self.random.randint(0, self.width - 1)
            y = self.random.randint(0, self.height - 1)
            position = (x, y)
            
            if position not in self.grid or len(self.grid[position]) == 0:
                if position not in empty_positions:
                    empty_positions.append(position)
            
            attempts += 1
        
        return empty_positions
    
    def get_statistics(self) -> Dict[str, Any]:
        """Возвращает статистику по среде"""
        total_objects = len(self.objects)
        object_types = {}
        
        for obj in self.objects.values():
            object_types[obj.type] = object_types.get(obj.type, 0) + 1
        
        total_tools = len(self.tools)
        tool_types = {}
        
        for tool in self.tools.values():
            tool_type = tool.get_tool_type()
            tool_types[tool_type] = tool_types.get(tool_type, 0) + 1
        
        occupied_cells = len(self.grid)
        total_cells = self.width * self.height
        occupation_rate = occupied_cells / total_cells

        water_cells = int(np.sum(self.terrain == 1))
        land_cells = int(total_cells - water_cells)
        
        return {
            'timestep': self.timestep,
            'day': self.day_count,
            'hour': self.hour,
            'minute': self.minute,
            'is_daytime': self.is_daytime,
            'season': self.season,
            'temperature': self.temperature,
            'resource_abundance': self.resource_abundance,
            'water_cells': water_cells,
            'land_cells': land_cells,
            'total_objects': total_objects,
            'object_types': object_types,
            'total_tools': total_tools,
            'tool_types': tool_types,
            'occupied_cells': occupied_cells,
            'occupation_rate': occupation_rate,
            'discoveries_made': self.tool_library.get_discovery_count()
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """Преобразует среду в словарь для сериализации"""
        return {
            'config': {
                'width': self.width,
                'height': self.height,
                'seed': self.seed,
                'resource_spawn_rate': self.config.resource_spawn_rate,
                'max_objects_per_cell': self.config.max_objects_per_cell,
                'season_length': self.config.season_length
            },
            'timestep': self.timestep,
            'season': self.season,
            'temperature': self.temperature,
            'resource_abundance': self.resource_abundance,
            'objects': {obj_id: obj.to_dict() for obj_id, obj in self.objects.items()},
            'tools': {tool_id: tool.to_dict() for tool_id, tool in self.tools.items()},
            'grid': {str(pos): obj_ids for pos, obj_ids in self.grid.items()}
        }
