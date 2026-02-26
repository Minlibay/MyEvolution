"""
Objects module - defines physical objects in the simulation environment
"""

from dataclasses import dataclass, field
from typing import Tuple, Dict, Any
import numpy as np
import random


@dataclass
class Object:
    """Базовый объект в среде с физическими свойствами"""
    id: str
    type: str  # 'stone', 'wood', 'plant', 'animal', etc.
    position: Tuple[int, int]
    
    # Физические свойства (нормализованные [0,1])
    hardness: float      # Твердость
    nutrition: float     # Питательность  
    toxicity: float      # Токсичность
    weight: float        # Вес
    durability: float    # Прочность
    flammability: float  # Воспламеняемость
    energy_cost: float   # Энергетическая стоимость использования
    
    # Метаданные
    created_at: int
    quantity: int = 1
    
    def __post_init__(self):
        """Валидация свойств после инициализации"""
        self._validate_properties()
    
    def _validate_properties(self):
        """Проверяет, что все свойства в диапазоне [0,1]"""
        properties = [
            self.hardness, self.nutrition, self.toxicity,
            self.weight, self.durability, self.flammability, self.energy_cost
        ]
        
        for i, prop in enumerate(properties):
            if not 0.0 <= prop <= 1.0:
                raise ValueError(f"Property {i} ({prop}) must be in range [0,1]")
    
    def get_properties_vector(self) -> np.ndarray:
        """Возвращает вектор свойств объекта"""
        return np.array([
            self.hardness, self.nutrition, self.toxicity,
            self.weight, self.durability, self.flammability,
            self.energy_cost
        ])
    
    def degrade(self, rate: float = 0.01):
        """Деградация объекта со временем"""
        self.durability = max(0.0, self.durability - rate)
        if self.durability <= 0:
            self.quantity = 0
    
    def combine_with(self, other: 'Object') -> Dict[str, float]:
        """Рассчитывает свойства комбинации с другим объектом"""
        # Взвешенная сумма свойств
        weight1 = self.weight / (self.weight + other.weight)
        weight2 = other.weight / (self.weight + other.weight)
        
        combined_properties = {}
        property_names = [
            'hardness', 'nutrition', 'toxicity', 'weight', 
            'durability', 'flammability', 'energy_cost'
        ]
        
        for prop_name in property_names:
            prop1 = getattr(self, prop_name)
            prop2 = getattr(other, prop_name)
            combined_properties[prop_name] = prop1 * weight1 + prop2 * weight2
        
        # Эмерджентные свойства для комбинаций
        combined_properties['sharpness'] = self._calculate_sharpness(other)
        combined_properties['effectiveness'] = self._calculate_effectiveness(other)
        
        return combined_properties
    
    def _calculate_sharpness(self, other: 'Object') -> float:
        """Рассчитывает остроту комбинации"""
        # Камень + дерево = потенциально острый инструмент
        if (self.type == 'stone' and other.type == 'wood') or \
           (self.type == 'wood' and other.type == 'stone'):
            return min(1.0, self.hardness * 0.8 + other.hardness * 0.2)
        return 0.0
    
    def _calculate_effectiveness(self, other: 'Object') -> float:
        """Рассчитывает общую эффективность комбинации"""
        base_effectiveness = 1.0
        
        # Бонус за комбинацию твердых объектов
        if self.hardness > 0.7 and other.hardness > 0.5:
            base_effectiveness += 0.3
        
        # Бонус за комбинацию легких и прочных объектов
        if (self.weight < 0.5 and self.durability > 0.6) and \
           (other.weight < 0.5 and other.durability > 0.6):
            base_effectiveness += 0.2
        
        return min(2.0, base_effectiveness)  # Ограничиваем максимальную эффективность
    
    def is_edible(self) -> bool:
        """Проверяет, является ли объект съедобным"""
        return self.nutrition > 0.3 and self.toxicity < 0.4
    
    def is_tool_material(self) -> bool:
        """Проверяет, может ли объект использоваться для создания инструментов"""
        return self.hardness > 0.4 or self.durability > 0.5
    
    def get_energy_value(self) -> float:
        """Возвращает энергетическую ценность объекта"""
        return self.nutrition * (1.0 - self.toxicity) * self.quantity
    
    def to_dict(self) -> Dict[str, Any]:
        """Преобразует объект в словарь для сериализации"""
        return {
            'id': self.id,
            'type': self.type,
            'position': self.position,
            'hardness': self.hardness,
            'nutrition': self.nutrition,
            'toxicity': self.toxicity,
            'weight': self.weight,
            'durability': self.durability,
            'flammability': self.flammability,
            'energy_cost': self.energy_cost,
            'created_at': self.created_at,
            'quantity': self.quantity
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Object':
        """Создает объект из словаря"""
        return cls(**data)


class ObjectFactory:
    """Фабрика для создания объектов с различными свойствами"""
    
    # Базовые шаблоны объектов
    OBJECT_TEMPLATES = {
        'stone': {
            'hardness': 0.9, 'nutrition': 0.0, 'toxicity': 0.0,
            'weight': 0.8, 'durability': 0.9, 'flammability': 0.0, 'energy_cost': 0.3
        },
        'wood': {
            'hardness': 0.4, 'nutrition': 0.1, 'toxicity': 0.0,
            'weight': 0.5, 'durability': 0.6, 'flammability': 0.8, 'energy_cost': 0.2
        },
        'plant': {
            'hardness': 0.1, 'nutrition': 0.7, 'toxicity': 0.2,
            'weight': 0.2, 'durability': 0.3, 'flammability': 0.4, 'energy_cost': 0.1
        },
        'berry': {
            'hardness': 0.05, 'nutrition': 0.8, 'toxicity': 0.3,
            'weight': 0.1, 'durability': 0.2, 'flammability': 0.1, 'energy_cost': 0.05
        },
        'bone': {
            'hardness': 0.7, 'nutrition': 0.2, 'toxicity': 0.1,
            'weight': 0.6, 'durability': 0.8, 'flammability': 0.0, 'energy_cost': 0.25
        },
        'fiber': {
            'hardness': 0.05, 'nutrition': 0.1, 'toxicity': 0.0,
            'weight': 0.05, 'durability': 0.4, 'flammability': 0.3, 'energy_cost': 0.02
        },
        'water': {
            'hardness': 0.0, 'nutrition': 0.0, 'toxicity': 0.0,
            'weight': 0.0, 'durability': 1.0, 'flammability': 0.0, 'energy_cost': 0.0
        }
    }
    
    @classmethod
    def create_object(cls, obj_type: str, position: Tuple[int, int], 
                     obj_id: str, timestamp: int, season: int = 0) -> Object:
        """Создает объект с учетом сезонных модификаторов"""
        
        if obj_type not in cls.OBJECT_TEMPLATES:
            raise ValueError(f"Unknown object type: {obj_type}")
        
        # Базовые свойства
        properties = cls.OBJECT_TEMPLATES[obj_type].copy()
        
        # Сезонные модификаторы
        properties = cls._apply_seasonal_modifiers(properties, season)
        
        # Добавляем случайные вариации
        properties = cls._add_random_variations(properties)
        
        return Object(
            id=obj_id,
            type=obj_type,
            position=position,
            created_at=timestamp,
            **properties
        )
    
    @classmethod
    def _apply_seasonal_modifiers(cls, properties: Dict[str, float], season: int) -> Dict[str, float]:
        """Применяет сезонные модификаторы к свойствам"""
        modifiers = {
            0: {'nutrition': 1.2, 'toxicity': 0.8},  # Весна
            1: {'nutrition': 1.5, 'toxicity': 0.5},  # Лето
            2: {'nutrition': 1.0, 'toxicity': 1.0},  # Осень
            3: {'nutrition': 0.5, 'toxicity': 1.5},  # Зима
        }
        
        season_mods = modifiers.get(season, {})
        
        for key, modifier in season_mods.items():
            if key in properties:
                properties[key] = min(1.0, properties[key] * modifier)
        
        return properties
    
    @classmethod
    def _add_random_variations(cls, properties: Dict[str, float]) -> Dict[str, float]:
        """Добавляет случайные вариации к свойствам"""
        for key in properties:
            if key != 'energy_cost':  # Энергетическая стоимость остается стабильной
                noise = random.gauss(0, 0.1)
                properties[key] = max(0.0, min(1.0, properties[key] + noise))
        
        return properties
    
    @classmethod
    def get_random_object_type(cls, season: int = 0) -> str:
        """Возвращает случайный тип объекта с учетом сезона"""
        # Сезонные вероятности
        season_weights = {
            0: {'plant': 0.3, 'berry': 0.2, 'wood': 0.2, 'stone': 0.2, 'bone': 0.05, 'fiber': 0.05},
            1: {'plant': 0.2, 'berry': 0.4, 'wood': 0.2, 'stone': 0.1, 'bone': 0.05, 'fiber': 0.05},
            2: {'plant': 0.25, 'berry': 0.25, 'wood': 0.25, 'stone': 0.15, 'bone': 0.05, 'fiber': 0.05},
            3: {'plant': 0.1, 'berry': 0.05, 'wood': 0.3, 'stone': 0.4, 'bone': 0.1, 'fiber': 0.05},
        }
        
        weights = season_weights.get(season, season_weights[0])
        object_types = list(weights.keys())
        probabilities = list(weights.values())
        
        return random.choices(object_types, weights=probabilities)[0]
