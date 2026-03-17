"""
Tools module - defines tools as combinations of objects
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
import numpy as np
import random


@dataclass
class Tool:
    """Инструмент как комбинация объектов"""
    id: str
    components: List[str]  # ID компонентов
    creator_id: str  # ID агента-создателя
    created_at: int
    
    # Эмерджентные свойства
    effectiveness: Dict[str, float]  # {action: effectiveness_multiplier}
    durability_left: float
    properties: Dict[str, float] = field(default_factory=dict)

    # Named tool kind (optional, for UI / recipes)
    kind: Optional[str] = None
    
    # Метаданные
    usage_count: int = 0
    last_used: Optional[int] = None
    
    def __post_init__(self):
        """Инициализация после создания"""
        if not self.properties:
            # Если свойства не заданы, устанавливаем базовые
            self.properties = {
                'sharpness': 0.0,
                'effectiveness': 1.0,
                'weight': 0.5,
                'durability': self.durability_left / 100.0
            }
    
    def calculate_effectiveness(self, action: str) -> float:
        """Рассчитывает эффективность для конкретного действия"""
        base_effectiveness = self.effectiveness.get(action, 1.0)
        durability_factor = self.durability_left / 100.0
        
        # Специфические бонусы для разных типов действий
        action_bonuses = {
            'gather': self.properties.get('sharpness', 0.0) * 0.5,
            'attack': self.properties.get('sharpness', 0.0) * 0.8,
            'break': self.properties.get('sharpness', 0.0) * 0.6,
            'craft': self.properties.get('effectiveness', 1.0) * 0.3,
        }
        
        bonus = action_bonuses.get(action, 0.0)
        
        return base_effectiveness * durability_factor + bonus
    
    def use(self, action: str) -> float:
        """Использует инструмент для указанного действия"""
        effectiveness = self.calculate_effectiveness(action)
        
        # Деградация инструмента
        degradation_rate = 0.5  # Базовая скорость деградации
        
        # Увеличенная деградация для сложных действий
        if action in ['attack', 'break']:
            degradation_rate *= 1.5
        
        # Уменьшение прочности
        self.durability_left -= degradation_rate
        self.durability_left = max(0.0, self.durability_left)
        
        # Обновление счетчиков использования
        self.usage_count += 1
        self.last_used = self.created_at + self.usage_count  # Приблизительно
        
        return effectiveness
    
    def repair(self, repair_amount: float = 10.0):
        """Ремонт инструмента"""
        self.durability_left = min(100.0, self.durability_left + repair_amount)
    
    def is_broken(self) -> bool:
        """Проверяет, сломан ли инструмент"""
        return self.durability_left <= 0
    
    def get_complexity_score(self) -> float:
        """Рассчитывает сложность инструмента"""
        # Базовая сложность based on количество компонентов
        base_complexity = len(self.components) * 0.2
        
        # Бонус за разнообразие свойств
        property_diversity = len(set(self.properties.values())) * 0.1
        
        # Бонус за высокую эффективность
        effectiveness_bonus = max(self.effectiveness.values()) * 0.3
        
        return min(1.0, base_complexity + property_diversity + effectiveness_bonus)
    
    def get_tool_type(self) -> str:
        """Определяет тип инструмента на основе свойств"""
        sharpness = self.properties.get('sharpness', 0.0)
        effectiveness = self.properties.get('effectiveness', 1.0)
        
        if sharpness > 0.7:
            return 'cutting_tool'
        elif sharpness > 0.4:
            return 'scraping_tool'
        elif effectiveness > 1.5:
            return 'advanced_tool'
        else:
            return 'basic_tool'
    
    def to_dict(self) -> Dict[str, Any]:
        """Преобразует инструмент в словарь для сериализации"""
        return {
            'id': self.id,
            'components': self.components,
            'creator_id': self.creator_id,
            'created_at': self.created_at,
            'kind': self.kind,
            'effectiveness': self.effectiveness,
            'durability_left': self.durability_left,
            'properties': self.properties,
            'usage_count': self.usage_count,
            'last_used': self.last_used
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Tool':
        """Создает инструмент из словаря"""
        return cls(**data)


class ToolFactory:
    """Фабрика для создания инструментов из объектов"""

    _custom_recipes: List[Dict[str, Any]] = []

    @classmethod
    def set_custom_recipes(cls, recipes: List[Dict[str, Any]]):
        try:
            cls._custom_recipes = list(recipes or [])
        except Exception:
            cls._custom_recipes = []

    @classmethod
    def get_custom_recipes(cls) -> List[Dict[str, Any]]:
        try:
            return list(cls._custom_recipes or [])
        except Exception:
            return []

    @staticmethod
    def _item_token(item: Any) -> Optional[str]:
        # Object ingredient
        t = getattr(item, 'type', None)
        if t is not None:
            return f"obj:{str(t)}"

        # Tool ingredient (only if it has a kind)
        k = getattr(item, 'kind', None)
        if k:
            return f"tool:{str(k)}"
        return None
    
    @staticmethod
    def create_tool_from_objects(component_objects: List, creator_id: str,
                                tool_id: str, timestamp: int,
                                agent_skills: Optional[Dict[str, float]] = None) -> Optional[Tool]:
        """Создает инструмент из списка объектов"""

        if len(component_objects) < 2:
            return None

        # Admin-defined recipes first
        try:
            recipe_tool = ToolFactory._try_create_custom_recipe_tool(component_objects, creator_id, tool_id, timestamp)
            if recipe_tool is not None:
                return recipe_tool
        except Exception:
            pass

        # Simple named recipes next (keep emergent tools as fallback)
        try:
            recipe_tool = ToolFactory._try_create_named_tool(component_objects, creator_id, tool_id, timestamp,
                                                              agent_skills=agent_skills)
            if recipe_tool is not None:
                return recipe_tool
        except Exception:
            pass
        
        # Рассчитываем комбинированные свойства
        combined_properties = ToolFactory._calculate_combined_properties(component_objects)
        
        # Рассчитываем эффективность для разных действий
        effectiveness = ToolFactory._calculate_action_effectiveness(combined_properties)
        
        # Рассчитываем прочность
        durability = ToolFactory._calculate_durability(combined_properties)
        
        # Проверяем, является ли комбинация полезной
        if not ToolFactory._is_useful_combination(effectiveness):
            return None
        
        tool = Tool(
            id=tool_id,
            components=[obj.id for obj in component_objects],
            creator_id=creator_id,
            created_at=timestamp,
            effectiveness=effectiveness,
            durability_left=durability,
            properties=combined_properties
        )
        
        return tool

    # ── Таблица именованных рецептов ─────────────────────────────────────
    # Ключ: отсортированный кортеж токенов ("obj:<type>" или "tool:<kind>")
    # skill_req: минимальные уровни навыков для крафта
    # produces_object: если задан — рецепт создаёт Object, а не Tool
    NAMED_RECIPES: Dict[tuple, Dict[str, Any]] = {
        # ── Tier 1: без требований к навыкам ──
        ("obj:stone", "obj:wood"): {
            'kind': 'wooden_axe',
            'effectiveness': {'gather': 1.8, 'break': 1.4, 'attack': 1.1, 'craft': 1.1},
            'durability_left': 80.0,
            'props': {'sharpness': 0.75, 'effectiveness': 1.6, 'durability': 0.8},
        },
        ("obj:bone", "obj:wood"): {
            'kind': 'wooden_spear',
            'effectiveness': {'gather': 1.2, 'break': 1.1, 'attack': 1.7, 'craft': 1.1},
            'durability_left': 70.0,
            'props': {'sharpness': 0.65, 'effectiveness': 1.5, 'durability': 0.7},
        },
        ("obj:stone", "obj:stone"): {
            'kind': 'stone_hammer',
            'effectiveness': {'gather': 1.1, 'break': 1.9, 'attack': 1.2, 'craft': 1.1},
            'durability_left': 90.0,
            'props': {'sharpness': 0.2, 'effectiveness': 1.6, 'durability': 0.9},
        },
        ("obj:fiber", "obj:fiber"): {
            'kind': 'rope',
            'produces_object': 'rope',  # создаёт объект rope, а не инструмент
        },
        ("obj:stone", "obj:wood", "obj:wood"): {
            'kind': 'wooden_pickaxe',
            'effectiveness': {'gather': 1.5, 'break': 1.8, 'attack': 0.9, 'craft': 1.0},
            'durability_left': 75.0,
            'props': {'sharpness': 0.5, 'effectiveness': 1.5, 'durability': 0.75},
        },
        ("obj:bone", "obj:bone", "obj:stone"): {
            'kind': 'bone_knife',
            'effectiveness': {'gather': 1.3, 'break': 1.0, 'attack': 1.5, 'craft': 1.3},
            'durability_left': 65.0,
            'props': {'sharpness': 0.8, 'effectiveness': 1.4, 'durability': 0.65},
        },

        # ── Tier 2: crafting >= 0.2 (lv3) ──
        ("obj:clay", "obj:clay"): {
            'kind': 'clay_pot',
            'skill_req': {'crafting': 0.2},
            'effectiveness': {'gather': 1.0, 'craft': 1.4, 'break': 1.0, 'attack': 1.0},
            'durability_left': 50.0,
            'props': {'sharpness': 0.0, 'effectiveness': 1.4, 'durability': 0.5},
        },
        ("obj:fiber", "obj:leather"): {
            'kind': 'leather_bag',
            'skill_req': {'crafting': 0.2},
            'effectiveness': {'gather': 1.0, 'craft': 1.0, 'break': 1.0, 'attack': 1.0},
            'durability_left': 80.0,
            'props': {'sharpness': 0.0, 'effectiveness': 1.0, 'durability': 0.8},
            'passive_bonus': {'inventory_bonus': 3},
        },
        ("obj:rope", "obj:wood", "obj:wood"): {
            'kind': 'fishing_rod',
            'skill_req': {'crafting': 0.2, 'gathering': 0.2},
            'effectiveness': {'gather': 2.0, 'break': 0.8, 'attack': 0.8, 'craft': 1.0},
            'durability_left': 60.0,
            'props': {'sharpness': 0.1, 'effectiveness': 1.5, 'durability': 0.6},
        },
        ("obj:clay", "obj:wood"): {
            'kind': 'torch',
            'skill_req': {'crafting': 0.2},
            'effectiveness': {'gather': 1.0, 'break': 1.0, 'attack': 1.3, 'craft': 1.0},
            'durability_left': 40.0,
            'props': {'sharpness': 0.0, 'effectiveness': 1.2, 'durability': 0.4},
        },
        ("obj:herb", "obj:herb"): {
            'kind': 'medicine_pouch',
            'skill_req': {'survival': 0.2},
            'produces_object': 'herb',  # улучшенное лекарство (qty=3)
            'produces_quantity': 3,
        },
        ("obj:bone", "obj:fiber"): {
            'kind': 'bone_needle',
            'skill_req': {'crafting': 0.2},
            'effectiveness': {'gather': 1.0, 'break': 1.0, 'attack': 1.0, 'craft': 1.6},
            'durability_left': 45.0,
            'props': {'sharpness': 0.6, 'effectiveness': 1.5, 'durability': 0.45},
        },

        # ── Tier 3: crafting >= 0.4 (lv5) ──
        ("obj:leather", "obj:leather", "obj:rope"): {
            'kind': 'leather_armor',
            'skill_req': {'crafting': 0.4},
            'effectiveness': {'gather': 1.0, 'break': 1.0, 'attack': 1.0, 'craft': 1.0},
            'durability_left': 70.0,
            'props': {'sharpness': 0.0, 'effectiveness': 1.0, 'durability': 0.7},
            'passive_bonus': {'damage_reduction': 0.25},
        },
        ("obj:ore", "obj:stone", "obj:wood"): {
            'kind': 'stone_furnace',
            'skill_req': {'crafting': 0.4},
            'placed_object': 'stone_furnace',  # размещается на карте
        },
        ("obj:clay", "obj:clay", "obj:wood"): {
            'kind': 'clay_oven',
            'skill_req': {'crafting': 0.4},
            'placed_object': 'clay_oven',  # размещается на карте
        },
        ("obj:ore", "tool:stone_hammer"): {
            'kind': 'metal_ingot',
            'skill_req': {'crafting': 0.4},
            'produces_object': 'metal_ingot',
        },
        ("obj:leather", "obj:wood", "obj:rope"): {
            'kind': 'sling',
            'skill_req': {'crafting': 0.4},
            'effectiveness': {'gather': 1.0, 'break': 1.2, 'attack': 1.8, 'craft': 1.0},
            'durability_left': 55.0,
            'props': {'sharpness': 0.3, 'effectiveness': 1.4, 'durability': 0.55},
        },

        # ── Tier 4: crafting >= 0.6 (lv7) ──
        ("obj:metal_ingot", "obj:wood"): {
            'kind': 'metal_axe',
            'skill_req': {'crafting': 0.6},
            'effectiveness': {'gather': 2.5, 'break': 2.0, 'attack': 1.5, 'craft': 1.3},
            'durability_left': 95.0,
            'props': {'sharpness': 0.9, 'effectiveness': 2.2, 'durability': 0.95},
        },
        ("obj:metal_ingot", "obj:bone"): {
            'kind': 'metal_spear',
            'skill_req': {'crafting': 0.6},
            'effectiveness': {'gather': 1.5, 'break': 1.3, 'attack': 2.5, 'craft': 1.1},
            'durability_left': 90.0,
            'props': {'sharpness': 0.85, 'effectiveness': 2.0, 'durability': 0.9},
        },
        ("obj:metal_ingot", "obj:metal_ingot"): {
            'kind': 'metal_shield',
            'skill_req': {'crafting': 0.6},
            'effectiveness': {'gather': 1.0, 'break': 1.0, 'attack': 1.2, 'craft': 1.0},
            'durability_left': 95.0,
            'props': {'sharpness': 0.0, 'effectiveness': 1.2, 'durability': 0.95},
            'passive_bonus': {'damage_reduction': 0.4},
        },
        ("obj:leather", "obj:metal_ingot", "obj:rope"): {
            'kind': 'metal_armor',
            'skill_req': {'crafting': 0.6},
            'effectiveness': {'gather': 1.0, 'break': 1.0, 'attack': 1.0, 'craft': 1.0},
            'durability_left': 90.0,
            'props': {'sharpness': 0.0, 'effectiveness': 1.0, 'durability': 0.9},
            'passive_bonus': {'damage_reduction': 0.45},
        },
        ("obj:metal_ingot", "obj:wood", "obj:rope"): {
            'kind': 'metal_pickaxe',
            'skill_req': {'crafting': 0.6},
            'effectiveness': {'gather': 2.0, 'break': 2.5, 'attack': 1.0, 'craft': 1.2},
            'durability_left': 95.0,
            'props': {'sharpness': 0.7, 'effectiveness': 2.0, 'durability': 0.95},
        },
    }

    # Обратный индекс: kind -> ключ рецепта (для целевого крафта по команде)
    RECIPE_BY_KIND: Dict[str, tuple] = {}
    for _k, _v in NAMED_RECIPES.items():
        RECIPE_BY_KIND[_v['kind']] = _k

    @staticmethod
    def _try_create_named_tool(component_objects: List, creator_id: str,
                               tool_id: str, timestamp: int,
                               agent_skills: Optional[Dict[str, float]] = None) -> Optional[Tool]:
        """Try to create a named tool from the recipe dictionary."""
        if len(component_objects) < 2:
            return None

        # Собираем токены из компонентов
        tokens = []
        for it in component_objects:
            tok = ToolFactory._item_token(it)
            if tok is None:
                # plain object without type — fallback to type attr
                t = getattr(it, 'type', None)
                if t is not None:
                    tok = f"obj:{t}"
                else:
                    return None
            tokens.append(tok)
        tokens_key = tuple(sorted(tokens))

        recipe = ToolFactory.NAMED_RECIPES.get(tokens_key)
        if recipe is None:
            return None

        # Проверка навыков
        skill_req = recipe.get('skill_req', {})
        if skill_req and agent_skills:
            for sk, req_val in skill_req.items():
                if agent_skills.get(sk, 0.0) < req_val:
                    return None

        # Рецепт, создающий объект или размещаемый — обрабатывается отдельно
        if recipe.get('produces_object') or recipe.get('placed_object'):
            return None  # будет обработан в _try_create_named_object

        kind = recipe['kind']
        effectiveness = dict(recipe['effectiveness'])
        durability_left = recipe['durability_left']
        props = dict(recipe['props'])
        props['weight'] = float(sum(getattr(o, 'weight', 0.5) for o in component_objects) / max(1, len(component_objects)))

        return Tool(
            id=tool_id,
            components=[getattr(obj, 'id', str(obj)) for obj in component_objects],
            creator_id=creator_id,
            created_at=timestamp,
            kind=kind,
            effectiveness=effectiveness,
            durability_left=durability_left,
            properties=props,
        )

    @staticmethod
    def _try_create_named_object(component_objects: List,
                                 agent_skills: Optional[Dict[str, float]] = None) -> Optional[Dict[str, Any]]:
        """Check if components match a recipe that produces an Object or placed structure.
        Returns a dict describing the result, or None."""
        if len(component_objects) < 2:
            return None

        tokens = []
        for it in component_objects:
            tok = ToolFactory._item_token(it)
            if tok is None:
                t = getattr(it, 'type', None)
                if t is not None:
                    tok = f"obj:{t}"
                else:
                    return None
            tokens.append(tok)
        tokens_key = tuple(sorted(tokens))

        recipe = ToolFactory.NAMED_RECIPES.get(tokens_key)
        if recipe is None:
            return None

        skill_req = recipe.get('skill_req', {})
        if skill_req and agent_skills:
            for sk, req_val in skill_req.items():
                if agent_skills.get(sk, 0.0) < req_val:
                    return None

        if recipe.get('produces_object'):
            return {
                'type': 'object',
                'object_type': recipe['produces_object'],
                'quantity': recipe.get('produces_quantity', 1),
                'kind': recipe['kind'],
            }
        elif recipe.get('placed_object'):
            return {
                'type': 'placed',
                'object_type': recipe['placed_object'],
                'kind': recipe['kind'],
            }
        return None

    @staticmethod
    def get_recipes_for_api() -> List[Dict[str, Any]]:
        """Return recipe list for the /api/recipes endpoint."""
        result = []
        for tokens_key, recipe in ToolFactory.NAMED_RECIPES.items():
            result.append({
                'kind': recipe['kind'],
                'components': list(tokens_key),
                'skill_req': recipe.get('skill_req', {}),
                'produces_object': recipe.get('produces_object'),
                'placed_object': recipe.get('placed_object'),
                'passive_bonus': recipe.get('passive_bonus'),
                'durability_left': recipe.get('durability_left'),
            })
        return result

    @staticmethod
    def _try_create_custom_recipe_tool(component_objects: List, creator_id: str,
                                       tool_id: str, timestamp: int) -> Optional[Tool]:
        if len(component_objects) < 2:
            return None

        tokens = []
        for it in component_objects:
            tok = ToolFactory._item_token(it)
            if tok is None:
                return None
            tokens.append(tok)
        tokens.sort()

        for r in (ToolFactory._custom_recipes or []):
            try:
                comps = r.get('components') or []
                if not isinstance(comps, list) or len(comps) < 2:
                    continue
                rc = sorted([str(c) for c in comps])
                if rc != tokens:
                    continue

                kind = str(r.get('kind') or '').strip()
                if not kind:
                    continue

                eff = r.get('effectiveness') or {}
                if not isinstance(eff, dict) or not eff:
                    continue
                effectiveness = {str(k): float(v) for k, v in eff.items()}

                durability_left = float(r.get('durability_left') or 60.0)
                durability_left = float(max(5.0, min(100.0, durability_left)))

                props_in = r.get('properties') or {}
                props = {
                    'sharpness': float(props_in.get('sharpness', 0.0) or 0.0),
                    'effectiveness': float(props_in.get('effectiveness', 1.0) or 1.0),
                    'weight': float(sum(getattr(o, 'weight', 0.5) for o in component_objects) / max(1, len(component_objects))),
                    'durability': float(props_in.get('durability', durability_left / 100.0) or (durability_left / 100.0)),
                }

                return Tool(
                    id=tool_id,
                    components=[obj.id for obj in component_objects],
                    creator_id=creator_id,
                    created_at=timestamp,
                    kind=kind,
                    effectiveness=effectiveness,
                    durability_left=durability_left,
                    properties=props,
                )
            except Exception:
                continue
        return None
    
    @staticmethod
    def _calculate_combined_properties(objects: List) -> Dict[str, float]:
        """Рассчитывает комбинированные свойства объектов"""
        if not objects:
            return {}
        
        # Взвешенная сумма свойств
        weights: List[float] = []
        for obj in objects:
            try:
                w = float(getattr(obj, 'weight', 0.0) or 0.0)
            except Exception:
                w = 0.0
            weights.append(w)

        total_weight = float(sum(weights))
        if total_weight <= 0.0:
            # Fallback: treat all objects equally to avoid division by zero.
            weights = [1.0 for _ in objects]
            total_weight = float(len(objects))
        
        combined = {}
        property_names = [
            'hardness', 'nutrition', 'toxicity', 'weight', 
            'durability', 'flammability', 'energy_cost'
        ]
        
        for prop_name in property_names:
            weighted_sum = 0.0
            for obj, w in zip(objects, weights):
                try:
                    v = float(getattr(obj, prop_name, 0.0) or 0.0)
                except Exception:
                    v = 0.0
                weighted_sum += v * w / total_weight
            combined[prop_name] = float(weighted_sum)
        
        # Эмерджентные свойства
        combined['sharpness'] = ToolFactory._calculate_sharpness(objects)
        combined['effectiveness'] = ToolFactory._calculate_base_effectiveness(objects)
        
        return combined
    
    @staticmethod
    def _calculate_sharpness(objects: List) -> float:
        """Рассчитывает остроту комбинации"""
        sharpness = 0.0
        
        # Камень + дерево = потенциально острый инструмент
        has_stone = any(obj.type == 'stone' for obj in objects)
        has_wood = any(obj.type == 'wood' for obj in objects)
        has_bone = any(obj.type == 'bone' for obj in objects)
        
        if has_stone and (has_wood or has_bone):
            sharpness = 0.8
        elif has_stone:
            sharpness = 0.4
        elif has_bone:
            sharpness = 0.3
        
        # Модификатор based on твердость
        max_hardness = max(obj.hardness for obj in objects)
        sharpness *= max_hardness
        
        return min(1.0, sharpness)
    
    @staticmethod
    def _calculate_base_effectiveness(objects: List) -> float:
        """Рассчитывает базовую эффективность комбинации"""
        effectiveness = 1.0
        
        # Бонус за комбинацию твердых объектов
        hard_objects = [obj for obj in objects if obj.hardness > 0.7]
        if len(hard_objects) >= 2:
            effectiveness += 0.3
        
        # Бонус за комбинацию легких и прочных объектов
        light_durable = [
            obj for obj in objects 
            if obj.weight < 0.5 and obj.durability > 0.6
        ]
        if len(light_durable) >= 2:
            effectiveness += 0.2
        
        # Бонус за разнообразие типов
        types = set(obj.type for obj in objects)
        if len(types) >= 3:
            effectiveness += 0.1
        
        return min(2.0, effectiveness)
    
    @staticmethod
    def _calculate_action_effectiveness(properties: Dict[str, float]) -> Dict[str, float]:
        """Рассчитывает эффективность для разных действий"""
        sharpness = properties.get('sharpness', 0.0)
        base_effectiveness = properties.get('effectiveness', 1.0)
        hardness = properties.get('hardness', 0.0)
        
        effectiveness = {
            'gather': 1.0 + sharpness * 0.5 + hardness * 0.2,
            'attack': 1.0 + sharpness * 0.8 + hardness * 0.3,
            'break': 1.0 + sharpness * 0.6 + hardness * 0.4,
            'craft': 1.0 + base_effectiveness * 0.3,
            'move': 1.0,  # Инструменты обычно не помогают в движении
            'rest': 1.0,  # Инструменты обычно не помогают в отдыхе
        }
        
        return effectiveness
    
    @staticmethod
    def _calculate_durability(properties: Dict[str, float]) -> float:
        """Рассчитывает прочность инструмента"""
        base_durability = properties.get('durability', 0.5)
        hardness = properties.get('hardness', 0.0)
        
        # Прочность зависит от базовой прочности и твердости
        durability = (base_durability * 0.7 + hardness * 0.3) * 100
        
        return min(100.0, max(10.0, durability))
    
    @staticmethod
    def _is_useful_combination(effectiveness: Dict[str, float]) -> bool:
        """Проверяет, является ли комбинация полезной"""
        # Комбинация полезна, если хотя бы для одного действия эффективность > 1.2
        max_effectiveness = max(effectiveness.values())
        return max_effectiveness > 1.2


class ToolLibrary:
    """Библиотека для отслеживания известных типов инструментов"""
    
    def __init__(self):
        self.known_tools = {}  # {tool_signature: tool_type}
        self.tool_discoveries = []  # История открытий
    
    def register_tool(self, tool: Tool) -> str:
        """Регистрирует новый тип инструмента"""
        signature = self._create_signature(tool)
        
        if signature not in self.known_tools:
            tool_type = tool.get_tool_type()
            self.known_tools[signature] = tool_type
            self.tool_discoveries.append({
                'timestamp': tool.created_at,
                'tool_id': tool.id,
                'tool_type': tool_type,
                'signature': signature,
                'creator_id': tool.creator_id
            })
            return 'new_discovery'
        else:
            return 'known_type'
    
    def _create_signature(self, tool: Tool) -> str:
        """Создает уникальную сигнатуру инструмента"""
        # Сигнатура based on типы компонентов и основные свойства
        component_types = sorted(set(
            comp_id.split('_')[0] for comp_id in tool.components
        ))
        
        properties_key = (
            f"sharp_{tool.properties.get('sharpness', 0.0):.2f}_"
            f"eff_{tool.properties.get('effectiveness', 1.0):.2f}"
        )
        
        return f"{'_'.join(component_types)}_{properties_key}"
    
    def get_discovery_count(self) -> int:
        """Возвращает количество уникальных открытий"""
        return len(self.known_tools)
    
    def get_discovery_history(self) -> List[Dict]:
        """Возвращает историю открытий"""
        return self.tool_discoveries.copy()
    
    def get_tool_statistics(self) -> Dict[str, Any]:
        """Возвращает статистику по инструментам"""
        if not self.known_tools:
            return {}
        
        tool_types = list(self.known_tools.values())
        type_counts = {}
        for tool_type in tool_types:
            type_counts[tool_type] = type_counts.get(tool_type, 0) + 1
        
        return {
            'total_types': len(self.known_tools),
            'type_distribution': type_counts,
            'discovery_rate': len(self.tool_discoveries) / max(1, len(self.tool_discoveries))
        }
