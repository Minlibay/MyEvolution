"""
Memory module - advanced memory systems for agents
"""

from typing import Dict, List, Tuple, Any, Optional
import numpy as np
from collections import defaultdict, deque

from core.agent import Episode


class AdvancedEpisodicMemory:
    """Улучшенная эпизодическая память с приоритетами"""
    
    def __init__(self, capacity: int = 100):
        self.capacity = capacity
        self.episodes: deque = deque(maxlen=capacity)
        self.priority_weights = {
            'high_reward': 2.0,      # Высокая награда
            'low_energy': 1.5,       # Низкая энергия
            'discovery': 3.0,        # Открытие
            'death_near': 2.5,       # Близость к смерти
            'recent': 1.2            # Недавние события
        }
    
    def add_episode(self, episode: Episode):
        """Добавляет эпизод с приоритетом"""
        priority = self._calculate_priority(episode)
        
        # Вставляем с учетом приоритета
        inserted = False
        for i, (existing_episode, _) in enumerate(self.episodes):
            if priority > _:
                self.episodes.insert(i, (episode, priority))
                inserted = True
                break
        
        if not inserted:
            self.episodes.append((episode, priority))
    
    def _calculate_priority(self, episode: Episode) -> float:
        """Рассчитывает приоритет эпизода"""
        priority = 1.0
        
        # Высокая награда
        if episode.reward > 0.5:
            priority += self.priority_weights['high_reward']
        
        # Низкая энергия в контексте
        if 'energy' in episode.context:
            energy = episode.context['energy']
            if energy < 0.2:
                priority += self.priority_weights['low_energy']
        
        # Открытие
        if 'discovery' in episode.context.get('data', {}):
            priority += self.priority_weights['discovery']
        
        # Близость к смерти
        if 'health' in episode.context:
            health = episode.context['health']
            if health < 0.3:
                priority += self.priority_weights['death_near']
        
        # Недавние события
        priority += self.priority_weights['recent'] * (1.0 / (1.0 + episode.timestamp * 0.001))
        
        return priority
    
    def recall_similar(self, state: str, k: int = 5, 
                     context_filter: Optional[Dict] = None) -> List[Episode]:
        """Вспоминает похожие эпизоды с учетом приоритета"""
        if not self.episodes:
            return []
        
        # Фильтрация по контексту если нужно
        candidates = []
        for episode, priority in self.episodes:
            if context_filter is None or self._matches_context(episode, context_filter):
                similarity = self._calculate_similarity(state, episode.state)
                candidates.append((similarity * priority, episode))
        
        # Сортируем по взвешенной схожести
        candidates.sort(key=lambda x: x[0], reverse=True)
        
        return [episode for _, episode in candidates[:k]]
    
    def _matches_context(self, episode: Episode, context_filter: Dict) -> bool:
        """Проверяет соответствие эпизода фильтру контекста"""
        episode_context = episode.context
        
        for key, value in context_filter.items():
            if key not in episode_context:
                return False
            
            if isinstance(value, (int, float)):
                # Для числовых значений проверяем приблизительное равенство
                if abs(episode_context[key] - value) > 0.1:
                    return False
            else:
                # Для других значений проверяем точное равенство
                if episode_context[key] != value:
                    return False
        
        return True
    
    def _calculate_similarity(self, state1: str, state2: str) -> float:
        """Улучшенное расчет схожести состояний"""
        words1 = set(state1.split('|'))
        words2 = set(state2.split('|'))
        
        if not words1 or not words2:
            return 0.0
        
        # Jaccard similarity
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        
        base_similarity = intersection / union if union > 0 else 0.0
        
        # Учитываем порядок слов (для последовательных состояний)
        words1_list = state1.split('|')
        words2_list = state2.split('|')
        
        # Сравниваем начало последовательности (важно для контекста)
        sequence_match = 0
        min_len = min(len(words1_list), len(words2_list))
        for i in range(min_len):
            if words1_list[i] == words2_list[i]:
                sequence_match += 1
            else:
                break
        
        sequence_similarity = sequence_match / max(len(words1_list), len(words2_list))
        
        # Комбинированная схожесть
        return base_similarity * 0.7 + sequence_similarity * 0.3
    
    def get_memory_statistics(self) -> Dict[str, Any]:
        """Возвращает статистику памяти"""
        if not self.episodes:
            return {}
        
        episodes_list = [ep for ep, _ in self.episodes]
        
        # Распределение наград
        rewards = [ep.reward for ep in episodes_list]
        
        # Распределение действий
        actions = defaultdict(int)
        for ep in episodes_list:
            actions[ep.action] += 1
        
        # Временное распределение
        timestamps = [ep.timestamp for ep in episodes_list]
        
        return {
            'total_episodes': len(episodes_list),
            'average_reward': np.mean(rewards),
            'reward_std': np.std(rewards),
            'action_distribution': dict(actions),
            'time_span': (min(timestamps), max(timestamps)),
            'average_priority': np.mean([priority for _, priority in self.episodes])
        }


class SemanticMemory:
    """Семантическая память для хранения концепций и отношений"""
    
    def __init__(self):
        # Концепции: {concept_id: concept_data}
        self.concepts: Dict[str, Dict] = {}
        
        # Отношения: {relation_id: (concept1, concept2, relation_type, strength)}
        self.relations: Dict[str, Tuple[str, str, str, float]] = {}
        
        # Счетчик для ID
        self.concept_counter = 0
        self.relation_counter = 0
    
    def add_concept(self, concept_type: str, features: Dict[str, Any]) -> str:
        """Добавляет концепцию в память"""
        concept_id = f"concept_{self.concept_counter}"
        self.concept_counter += 1
        
        self.concepts[concept_id] = {
            'type': concept_type,
            'features': features,
            'activation': 1.0,
            'creation_time': 0,  # Будет установлено извне
            'access_count': 0
        }
        
        return concept_id
    
    def add_relation(self, concept1_id: str, concept2_id: str, 
                    relation_type: str, strength: float = 1.0) -> str:
        """Добавляет отношение между концепциями"""
        if concept1_id not in self.concepts or concept2_id not in self.concepts:
            raise ValueError("One or both concepts not found")
        
        relation_id = f"relation_{self.relation_counter}"
        self.relation_counter += 1
        
        self.relations[relation_id] = (concept1_id, concept2_id, relation_type, strength)
        
        return relation_id
    
    def find_concepts(self, concept_type: str = None, 
                     features_filter: Dict = None) -> List[str]:
        """Находит концепции по типу и/или признакам"""
        matching_concepts = []
        
        for concept_id, concept_data in self.concepts.items():
            # Фильтр по типу
            if concept_type and concept_data['type'] != concept_type:
                continue
            
            # Фильтр по признакам
            if features_filter:
                match = True
                for key, value in features_filter.items():
                    if key not in concept_data['features']:
                        match = False
                        break
                    
                    feature_value = concept_data['features'][key]
                    if isinstance(value, (int, float)):
                        if abs(feature_value - value) > 0.1:
                            match = False
                            break
                    else:
                        if feature_value != value:
                            match = False
                            break
                
                if not match:
                    continue
            
            matching_concepts.append(concept_id)
        
        return matching_concepts
    
    def get_related_concepts(self, concept_id: str, 
                           relation_type: str = None) -> List[Tuple[str, float]]:
        """Возвращает концепции связанные с указанной"""
        if concept_id not in self.concepts:
            return []
        
        related = []
        for relation_id, (c1, c2, rel_type, strength) in self.relations.items():
            if relation_type and rel_type != relation_type:
                continue
            
            if c1 == concept_id:
                related.append((c2, strength))
            elif c2 == concept_id:
                related.append((c1, strength))
        
        return related
    
    def activate_concept(self, concept_id: str, activation: float = 1.0, 
                     visited: Optional[set] = None):
        """Активирует концепцию и связанные с ней"""
        if visited is None:
            visited = set()
        
        if concept_id in visited or concept_id not in self.concepts:
            return
        
        visited.add(concept_id)
        
        # Активируем основную концепцию
        self.concepts[concept_id]['activation'] = min(1.0, 
            self.concepts[concept_id]['activation'] + activation)
        self.concepts[concept_id]['access_count'] += 1
        
        # Распространение активации на связанные концепции
        related = self.get_related_concepts(concept_id)
        for related_id, strength in related:
            spread_activation = activation * strength * 0.5
            self.activate_concept(related_id, spread_activation, visited)
    
    def decay_activations(self, decay_rate: float = 0.95):
        """Применяет затухание ко всем активациям"""
        for concept_data in self.concepts.values():
            concept_data['activation'] *= decay_rate
    
    def get_statistics(self) -> Dict[str, Any]:
        """Возвращает статистику семантической памяти"""
        if not self.concepts:
            return {}
        
        # Распределение типов концепций
        type_counts = defaultdict(int)
        for concept_data in self.concepts.values():
            type_counts[concept_data['type']] += 1
        
        # Распределение типов отношений
        relation_counts = defaultdict(int)
        for _, _, rel_type, _ in self.relations.values():
            relation_counts[rel_type] += 1
        
        # Средняя активация
        activations = [data['activation'] for data in self.concepts.values()]
        
        return {
            'total_concepts': len(self.concepts),
            'total_relations': len(self.relations),
            'concept_types': dict(type_counts),
            'relation_types': dict(relation_counts),
            'average_activation': np.mean(activations),
            'max_activation': max(activations),
            'most_accessed': max(
                self.concepts.items(), 
                key=lambda x: x[1]['access_count']
            )[0] if self.concepts else None
        }


class ProceduralMemory:
    """Процедурная память для хранения навыков и последовательностей действий"""
    
    def __init__(self):
        # Навыки: {skill_id: skill_data}
        self.skills: Dict[str, Dict] = {}
        
        # Последовательности: {sequence_id: sequence_data}
        self.sequences: Dict[str, Dict] = {}
        
        # Счетчики
        self.skill_counter = 0
        self.sequence_counter = 0
    
    def add_skill(self, action_sequence: List[str], context: Dict, 
                 success_rate: float = 0.0) -> str:
        """Добавляет навык (последовательность действий)"""
        skill_id = f"skill_{self.skill_counter}"
        self.skill_counter += 1
        
        self.skills[skill_id] = {
            'action_sequence': action_sequence,
            'context': context,
            'success_rate': success_rate,
            'usage_count': 0,
            'last_used': 0
        }
        
        return skill_id
    
    def update_skill_performance(self, skill_id: str, success: bool):
        """Обновляет показатели производительности навыка"""
        if skill_id not in self.skills:
            return
        
        skill = self.skills[skill_id]
        skill['usage_count'] += 1
        
        # Обновляем success_rate с экспоненциальным скользящим средним
        alpha = 0.1
        new_rate = 1.0 if success else 0.0
        skill['success_rate'] = (
            skill['success_rate'] * (1 - alpha) + new_rate * alpha
        )
    
    def find_relevant_skills(self, current_context: Dict) -> List[Tuple[str, float]]:
        """Находит релевантные навыки для текущего контекста"""
        relevant_skills = []
        
        for skill_id, skill_data in self.skills.items():
            relevance = self._calculate_context_relevance(
                current_context, skill_data['context']
            )
            
            if relevance > 0.1:  # Порог релевантности
                # Учитываем success_rate
                score = relevance * skill_data['success_rate']
                relevant_skills.append((skill_id, score))
        
        # Сортируем по релевантности
        relevant_skills.sort(key=lambda x: x[1], reverse=True)
        
        return relevant_skills
    
    def _calculate_context_relevance(self, context1: Dict, context2: Dict) -> float:
        """Рассчитывает релевантность контекстов"""
        if not context1 or not context2:
            return 0.0
        
        # Общие ключи
        common_keys = set(context1.keys()) & set(context2.keys())
        if not common_keys:
            return 0.0
        
        # Схожесть по общим ключам
        similarity_sum = 0.0
        for key in common_keys:
            val1, val2 = context1[key], context2[key]
            
            if isinstance(val1, (int, float)) and isinstance(val2, (int, float)):
                # Числовые значения
                diff = abs(val1 - val2)
                max_val = max(abs(val1), abs(val2), 1.0)
                similarity = 1.0 - (diff / max_val)
            else:
                # Категориальные значения
                similarity = 1.0 if val1 == val2 else 0.0
            
            similarity_sum += similarity
        
        return similarity_sum / len(common_keys)
    
    def add_sequence(self, actions: List[str], outcomes: List[bool]):
        """Добавляет последовательность действий с результатами"""
        sequence_id = f"sequence_{self.sequence_counter}"
        self.sequence_counter += 1
        
        self.sequences[sequence_id] = {
            'actions': actions,
            'outcomes': outcomes,
            'success_rate': sum(outcomes) / len(outcomes) if outcomes else 0.0,
            'length': len(actions)
        }
    
    def get_statistics(self) -> Dict[str, Any]:
        """Возвращает статистику процедурной памяти"""
        if not self.skills:
            return {}
        
        success_rates = [skill['success_rate'] for skill in self.skills.values()]
        usage_counts = [skill['usage_count'] for skill in self.skills.values()]
        
        return {
            'total_skills': len(self.skills),
            'total_sequences': len(self.sequences),
            'average_success_rate': np.mean(success_rates),
            'success_rate_std': np.std(success_rates),
            'total_skill_usage': sum(usage_counts),
            'most_used_skill': max(
                self.skills.items(), 
                key=lambda x: x[1]['usage_count']
            )[0] if self.skills else None
        }


class IntegratedMemorySystem:
    """Интегрированная система памяти для агента"""
    
    def __init__(self, agent):
        self.agent = agent
        
        # Различные типы памяти
        self.episodic_memory = AdvancedEpisodicMemory(agent.memory_capacity)
        self.semantic_memory = SemanticMemory()
        self.procedural_memory = ProceduralMemory()
        
        # Статистика
        self.memory_operations = 0
        self.consolidation_events = 0
    
    def add_experience(self, episode: Episode):
        """Добавляет опыт во все соответствующие системы памяти"""
        # Эпизодическая память
        self.episodic_memory.add_episode(episode)
        
        # Семантическая память - извлекаем концепции
        self._extract_concepts_from_episode(episode)
        
        # Процедурная память - извлекаем последовательности
        self._extract_procedures_from_episode(episode)
        
        self.memory_operations += 1
    
    def _extract_concepts_from_episode(self, episode: Episode):
        """Извлекает концепции из эпизода"""
        # Извлекаем объекты как концепции
        if 'objects' in episode.context:
            for obj_data in episode.context['objects']:
                concept_id = self.semantic_memory.add_concept(
                    concept_type='object',
                    features={
                        'type': obj_data.get('type'),
                        'properties': obj_data.get('properties', {})
                    }
                )
                
                # Активируем концепцию
                self.semantic_memory.activate_concept(concept_id)
    
    def _extract_procedures_from_episode(self, episode: Episode):
        """Извлекает процедуры из эпизода"""
        # Если эпизод успешный, добавляем как процедуру
        if episode.reward > 0:
            context = {
                'state_features': episode.state,
                'energy_level': episode.context.get('energy', 1.0)
            }
            
            skill_id = self.procedural_memory.add_skill(
                action_sequence=[episode.action],
                context=context,
                success_rate=1.0 if episode.success else 0.0
            )
            
            # Обновляем производительность
            self.procedural_memory.update_skill_performance(
                skill_id, episode.success
            )
    
    def consolidate_memories(self):
        """Консолидация памяти - перенос из эпизодической в семантическую"""
        # Находим важные эпизоды
        important_episodes = []
        for episode, priority in self.episodic_memory.episodes:
            if priority > 2.0:  # Порог важности
                important_episodes.append(episode)
        
        # Консолидируем важные эпизоды
        for episode in important_episodes:
            self._extract_concepts_from_episode(episode)
            self._extract_procedures_from_episode(episode)
        
        # Применяем затухание
        self.semantic_memory.decay_activations()
        
        self.consolidation_events += 1
    
    def get_comprehensive_statistics(self) -> Dict[str, Any]:
        """Возвращает полную статистику памяти"""
        episodic_stats = self.episodic_memory.get_memory_statistics()
        semantic_stats = self.semantic_memory.get_statistics()
        procedural_stats = self.procedural_memory.get_statistics()
        
        return {
            'memory_operations': self.memory_operations,
            'consolidation_events': self.consolidation_events,
            'episodic_memory': episodic_stats,
            'semantic_memory': semantic_stats,
            'procedural_memory': procedural_stats
        }
