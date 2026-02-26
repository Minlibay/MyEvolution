"""
Tests for environment module
"""

import pytest
import numpy as np
from src.core.environment import Environment, EnvironmentConfig
from src.core.objects import Object, ObjectFactory


class TestEnvironmentConfig:
    """Тесты конфигурации среды"""
    
    def test_default_config(self):
        """Тест конфигурации по умолчанию"""
        config = EnvironmentConfig()
        assert config.width == 100
        assert config.height == 100
        assert config.seed == 42
        assert config.resource_spawn_rate == 0.1


class TestEnvironment:
    """Тесты среды"""
    
    def setup_method(self):
        """Настройка тестовой среды"""
        config = EnvironmentConfig(width=10, height=10, seed=123)
        self.env = Environment(config)
    
    def test_initialization(self):
        """Тест инициализации среды"""
        assert self.env.width == 10
        assert self.env.height == 10
        assert self.env.season == 0
        assert self.env.timestep == 0
    
    def test_add_object(self):
        """Тест добавления объекта"""
        obj = ObjectFactory.create_object(
            'stone', (5, 5), 'test_obj', 0, 0
        )
        
        result = self.env.add_object(obj)
        assert result is True
        assert 'test_obj' in self.env.objects
        assert (5, 5) in self.env.grid
        assert 'test_obj' in self.env.grid[(5, 5)]
    
    def test_add_object_out_of_bounds(self):
        """Тест добавления объекта за пределами среды"""
        obj = ObjectFactory.create_object(
            'stone', (15, 5), 'test_obj', 0, 0
        )
        
        result = self.env.add_object(obj)
        assert result is False
        assert 'test_obj' not in self.env.objects
    
    def test_remove_object(self):
        """Тест удаления объекта"""
        obj = ObjectFactory.create_object(
            'stone', (5, 5), 'test_obj', 0, 0
        )
        self.env.add_object(obj)
        
        result = self.env.remove_object('test_obj')
        assert result is True
        assert 'test_obj' not in self.env.objects
        assert (5, 5) not in self.env.grid
    
    def test_get_local_environment(self):
        """Тест получения локальной среды"""
        obj = ObjectFactory.create_object(
            'stone', (5, 5), 'test_obj', 0, 0
        )
        self.env.add_object(obj)
        
        local_env = self.env.get_local_environment((5, 5), radius=1)
        # Проверяем, что наш объект есть в локальной среде
        object_ids = [obj.id for obj in local_env['objects']]
        assert 'test_obj' in object_ids
        assert local_env['position'] == (5, 5)
        assert local_env['season'] == 0
    
    def test_season_update(self):
        """Тест обновления сезона"""
        # Устанавливаем длину сезона для теста
        self.env.config.season_length = 5
        
        # Обновляем до смены сезона
        self.env.update(5)
        assert self.env.season == 1
        assert self.env.timestep == 5
        
        # Еще одна смена сезона
        self.env.update(10)
        assert self.env.season == 2
    
    def test_resource_spawning(self):
        """Тест появления ресурсов"""
        initial_count = len(self.env.objects)
        
        # Принудительно вызываем появление ресурсов
        self.env._spawn_resources(100)
        
        # Проверяем, что количество объектов не уменьшилось
        assert len(self.env.objects) >= initial_count
    
    def test_get_statistics(self):
        """Тест получения статистики"""
        obj = ObjectFactory.create_object(
            'stone', (5, 5), 'test_obj', 0, 0
        )
        self.env.add_object(obj)
        
        stats = self.env.get_statistics()
        # Проверяем, что статистика содержит камень
        assert 'stone' in stats['object_types']
        assert stats['object_types']['stone'] >= 1
        assert stats['season'] == 0
        assert 'occupation_rate' in stats


class TestObjectFactory:
    """Тесты фабрики объектов"""
    
    def test_create_object(self):
        """Тест создания объекта"""
        obj = ObjectFactory.create_object(
            'stone', (5, 5), 'test_obj', 0, 0
        )
        
        assert obj.id == 'test_obj'
        assert obj.type == 'stone'
        assert obj.position == (5, 5)
        assert obj.hardness > 0.8
        assert obj.nutrition < 0.2  # Камни имеют низкую питательность
        assert 0.0 <= obj.toxicity <= 1.0
    
    def test_seasonal_modifiers(self):
        """Тест сезонных модификаторов"""
        # Лето (сезон 1) должно увеличивать питательность
        obj_summer = ObjectFactory.create_object(
            'plant', (5, 5), 'test_summer', 0, season=1
        )
        
        # Зима (сезон 3) должно уменьшать питательность
        obj_winter = ObjectFactory.create_object(
            'plant', (5, 5), 'test_winter', 0, season=3
        )
        
        assert obj_summer.nutrition > obj_winter.nutrition
    
    def test_get_random_object_type(self):
        """Тест получения случайного типа объекта"""
        obj_type = ObjectFactory.get_random_object_type()
        assert obj_type in ObjectFactory.OBJECT_TEMPLATES


if __name__ == '__main__':
    pytest.main([__file__])
