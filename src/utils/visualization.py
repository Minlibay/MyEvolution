"""
Visualization module - comprehensive visualization tools for the simulation
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional, Tuple
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

from .metrics import SimulationMetrics, MetricsAnalyzer


class SimulationVisualizer:
    """Основной класс визуализации симуляции"""
    
    def __init__(self, figsize: Tuple[int, int] = (15, 10)):
        self.figsize = figsize
        self.color_palette = sns.color_palette("husl", 10)
        plt.style.use('seaborn-v0_8')
    
    def plot_world_state(self, environment, agents, timestamp: int, 
                        save_path: Optional[str] = None):
        """Визуализирует состояние мира"""
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # Рисуем сетку
        ax.set_xlim(0, environment.width)
        ax.set_ylim(0, environment.height)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        
        # Рисуем объекты
        for obj_id, obj in environment.objects.items():
            color = self._get_object_color(obj.type)
            size = self._get_object_size(obj)
            ax.scatter(obj.position[0], obj.position[1], 
                      c=color, s=size, alpha=0.7, edgecolors='black', linewidth=0.5)
        
        # Рисуем агентов
        for agent in agents:
            color = self._get_agent_color(agent)
            ax.scatter(agent.position[0], agent.position[1], 
                      c=color, s=100, marker='o', edgecolors='black', linewidth=2)
            
            # Добавляем информацию об агенте
            ax.annotate(f"A{agent.id[-1]}", 
                        (agent.position[0], agent.position[1]), 
                        fontsize=8, ha='center', va='center')
        
        # Добавляем легенду
        self._add_world_legend(ax)
        
        # Заголовок
        ax.set_title(f'Состояние мира - Шаг {timestamp}\n'
                    f'Сезон: {environment.season}, Температура: {environment.temperature:.1f}°C',
                    fontsize=14, fontweight='bold')
        
        ax.set_xlabel('X координата')
        ax.set_ylabel('Y координата')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_metrics_timeline(self, metrics_history: List[SimulationMetrics], 
                            save_path: Optional[str] = None):
        """Визуализирует метрики во времени"""
        if not metrics_history:
            return
        
        timestamps = [m.timestamp for m in metrics_history]
        
        fig, axes = plt.subplots(3, 3, figsize=(20, 15))
        axes = axes.flatten()
        
        # Популяционные метрики
        axes[0].plot(timestamps, [m.population_size for m in metrics_history], 
                   color=self.color_palette[0], linewidth=2)
        axes[0].set_title('Размер популяции')
        axes[0].set_ylabel('Количество агентов')
        
        axes[1].plot(timestamps, [m.average_health for m in metrics_history], 
                   color=self.color_palette[1], linewidth=2)
        axes[1].set_title('Среднее здоровье')
        axes[1].set_ylabel('Здоровье')
        
        axes[2].plot(timestamps, [m.average_age for m in metrics_history], 
                   color=self.color_palette[2], linewidth=2)
        axes[2].set_title('Средний возраст')
        axes[2].set_ylabel('Возраст')
        
        # Технологические метрики
        axes[3].plot(timestamps, [m.total_discoveries for m in metrics_history], 
                   color=self.color_palette[3], linewidth=2)
        axes[3].set_title('Общие открытия')
        axes[3].set_ylabel('Количество открытий')
        
        axes[4].plot(timestamps, [m.total_tools for m in metrics_history], 
                   color=self.color_palette[4], linewidth=2)
        axes[4].set_title('Общее количество инструментов')
        axes[4].set_ylabel('Количество инструментов')
        
        axes[5].plot(timestamps, [m.average_tool_complexity for m in metrics_history], 
                   color=self.color_palette[5], linewidth=2)
        axes[5].set_title('Сложность инструментов')
        axes[5].set_ylabel('Сложность')
        
        # Генетические метрики
        axes[6].plot(timestamps, [m.genetic_diversity for m in metrics_history], 
                   color=self.color_palette[6], linewidth=2)
        axes[6].set_title('Генетическое разнообразие')
        axes[6].set_ylabel('Разнообразие')
        
        axes[7].plot(timestamps, [m.average_intelligence for m in metrics_history], 
                   color=self.color_palette[7], linewidth=2)
        axes[7].set_title('Средний интеллект')
        axes[7].set_ylabel('Интеллект')
        
        axes[8].plot(timestamps, [m.average_exploration_rate for m in metrics_history], 
                   color=self.color_palette[8], linewidth=2)
        axes[8].set_title('Исследовательское поведение')
        axes[8].set_ylabel('Exploration Rate')
        
        # Общий заголовок
        fig.suptitle('Метрики симуляции во времени', fontsize=16, fontweight='bold')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_agent_behavior(self, agent, save_path: Optional[str] = None):
        """Визуализирует поведение отдельного агента"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Физиологические параметры
        timestamps = list(range(agent.age + 1))
        
        axes[0, 0].plot(timestamps, [agent.health] * len(timestamps), 
                       label='Здоровье', color='green', linewidth=2)
        axes[0, 0].plot(timestamps, [agent.energy] * len(timestamps), 
                       label='Энергия', color='blue', linewidth=2)
        axes[0, 0].plot(timestamps, [agent.hunger] * len(timestamps), 
                       label='Голод', color='red', linewidth=2)
        axes[0, 0].set_title('Физиологические параметры')
        axes[0, 0].set_ylabel('Значение')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Гены
        genes = agent.genes.to_dict()
        gene_names = list(genes.keys())
        gene_values = list(genes.values())
        
        axes[0, 1].bar(gene_names, gene_values, color=self.color_palette[:len(gene_names)])
        axes[0, 1].set_title('Генетические параметры')
        axes[0, 1].set_ylabel('Значение')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # Q-таблица распределение
        if agent.q_table:
            q_values = list(agent.q_table.values())
            axes[1, 0].hist(q_values, bins=20, color=self.color_palette[3], alpha=0.7)
            axes[1, 0].set_title('Распределение Q-значений')
            axes[1, 0].set_xlabel('Q-значение')
            axes[1, 0].set_ylabel('Частота')
        else:
            axes[1, 0].text(0.5, 0.5, 'Нет Q-значений', 
                           ha='center', va='center', transform=axes[1, 0].transAxes)
            axes[1, 0].set_title('Распределение Q-значений')
        
        # Статистическая память
        if agent.statistical_memory.statistics:
            memory_keys = list(agent.statistical_memory.statistics.keys())[:10]
            memory_values = [agent.statistical_memory.get_statistic(key) for key in memory_keys]
            
            axes[1, 1].barh(memory_keys, memory_values, color=self.color_palette[4])
            axes[1, 1].set_title('Статистическая память (топ-10)')
            axes[1, 1].set_xlabel('Значение')
        else:
            axes[1, 1].text(0.5, 0.5, 'Нет статистической памяти', 
                           ha='center', va='center', transform=axes[1, 1].transAxes)
            axes[1, 1].set_title('Статистическая память')
        
        fig.suptitle(f'Агент {agent.id} - Поведенческий профиль', 
                    fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_correlation_matrix(self, metrics_history: List[SimulationMetrics], 
                             save_path: Optional[str] = None):
        """Визуализирует корреляционную матрицу метрик"""
        if not metrics_history:
            return
        
        # Собираем данные в DataFrame
        data = {
            'population_size': [m.population_size for m in metrics_history],
            'average_health': [m.average_health for m in metrics_history],
            'average_intelligence': [m.average_intelligence for m in metrics_history],
            'genetic_diversity': [m.genetic_diversity for m in metrics_history],
            'total_discoveries': [m.total_discoveries for m in metrics_history],
            'total_tools': [m.total_tools for m in metrics_history],
            'action_diversity': [m.action_diversity for m in metrics_history],
            'average_exploration_rate': [m.average_exploration_rate for m in metrics_history],
            'resource_density': [m.resource_density for m in metrics_history]
        }
        
        df = pd.DataFrame(data)
        
        # Расчет корреляционной матрицы
        corr_matrix = df.corr()
        
        # Визуализация
        fig, ax = plt.subplots(figsize=(12, 10))
        
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0,
                   square=True, fmt='.2f', cbar_kws={'label': 'Корреляция'})
        
        ax.set_title('Корреляционная матрица метрик', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_technology_evolution(self, metrics_history: List[SimulationMetrics], 
                                save_path: Optional[str] = None):
        """Визуализирует эволюцию технологий"""
        if not metrics_history:
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        timestamps = [m.timestamp for m in metrics_history]
        
        # Открытия во времени
        discoveries = [m.total_discoveries for m in metrics_history]
        axes[0, 0].plot(timestamps, discoveries, color=self.color_palette[0], linewidth=2)
        axes[0, 0].fill_between(timestamps, 0, discoveries, alpha=0.3, color=self.color_palette[0])
        axes[0, 0].set_title('Накопленные открытия')
        axes[0, 0].set_ylabel('Количество открытий')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Скорость открытий
        if len(discoveries) > 1:
            discovery_rate = np.diff(discoveries)
            axes[0, 1].plot(timestamps[1:], discovery_rate, color=self.color_palette[1], linewidth=2)
            axes[0, 1].axhline(y=0, color='black', linestyle='--', alpha=0.5)
            axes[0, 1].set_title('Скорость открытий')
            axes[0, 1].set_ylabel('Открытий за шаг')
            axes[0, 1].grid(True, alpha=0.3)
        
        # Инструменты
        tools = [m.total_tools for m in metrics_history]
        tool_types = [m.unique_tool_types for m in metrics_history]
        
        axes[1, 0].plot(timestamps, tools, color=self.color_palette[2], 
                       label='Всего инструментов', linewidth=2)
        axes[1, 0].plot(timestamps, tool_types, color=self.color_palette[3], 
                       label='Уникальных типов', linewidth=2)
        axes[1, 0].set_title('Инструменты')
        axes[1, 0].set_ylabel('Количество')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Сложность инструментов
        complexity = [m.average_tool_complexity for m in metrics_history]
        axes[1, 1].plot(timestamps, complexity, color=self.color_palette[4], linewidth=2)
        axes[1, 1].fill_between(timestamps, 0, complexity, alpha=0.3, color=self.color_palette[4])
        axes[1, 1].set_title('Средняя сложность инструментов')
        axes[1, 1].set_ylabel('Сложность')
        axes[1, 1].grid(True, alpha=0.3)
        
        fig.suptitle('Эволюция технологий', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_evolution_dynamics(self, metrics_history: List[SimulationMetrics], 
                              save_path: Optional[str] = None):
        """Визуализирует эволюционную динамику"""
        if not metrics_history:
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        timestamps = [m.timestamp for m in metrics_history]
        
        # Размер популяции и генетическое разнообразие
        population = [m.population_size for m in metrics_history]
        diversity = [m.genetic_diversity for m in metrics_history]
        
        ax1 = axes[0, 0]
        ax1_twin = ax1.twinx()
        
        line1 = ax1.plot(timestamps, population, color=self.color_palette[0], 
                       label='Популяция', linewidth=2)
        line2 = ax1_twin.plot(timestamps, diversity, color=self.color_palette[1], 
                              label='Разнообразие', linewidth=2)
        
        ax1.set_xlabel('Время')
        ax1.set_ylabel('Размер популяции', color=self.color_palette[0])
        ax1_twin.set_ylabel('Генетическое разнообразие', color=self.color_palette[1])
        
        ax1.tick_params(axis='y', labelcolor=self.color_palette[0])
        ax1_twin.tick_params(axis='y', labelcolor=self.color_palette[1])
        
        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        ax1.legend(lines, labels, loc='upper left')
        
        axes[0, 0].set_title('Популяция и разнообразие')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Интеллект и исследовательское поведение
        intelligence = [m.average_intelligence for m in metrics_history]
        exploration = [m.average_exploration_rate for m in metrics_history]
        
        axes[0, 1].plot(timestamps, intelligence, color=self.color_palette[2], 
                       label='Интеллект', linewidth=2)
        axes[0, 1].plot(timestamps, exploration, color=self.color_palette[3], 
                       label='Исследование', linewidth=2)
        axes[0, 1].set_title('Когнитивные параметры')
        axes[0, 1].set_ylabel('Значение')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Распределение генов
        if metrics_history:
            latest_metrics = metrics_history[-1]
            gene_names = ['intelligence', 'exploration_bias', 'strength']
            gene_values = [
                latest_metrics.average_intelligence,
                latest_metrics.average_exploration_bias,
                latest_metrics.average_strength
            ]
            
            axes[1, 0].bar(gene_names, gene_values, color=self.color_palette[:3])
            axes[1, 0].set_title('Текущее распределение генов')
            axes[1, 0].set_ylabel('Среднее значение')
        
        # Эволюция разнообразия
        axes[1, 1].plot(timestamps, diversity, color=self.color_palette[4], linewidth=2)
        axes[1, 1].fill_between(timestamps, 0, diversity, alpha=0.3, color=self.color_palette[4])
        
        # Добавляем тренд
        if len(diversity) > 1:
            z = np.polyfit(timestamps, diversity, 1)
            p = np.poly1d(z)
            axes[1, 1].plot(timestamps, p(timestamps), "--", color='red', alpha=0.8)
        
        axes[1, 1].set_title('Эволюция генетического разнообразия')
        axes[1, 1].set_ylabel('Разнообразие')
        axes[1, 1].grid(True, alpha=0.3)
        
        fig.suptitle('Эволюционная динамика', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def _get_object_color(self, obj_type: str) -> str:
        """Возвращает цвет для типа объекта"""
        colors = {
            'stone': 'gray',
            'wood': 'brown',
            'plant': 'green',
            'berry': 'red',
            'bone': 'white',
            'fiber': 'yellow',
            'animal': 'orange'
        }
        return colors.get(obj_type, 'blue')
    
    def _get_object_size(self, obj) -> float:
        """Возвращает размер для объекта"""
        base_size = 20
        size_multiplier = 1 + obj.weight
        return base_size * size_multiplier
    
    def _get_agent_color(self, agent) -> str:
        """Возвращает цвет для агента based on здоровье"""
        if agent.health > 0.7:
            return 'green'
        elif agent.health > 0.4:
            return 'yellow'
        else:
            return 'red'
    
    def _add_world_legend(self, ax):
        """Добавляет легенду для мира"""
        legend_elements = [
            plt.scatter([], [], c='gray', s=30, label='Камень'),
            plt.scatter([], [], c='brown', s=30, label='Дерево'),
            plt.scatter([], [], c='green', s=30, label='Растение'),
            plt.scatter([], [], c='red', s=30, label='Ягода'),
            plt.scatter([], [], c='blue', s=100, marker='o', label='Агент (здоров)'),
            plt.scatter([], [], c='yellow', s=100, marker='o', label='Агент (ранен)'),
            plt.scatter([], [], c='red', s=100, marker='o', label='Агент (болен)')
        ]
        
        ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.15, 1))


class InteractiveVisualizer:
    """Интерактивная визуализация с Plotly"""
    
    def create_interactive_dashboard(self, metrics_history: List[SimulationMetrics]) -> go.Figure:
        """Создает интерактивный дашборд"""
        if not metrics_history:
            return go.Figure()
        
        # Создаем подграфики
        fig = make_subplots(
            rows=3, cols=3,
            subplot_titles=(
                'Популяция', 'Здоровье', 'Возраст',
                'Открытия', 'Инструменты', 'Разнообразие',
                'Интеллект', 'Исследование', 'Ресурсы'
            ),
            specs=[[{"secondary_y": True}, {"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}, {"secondary_y": False}]]
        )
        
        timestamps = [m.timestamp for m in metrics_history]
        
        # Популяция
        fig.add_trace(
            go.Scatter(x=timestamps, y=[m.population_size for m in metrics_history],
                      name='Популяция', line=dict(color='blue')),
            row=1, col=1
        )
        
        # Здоровье
        fig.add_trace(
            go.Scatter(x=timestamps, y=[m.average_health for m in metrics_history],
                      name='Здоровье', line=dict(color='green')),
            row=1, col=2
        )
        
        # Возраст
        fig.add_trace(
            go.Scatter(x=timestamps, y=[m.average_age for m in metrics_history],
                      name='Возраст', line=dict(color='orange')),
            row=1, col=3
        )
        
        # Открытия
        fig.add_trace(
            go.Scatter(x=timestamps, y=[m.total_discoveries for m in metrics_history],
                      name='Открытия', line=dict(color='purple')),
            row=2, col=1
        )
        
        # Инструменты
        fig.add_trace(
            go.Scatter(x=timestamps, y=[m.total_tools for m in metrics_history],
                      name='Инструменты', line=dict(color='red')),
            row=2, col=2
        )
        
        # Генетическое разнообразие
        fig.add_trace(
            go.Scatter(x=timestamps, y=[m.genetic_diversity for m in metrics_history],
                      name='Разнообразие', line=dict(color='cyan')),
            row=2, col=3
        )
        
        # Интеллект
        fig.add_trace(
            go.Scatter(x=timestamps, y=[m.average_intelligence for m in metrics_history],
                      name='Интеллект', line=dict(color='magenta')),
            row=3, col=1
        )
        
        # Исследование
        fig.add_trace(
            go.Scatter(x=timestamps, y=[m.average_exploration_rate for m in metrics_history],
                      name='Исследование', line=dict(color='brown')),
            row=3, col=2
        )
        
        # Ресурсы
        fig.add_trace(
            go.Scatter(x=timestamps, y=[m.resource_density for m in metrics_history],
                      name='Плотность ресурсов', line=dict(color='black')),
            row=3, col=3
        )
        
        fig.update_layout(
            title='Интерактивный дашборд симуляции',
            height=900,
            showlegend=False
        )
        
        return fig
    
    def create_3d_evolution_plot(self, metrics_history: List[SimulationMetrics]) -> go.Figure:
        """Создает 3D график эволюции"""
        if not metrics_history:
            return go.Figure()
        
        # Собираем данные
        time_points = [m.timestamp for m in metrics_history]
        intelligence = [m.average_intelligence for m in metrics_history]
        discoveries = [m.total_discoveries for m in metrics_history]
        diversity = [m.genetic_diversity for m in metrics_history]
        
        fig = go.Figure(data=[go.Scatter3d(
            x=time_points,
            y=intelligence,
            z=discoveries,
            mode='markers',
            marker=dict(
                size=5,
                color=diversity,
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title="Генетическое разнообразие")
            ),
            text=[f'Шаг: {t}<br>Интеллект: {i:.2f}<br>Открытия: {d}<br>Разнообразие: {div:.2f}' 
                  for t, i, d, div in zip(time_points, intelligence, discoveries, diversity)],
            hovertemplate='%{text}<extra></extra>'
        )])
        
        fig.update_layout(
            title='3D визуализация эволюции',
            scene=dict(
                xaxis_title='Время',
                yaxis_title='Интеллект',
                zaxis_title='Открытия'
            )
        )
        
        return fig
