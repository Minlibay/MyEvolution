"""FastAPI web server for real-time simulation visualization."""

from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path
from typing import Optional, Dict, Any, Set

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse

from ..utils.config_loader import load_config
from ..core.simulation import Simulation
from ..core.agent import AgentFactory


app = FastAPI(title="Evolution Simulation")

# Serve static files (sprites, js, css)
app.mount(
    "/static",
    StaticFiles(directory=Path(__file__).parent / "static"),
    name="static",
)


class SimulationController:
    def __init__(self):
        self.simulation: Optional[Simulation] = None
        self.step_delay_sec: float = 1.0
        self._task: Optional[asyncio.Task] = None
        self._lock = asyncio.Lock()
        self._clients: Set[WebSocket] = set()
        self._config_path: str = "config/experiment_configs/basic_experiment.yaml"

        # Run bookkeeping
        self.run_number: int = 0
        self.run_started_wall: float = 0.0
        self.history: list[dict[str, Any]] = []

    async def ensure_simulation(self):
        async with self._lock:
            if self.simulation is not None:
                return
            self._start_new_run_locked(reason=None)

    def _start_new_run_locked(self, reason: Optional[str]):
        """Start a new run. Must be called under _lock."""
        # Finalize previous run history entry
        if self.simulation is not None and self.simulation.state is not None and reason:
            prev_state = self.simulation.state
            env = prev_state.environment

            # Try to enrich reason with last death details.
            detail = None
            for e in reversed(getattr(prev_state, 'events_log', []) or []):
                if e.get('type') == 'agent_death':
                    detail = e
                    break

            reason_details = None
            if detail:
                cause = detail.get('cause', 'unknown')
                hunger = detail.get('hunger')
                thirst = detail.get('thirst')
                sleepiness = detail.get('sleepiness')
                energy = detail.get('energy')
                health = detail.get('health')
                age = detail.get('age')

                ru = {
                    'starvation': 'умер от голода',
                    'dehydration': 'умер от жажды',
                    'exhaustion': 'умер от истощения',
                    'old_age': 'умер от старости',
                    'health_collapse': 'умер от потери здоровья',
                    'unknown': 'причина неизвестна',
                }
                reason_details = {
                    'agent_id': detail.get('agent_id'),
                    'cause': cause,
                    'cause_ru': ru.get(cause, cause),
                    'hunger': hunger,
                    'thirst': thirst,
                    'sleepiness': sleepiness,
                    'energy': energy,
                    'health': health,
                    'age': age,
                }
            self.history.append(
                {
                    "run": int(self.run_number),
                    "reason": reason,
                    "reason_details": reason_details,
                    "ended_at_timestep": int(prev_state.timestep),
                    "ended_at_day": int(getattr(env, "day_count", 0)),
                    "ended_at_time": f"{int(getattr(env, 'hour', 0)):02d}:{int(getattr(env, 'minute', 0)):02d}",
                    "duration_sec_wall": float(max(0.0, time.time() - self.run_started_wall)),
                }
            )

        config = load_config(self._config_path)
        self.simulation = Simulation(config)
        self.simulation.initialize()
        if self.simulation.state is not None:
            self.simulation.state.max_steps = None

        self.run_number += 1
        self.run_started_wall = time.time()

        self._force_two_agents()

    def _force_two_agents(self):
        if self.simulation is None or self.simulation.state is None:
            return

        state = self.simulation.state
        state.agents.clear()

        # Place them deterministically into empty positions.
        empty_positions = state.environment.get_empty_positions(2)
        pos_male = empty_positions[0] if len(empty_positions) > 0 else (0, 0)
        pos_female = empty_positions[1] if len(empty_positions) > 1 else (1, 0)

        male = AgentFactory.create_random_agent("adam", pos_male)
        female = AgentFactory.create_random_agent("eva", pos_female)

        male.birth_time = 0
        female.birth_time = 0

        # Tag for UI / future modelling
        setattr(male, "sex", "male")
        setattr(female, "sex", "female")

        setattr(male, "display_name", "Адам")
        setattr(female, "display_name", "Ева")

        state.agents[male.id] = male
        state.agents[female.id] = female

        # Re-register learners
        state.learning_manager.learners.clear()
        state.learning_manager.register_agent(male)
        state.learning_manager.register_agent(female)

    async def reset(self):
        async with self._lock:
            self._start_new_run_locked(reason="ручной перезапуск")

    async def step_once(self):
        async with self._lock:
            if self.simulation is None or self.simulation.state is None:
                raise RuntimeError("Simulation is not initialized")
            self.simulation.state.step()

            # Auto-restart when population is extinct
            if len(self.simulation.state.agents) == 0:
                self._start_new_run_locked(reason="смерть Адама и Евы")

    async def get_snapshot(self) -> Dict[str, Any]:
        async with self._lock:
            if self.simulation is None or self.simulation.state is None:
                return {"initialized": False}

            state = self.simulation.state
            env = state.environment

            agents_payload = []
            for agent in state.agents.values():
                # Inventory summary for UI tooltip
                inv_counts: Dict[str, int] = {}
                food_count = 0
                try:
                    for obj_id in getattr(agent, 'inventory', []) or []:
                        obj = env.objects.get(obj_id)
                        if obj is None:
                            continue
                        qty = int(getattr(obj, 'quantity', 1) or 1)
                        inv_counts[obj.type] = inv_counts.get(obj.type, 0) + qty
                        try:
                            if obj.is_edible():
                                food_count += qty
                        except Exception:
                            pass
                except Exception:
                    inv_counts = {}
                    food_count = 0

                tools_payload = []
                try:
                    for tool_id in getattr(agent, 'tools', []) or []:
                        t = env.tools.get(tool_id)
                        if t is None:
                            continue
                        tools_payload.append(
                            {
                                'id': tool_id,
                                'kind': getattr(t, 'kind', None),
                                'tool_type': (t.get_tool_type() if hasattr(t, 'get_tool_type') else None),
                                'durability_left': float(getattr(t, 'durability_left', 0.0)),
                            }
                        )
                except Exception:
                    tools_payload = []

                rl = None
                try:
                    dm = state.learning_manager.get_decision_maker(agent.id)
                    if dm is not None:
                        # Current state approximation for UI
                        local_env = agent.perceive(env)
                        current_state = dm.state_encoder.encode_state(agent, local_env)
                        available_actions = state.action_executor.get_available_actions(agent, env)

                        q_actions = []
                        for a in available_actions:
                            q_actions.append({
                                'action': a,
                                'q': float(dm.learner.get_q_value(current_state, a)),
                            })
                        q_actions.sort(key=lambda x: x['q'], reverse=True)

                        stats = dm.get_decision_stats()
                        rl = {
                            'epsilon': float(stats.get('current_epsilon', agent.exploration_rate)),
                            'episodes': int(stats.get('learning_episodes', 0)),
                            'q_table_size': int(stats.get('q_table_size', 0)),
                            'avg_reward': float(stats.get('average_reward', 0.0)),
                            'top_actions': q_actions[:3],
                        }
                except Exception:
                    rl = None

                agents_payload.append(
                    {
                        "id": agent.id,
                        "name": getattr(agent, "display_name", agent.id),
                        "sex": getattr(agent, "sex", "unknown"),
                        "x": int(agent.position[0]),
                        "y": int(agent.position[1]),
                        "health": float(agent.health),
                        "energy": float(agent.energy),
                        "hunger": float(agent.hunger),
                        "thirst": float(getattr(agent, 'thirst', 0.0)),
                        "sleepiness": float(getattr(agent, 'sleepiness', 0.0)),
                        "age": int(agent.age),
                        "is_child": bool(getattr(agent, 'is_child', lambda: False)()),
                        "pregnant": bool(getattr(agent, 'pregnant', False)),
                        "pregnancy_remaining": int(getattr(agent, 'pregnancy_remaining', 0) or 0),
                        "mother_id": getattr(agent, 'mother_id', None),
                        "father_id": getattr(agent, 'father_id', None),
                        "exploration_rate": float(agent.exploration_rate),
                        "inventory_size": int(len(agent.inventory)),
                        "tools": int(len(agent.tools)),
                        "discoveries": int(len(agent.discoveries_made)),
                        "inventory": {
                            "by_type": inv_counts,
                            "food": int(food_count),
                        },
                        "tools_list": tools_payload,
                        "last_action": getattr(agent, "last_action", None),
                        "last_action_success": getattr(agent, "last_action_success", None),
                        "speech": getattr(agent, "last_utterance", None),
                        "speech_meaning": getattr(agent, "last_intended_meaning", None),
                        "heard": getattr(agent, "last_heard", None),
                        "rl": rl,
                    }
                )

            # Objects can be large; send a capped sample for UI.
            objects_payload = []
            max_objects = 500
            for i, obj in enumerate(env.objects.values()):
                if i >= max_objects:
                    break
                # Skip objects not present on the map (e.g., carried in inventory)
                if obj.position[0] < 0 or obj.position[1] < 0:
                    continue
                if obj.position[0] >= env.width or obj.position[1] >= env.height:
                    continue
                objects_payload.append(
                    {
                        "id": obj.id,
                        "type": obj.type,
                        "x": int(obj.position[0]),
                        "y": int(obj.position[1]),
                        "quantity": int(getattr(obj, 'quantity', 1) or 1),
                        "nutrition": float(getattr(obj, 'nutrition', 0.0) or 0.0),
                        "toxicity": float(getattr(obj, 'toxicity', 0.0) or 0.0),
                        "hardness": float(getattr(obj, 'hardness', 0.0) or 0.0),
                        "durability": float(getattr(obj, 'durability', 0.0) or 0.0),
                    }
                )

            events_tail = state.events_log[-200:]

            language_metrics = self._calculate_language_metrics(state, events_tail)

            agents_list = list(state.agents.values())
            needs_load_avg = 0.0
            critical_share = 0.0
            avg_age_alive = 0.0
            max_age_alive = 0
            if agents_list:
                loads = []
                critical = 0
                ages = []
                for a in agents_list:
                    h = float(getattr(a, 'hunger', 0.0) or 0.0)
                    t = float(getattr(a, 'thirst', 0.0) or 0.0)
                    s = float(getattr(a, 'sleepiness', 0.0) or 0.0)
                    loads.append((h + t + s) / 3.0)
                    if max(h, t, s) >= 0.85:
                        critical += 1
                    ages.append(int(getattr(a, 'age', 0) or 0))
                needs_load_avg = float(sum(loads) / max(1, len(loads)))
                critical_share = float(critical / max(1, len(agents_list)))
                avg_age_alive = float(sum(ages) / max(1, len(ages)))
                max_age_alive = int(max(ages) if ages else 0)

            death_causes_recent: Dict[str, int] = {}
            combine_success_recent = 0
            for e in events_tail:
                if e.get('type') == 'agent_death':
                    cause = e.get('cause', 'unknown')
                    death_causes_recent[cause] = int(death_causes_recent.get(cause, 0)) + 1
                if e.get('type') == 'agent_action' and e.get('action') == 'combine' and bool(e.get('success')):
                    combine_success_recent += 1

            tool_kinds: Dict[str, int] = {}
            for tool in getattr(env, 'tools', {}).values():
                k = getattr(tool, 'kind', None) or 'unknown'
                tool_kinds[k] = int(tool_kinds.get(k, 0)) + 1

            progress_metrics = {
                'needs_load_avg': float(needs_load_avg),
                'critical_share': float(critical_share),
                'death_causes_recent': death_causes_recent,
                'combine_success_recent': int(combine_success_recent),
                'total_tools': int(len(getattr(env, 'tools', {}) or {})),
                'tool_kinds': tool_kinds,
                'avg_age_alive': float(avg_age_alive),
                'max_age_alive': int(max_age_alive),
            }

            return {
                "initialized": True,
                "timestep": int(state.timestep),
                "world": {
                    "width": int(env.width),
                    "height": int(env.height),
                    "season": int(env.season),
                    "temperature": float(env.temperature),
                    "day": int(getattr(env, 'day_count', 0)),
                    "hour": int(getattr(env, 'hour', 0)),
                    "minute": int(getattr(env, 'minute', 0)),
                    "is_daytime": bool(getattr(env, 'is_daytime', True)),
                },
                "agents": agents_payload,
                "objects": objects_payload,
                "events": events_tail,
                "stats": state.get_statistics(),
                "language": language_metrics,
                "progress": progress_metrics,
                "run": {
                    "number": int(self.run_number),
                    "started_wall": float(self.run_started_wall),
                },
                "history": self.history[-20:],
            }

    def _calculate_language_metrics(self, state, events_tail):
        agents = list(state.agents.values())
        if not agents:
            return {
                "lexicon_size": 0,
                "stability": 0.0,
                "convergence": 0.0,
                "comm_success_rate_recent": 0.0,
                "comm_count_recent": 0,
            }

        # Lexicon size: how many tokens have meaningful weight.
        token_weight_threshold = 0.2
        token_set = set()
        stabilities = []

        for a in agents:
            out = getattr(a, "lexicon_out", {}) or {}
            for meaning, token_map in out.items():
                if not token_map:
                    continue
                total = float(sum(token_map.values()))
                if total <= 0:
                    continue
                top = max(token_map.values())
                stabilities.append(top / total)
                for tok, w in token_map.items():
                    if w >= token_weight_threshold:
                        token_set.add(tok)

        stability = float(sum(stabilities) / len(stabilities)) if stabilities else 0.0

        # Convergence: for each meaning, do agents agree on the top token?
        if len(agents) >= 2:
            a0 = agents[0]
            a1 = agents[1]
            m0 = getattr(a0, "lexicon_out", {}) or {}
            m1 = getattr(a1, "lexicon_out", {}) or {}
            meanings = set(m0.keys()) | set(m1.keys())
            agree = 0
            total_meanings = 0
            for meaning in meanings:
                t0 = m0.get(meaning) or {}
                t1 = m1.get(meaning) or {}
                if not t0 or not t1:
                    continue
                top0 = max(t0.items(), key=lambda kv: kv[1])[0]
                top1 = max(t1.items(), key=lambda kv: kv[1])[0]
                total_meanings += 1
                if top0 == top1:
                    agree += 1
            convergence = float(agree / total_meanings) if total_meanings else 0.0
        else:
            convergence = 0.0

        # Recent communication success rate
        comm_events = [e for e in events_tail if e.get("type") == "communication"]
        comm_count = len(comm_events)
        comm_success = sum(1 for e in comm_events if e.get("success"))
        comm_success_rate = float(comm_success / comm_count) if comm_count else 0.0

        return {
            "lexicon_size": int(len(token_set)),
            "stability": stability,
            "convergence": convergence,
            "comm_success_rate_recent": comm_success_rate,
            "comm_count_recent": int(comm_count),
        }

    async def add_client(self, ws: WebSocket):
        async with self._lock:
            self._clients.add(ws)

    async def remove_client(self, ws: WebSocket):
        async with self._lock:
            self._clients.discard(ws)

    async def _broadcast(self, payload: Dict[str, Any]):
        msg = json.dumps(payload)
        async with self._lock:
            clients = list(self._clients)

        to_remove = []
        for ws in clients:
            try:
                await ws.send_text(msg)
            except Exception:
                to_remove.append(ws)

        if to_remove:
            async with self._lock:
                for ws in to_remove:
                    self._clients.discard(ws)

    async def run_forever(self):
        await self.ensure_simulation()

        while True:
            # advance simulation
            try:
                await self.step_once()
            except Exception:
                # If something went wrong, reset and continue.
                await self.reset()

            snapshot = await self.get_snapshot()
            await self._broadcast(snapshot)
            await asyncio.sleep(self.step_delay_sec)


controller = SimulationController()


@app.on_event("startup")
async def _startup():
    await controller.ensure_simulation()
    controller._task = asyncio.create_task(controller.run_forever())


@app.get("/")
async def index():
    html_path = Path(__file__).parent / "static" / "index.html"
    return HTMLResponse(html_path.read_text(encoding="utf-8"))


@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    await ws.accept()

    await controller.add_client(ws)
    try:
        # Send one immediate snapshot on connect.
        snapshot = await controller.get_snapshot()
        await ws.send_text(json.dumps(snapshot))

        # Keep connection open; server pushes updates.
        while True:
            await ws.receive_text()
    except WebSocketDisconnect:
        await controller.remove_client(ws)
        return
