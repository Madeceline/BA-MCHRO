"""Warm-start programs and a deterministic offline generation client."""
from __future__ import annotations

import hashlib


# Solver 暖启动的公开名称。它们是训练初始化程序，不是最终部署算法。
WARM_START_ALGORITHM_NAMES = (
    "variable_cost", "fixed_cost_greedy", "capacity_aware", "grasp",
    "genetic_algorithm", "simulated_annealing", "tabu_search",
)


BASE_SOLVER = '''def solve_snd(instance):
    # 残余容量最短路构造；可拆分商品，共享已开启弧的固定费用。
    # 对时间展开实例，nodes/arcs 已编码出发和到达时刻，普通最短路会自然遵守时间顺序。
    nodes, arcs = instance["nodes"], instance["arcs"]
    residual = {str(a["id"]): float(a["capacity"]) for a in arcs}
    opened = {str(a["id"]): 0.0 for a in arcs}
    flows = {}
    for commodity in sorted(instance["commodities"], key=lambda k: (-k["demand"], str(k["id"]))):
        demand = float(commodity["demand"])
        remaining = demand
        values = {str(a["id"]): 0.0 for a in arcs}
        while remaining > 0.00000001:
            distance = {node: float("inf") for node in nodes}
            previous = {}
            unused = list(nodes)
            distance[commodity["origin"]] = 0.0
            while unused:
                node = min(unused, key=lambda v: (distance[v], v))
                unused.remove(node)
                if node == commodity["destination"] or distance[node] == float("inf"):
                    break
                for arc in arcs:
                    key = str(arc["id"])
                    if arc["tail"] == node and arc["head"] in unused and residual[key] > 0.00000001:
                        weight = arc["variable_cost"] + 1.0 * arc["fixed_cost"] * (1.0-opened[key]) / demand
                        candidate = distance[node] + weight
                        if candidate < distance[arc["head"]]:
                            distance[arc["head"]] = candidate
                            previous[arc["head"]] = arc
            if commodity["destination"] not in previous:
                return {"arc_open": opened, "flows": flows}
            path = []
            node = commodity["destination"]
            while node != commodity["origin"]:
                arc = previous[node]
                path.append(str(arc["id"]))
                node = arc["tail"]
            amount = min(remaining, min(residual[key] for key in path))
            for key in path:
                values[key] += amount
                residual[key] -= amount
                opened[key] = 1.0
            remaining -= amount
        flows[str(commodity["id"])] = values
    return {"arc_open": opened, "flows": flows}
'''

BASE_GENERATOR = '''def generate_snd(base_instance, seed):
    return dict(base_instance)
'''


def _simple_warm_start_source(name: str) -> str:
    """Return a self-contained deterministic constructor for a named policy.

    The first four policies intentionally share the proven residual shortest-path
    decoder, while changing its arc score.  This keeps every warm start usable
    in the candidate subprocess and makes their source hashes distinct.
    """
    if name == "fixed_cost_greedy":
        return BASE_SOLVER
    old = 'weight = arc["variable_cost"] + 1.0 * arc["fixed_cost"] * (1.0-opened[key]) / demand'
    if name == "variable_cost":
        new = 'weight = arc["variable_cost"]'
    elif name == "capacity_aware":
        new = ('ratio = residual[key] / max(float(arc["capacity"]), 1.0)\n'
               '                        weight = (arc["variable_cost"] * '
               '(1.0 + 0.5 / max(ratio, 0.05)) + '
               'arc["fixed_cost"] * (1.0-opened[key]) / demand)')
    elif name == "grasp":
        # A deterministic multi-start-like perturbation keeps replay/cache keys
        # stable while exploring different residual shortest-path rankings.
        new = ('weight = (arc["variable_cost"] + '
               'arc["fixed_cost"] * (1.0-opened[key]) / demand) * '
               '(1.0 + (sum(ord(ch) for ch in key) % 7 - 3) * 0.02)')
    else:
        raise ValueError(f"not a simple warm-start algorithm: {name}")
    if old not in BASE_SOLVER:
        raise RuntimeError("BASE_SOLVER score expression changed; update warm-start variants")
    return BASE_SOLVER.replace(old, new)


def _metaheuristic_warm_start_source(name: str) -> str:
    """Use the project's bounded generic search implementation as a warm start.

    The source remains a normal candidate program; the import is resolved in the
    project execution environment and the returned incumbent is independently
    validated by SNDEvaluator.  Small settings keep initialization affordable.
    """
    return f'''def solve_snd(instance):
    from multi_agent_snd.benchmark_baselines import MetaheuristicConfig, run_metaheuristic
    config = MetaheuristicConfig(
        population_size=6, generations=8, mutation_rate=0.12, mutation_scale=0.5,
        sa_iterations=80, sa_initial_temperature=0.1, sa_cooling=0.98,
        tabu_iterations=30, tabu_neighbors=6, tabu_tenure=8)
    return run_metaheuristic(instance, {name!r}, timeout=8.0, seed=0, config=config)
'''


def warm_start_source(name: str) -> str:
    """Return source for one configured Solver warm start."""
    if name not in WARM_START_ALGORITHM_NAMES:
        raise ValueError(f"warm_start_algorithms must be from {WARM_START_ALGORITHM_NAMES}")
    if name in {"variable_cost", "fixed_cost_greedy", "capacity_aware", "grasp"}:
        return _simple_warm_start_source(name)
    return _metaheuristic_warm_start_source(name)


def warm_start_sources(names: tuple[str, ...] | list[str]) -> dict[str, str]:
    """Resolve and validate an ordered, user-selected warm-start collection."""
    names = tuple(names)
    if not names or len(set(names)) != len(names):
        raise ValueError("warm_start_algorithms must be a non-empty sequence without duplicates")
    return {name: warm_start_source(name) for name in names}


class OfflineMultiAgentClient:
    """根据 prompt 哈希产生确定性代码变体，仅用于集成测试和演示。"""

    def generate(self, prompt: str, temperature: float = 1.0, n: int = 1) -> list[str]:
        result = []
        for index in range(n):
            value = int(hashlib.sha256((prompt + str(index)).encode()).hexdigest()[:8], 16)
            if "PLAYER=solver\n" in prompt:
                factor = (value % 11 + 1) / 5
                code = BASE_SOLVER.replace("1.0 * arc", f"{factor} * arc")
                thought = f"残余容量最短路逐商品构造，固定费用权重 {factor}，优先共享已开启弧。可能因贪心路由失败。"
            else:
                factor = 0.75 + (value % 51) / 100
                # 只改固定成本，保证基础实例的可行性不变。
                code = ('def generate_snd(base_instance, seed):\n'
                        '    result = dict(base_instance)\n'
                        '    result["arcs"] = [dict(a) for a in base_instance["arcs"]]\n'
                        '    for arc in result["arcs"]:\n'
                        f'        if arc["fixed_cost"] > 0 and (int(arc["id"])+seed)%2:\n'
                        f'            arc["fixed_cost"] *= {factor}\n'
                        '    return result\n')
                thought = f"按弧 ID 和 seed 确定性扰动固定费用，倍率 {factor}，不改拓扑与可行域。"
            result.append(f"EXPLANATION: {thought}\n```python\n{code}\n```")
        return result
