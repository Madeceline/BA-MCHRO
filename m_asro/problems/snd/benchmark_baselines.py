"""Offline SND comparison baselines; no training or LLM calls.

GA/SA/TS search random-key routing policies, not arbitrary arc-flow arrays.
The common decoder constructs capacity-feasible paths; only independently
validated complete solutions may become incumbents. These are transparent
generic baselines, not reproductions of specialised published SND algorithms.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import heapq
import math
import random
import time

from m_asro.problems.snd.evaluation import validate_solution


@dataclass(frozen=True)
class MetaheuristicConfig:
    population_size: int = 20
    generations: int = 50
    mutation_rate: float = 0.08
    mutation_scale: float = 0.5
    sa_iterations: int = 1000
    sa_initial_temperature: float = 0.1
    sa_cooling: float = 0.995
    tabu_iterations: int = 300
    tabu_neighbors: int = 12
    tabu_tenure: int = 15

    def __post_init__(self) -> None:
        for name in ("population_size", "generations", "sa_iterations",
                     "tabu_iterations", "tabu_neighbors", "tabu_tenure"):
            value = getattr(self, name)
            if type(value) is not int or value < (2 if name == "population_size" else 1):
                raise ValueError(f"{name} must be a positive integer (population >= 2)")
        for name in ("mutation_scale", "sa_initial_temperature"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive and finite")
        if not 0 < self.mutation_rate <= 1 or not 0 < self.sa_cooling <= 1:
            raise ValueError("mutation_rate and sa_cooling must be in (0, 1]")


class BaselineFailure(RuntimeError):
    """No validated incumbent, with diagnostics retained in the report."""

    def __init__(self, message: str, **diagnostics):
        super().__init__(message)
        self.diagnostics = diagnostics


def _check_deadline(deadline: float) -> None:
    if time.perf_counter() >= deadline:
        raise TimeoutError("comparison baseline wall budget reached")


class _RoutingSearch:
    def __init__(self, data: dict, seed: int, deadline: float):
        self.data, self.deadline = data, deadline
        self.rng = random.Random(seed)
        self.arcs, self.commodities = data["arcs"], data["commodities"]
        self.adjacency = {node: [] for node in data["nodes"]}
        for index, arc in enumerate(self.arcs):
            self.adjacency[arc["tail"]].append(index)
        self.k = len(self.commodities)
        # Keys for commodity order, log arc multipliers, log fixed-cost multiplier.
        self.dimension = self.k + len(self.arcs) + 1
        self.scale = max(1., sum(float(a["fixed_cost"]) for a in self.arcs)
                         + sum(float(k["demand"]) for k in self.commodities)
                         * sum(float(a["variable_cost"]) for a in self.arcs))
        self.cache: dict[tuple[float, ...], float] = {}
        self.best: dict | None = None
        self.best_objective = math.inf
        self.evaluations = self.cache_hits = self.feasible_evaluations = 0

    def initial(self, deterministic: bool = False) -> tuple[float, ...]:
        if not deterministic:
            return tuple([self.rng.random() for _ in range(self.k)]
                         + [self.rng.uniform(-1., 1.) for _ in self.arcs]
                         + [self.rng.uniform(-1., 1.)])
        order = sorted(range(self.k), key=lambda i: (-float(self.commodities[i]["demand"]),
                                                     str(self.commodities[i]["id"])))
        keys = [0.] * self.k
        for rank, index in enumerate(order):
            keys[index] = rank / max(self.k, 1)
        return tuple(keys + [0.] * (len(self.arcs) + 1))

    def neighbor(self, state: tuple[float, ...], scale: float) -> tuple[tuple[float, ...], tuple]:
        values = list(state)
        if self.k > 1 and self.rng.random() < .4:
            left, right = sorted(self.rng.sample(range(self.k), 2))
            values[left], values[right] = values[right], values[left]
            move = ("order", left, right)
        else:
            index = self.rng.randrange(self.k, self.dimension)
            # Discrete reversible moves give tabu attributes exact meanings.
            direction = self.rng.choice((-1, 1))
            changed = values[index] + direction * scale
            if abs(changed) > 3.:
                direction = -direction
            values[index] = max(-3., min(3., values[index] + direction * scale))
            move = ("weight", index, direction)
        return tuple(values), move

    def evaluate(self, state: tuple[float, ...]) -> float:
        _check_deadline(self.deadline)
        if state in self.cache:
            self.cache_hits += 1
            return self.cache[state]
        self.evaluations += 1
        residual = [float(a["capacity"]) for a in self.arcs]
        opened: set[int] = set()
        flows = {}
        factors = [math.exp(value) for value in state[self.k:-1]]
        fixed_factor = math.exp(state[-1])
        missing = 0.
        for ki in sorted(range(self.k), key=lambda i: (state[i], i)):
            _check_deadline(self.deadline)
            commodity = self.commodities[ki]
            source, target = commodity["origin"], commodity["destination"]
            remaining = float(commodity["demand"])
            flow: dict[str, float] = {}
            while remaining > 1e-9:
                _check_deadline(self.deadline)
                distances, previous = {source: 0.}, {}
                queue = [(0., source)]
                while queue:
                    _check_deadline(self.deadline)
                    distance, node = heapq.heappop(queue)
                    if distance != distances[node]:
                        continue
                    if node == target:
                        break
                    for ai in self.adjacency[node]:
                        if residual[ai] <= 1e-9:
                            continue
                        arc = self.arcs[ai]
                        cost = float(arc["variable_cost"])
                        if ai not in opened:
                            cost += fixed_factor * float(arc["fixed_cost"]) / max(remaining, 1e-9)
                        candidate = distance + max(1e-12, cost * factors[ai])
                        head = arc["head"]
                        if candidate < distances.get(head, math.inf):
                            distances[head], previous[head] = candidate, ai
                            heapq.heappush(queue, (candidate, head))
                if target not in previous:
                    missing += remaining
                    break
                path, node = [], target
                while node != source:
                    ai = previous[node]
                    path.append(ai)
                    node = self.arcs[ai]["tail"]
                amount = min(remaining, *(residual[ai] for ai in path))
                for ai in path:
                    key = str(self.arcs[ai]["id"])
                    flow[key] = flow.get(key, 0.) + amount
                    residual[ai] -= amount
                    opened.add(ai)
                remaining -= amount
            flows[str(commodity["id"])] = flow
        solution = {"arc_open": {str(self.arcs[ai]["id"]): 1. for ai in sorted(opened)},
                    "flows": flows}
        objective, checks = validate_solution(self.data, solution)
        if checks["feasible"]:
            self.feasible_evaluations += 1
            score = objective / self.scale
            if objective < self.best_objective:
                self.best_objective = objective
                self.best = {**solution, "objective": objective, "objective_value": objective,
                             "constraint_satisfaction": checks}
        else:
            # Partial flows guide exploration only; never exported as feasible.
            demand = sum(float(k["demand"]) for k in self.commodities)
            score = 1e6 * (1. + missing / max(demand, 1e-9))
        self.cache[state] = score
        return score


def _genetic(search: _RoutingSearch, config: MetaheuristicConfig) -> int:
    population = [search.initial(True)]
    population.extend(search.initial() for _ in range(config.population_size - 1))
    scored = [(search.evaluate(state), state) for state in population]
    for generation in range(config.generations):
        _check_deadline(search.deadline)
        scored.sort(key=lambda pair: pair[0])
        next_population = scored[:2]  # elitism; both may be infeasible during cold start
        while len(next_population) < config.population_size:
            left = min(search.rng.sample(scored, min(3, len(scored))), key=lambda p: p[0])[1]
            right = min(search.rng.sample(scored, min(3, len(scored))), key=lambda p: p[0])[1]
            child = []
            for index, (a, b) in enumerate(zip(left, right)):
                value = a if search.rng.random() < .5 else b  # uniform crossover
                if search.rng.random() < config.mutation_rate:
                    value += search.rng.gauss(0., config.mutation_scale)
                child.append(max(0., min(1., value)) if index < search.k
                             else max(-3., min(3., value)))
            state = tuple(child)
            next_population.append((search.evaluate(state), state))
        scored = next_population
    return config.generations


def _annealing(search: _RoutingSearch, config: MetaheuristicConfig) -> int:
    state = search.initial(True)
    score = search.evaluate(state)
    temperature = config.sa_initial_temperature
    for iteration in range(config.sa_iterations):
        proposal, _ = search.neighbor(state, config.mutation_scale)
        candidate = search.evaluate(proposal)
        delta = candidate - score
        if delta <= 0 or search.rng.random() < math.exp(-delta / max(temperature, 1e-12)):
            state, score = proposal, candidate
        temperature *= config.sa_cooling
    return config.sa_iterations


def _tabu(search: _RoutingSearch, config: MetaheuristicConfig) -> int:
    state = search.initial(True)
    best_score = search.evaluate(state)
    tabu: dict[tuple, int] = {}
    for iteration in range(config.tabu_iterations):
        chosen = None
        for _ in range(config.tabu_neighbors):
            proposal, move = search.neighbor(state, config.mutation_scale)
            score = search.evaluate(proposal)
            # Aspiration admits tabu moves only if they improve the global best.
            if tabu.get(move, -1) > iteration and score >= best_score:
                continue
            if chosen is None or score < chosen[0]:
                chosen = (score, proposal, move)
        if chosen is not None:
            score, state, move = chosen
            reverse = move if move[0] == "order" else (move[0], move[1], -move[2])
            tabu[reverse] = iteration + config.tabu_tenure + 1
            best_score = min(best_score, score)
        tabu = {move: until for move, until in tabu.items() if until > iteration}
    return config.tabu_iterations


def run_metaheuristic(data: dict, method: str, *, timeout: float = 120., seed: int = 1000,
                      config: MetaheuristicConfig | None = None) -> dict:
    algorithms = {"genetic_algorithm": _genetic, "simulated_annealing": _annealing,
                  "tabu_search": _tabu}
    if method not in algorithms:
        raise ValueError(f"Unknown metaheuristic: {method}")
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("timeout must be positive and finite")
    config = config or MetaheuristicConfig()
    started = time.perf_counter()
    search = _RoutingSearch(data, seed, started + timeout)
    interrupted, iterations = False, None
    try:
        iterations = algorithms[method](search, config)
    except TimeoutError:
        interrupted = True
    diagnostics = {"solver_used": method, "seed": seed, "search_config": asdict(config),
                   "candidate_evaluations": search.evaluations,
                   "feasible_evaluations": search.feasible_evaluations,
                   "cache_hits": search.cache_hits, "completed_iterations": iterations,
                   "terminated_by_budget": interrupted, "is_optimal": False,
                   "solve_time": time.perf_counter() - started}
    if search.best is None:
        raise BaselineFailure(f"{method}: no independently validated feasible incumbent", **diagnostics)
    return {**search.best, **diagnostics, "solver_status": "metaheuristic_feasible"}


def run_mip_baseline(data: dict, *, solver_type: str = "scipy", time_limit: float = 120.,
                     mip_gap: float = .01, wall_timeout: float = 150.) -> dict:
    """Use the original arc-flow MIP, without auto/heuristic reference fallback.

    Native time_limit covers optimisation. wall_timeout additionally caps process
    startup, model construction and IPC. A killed worker cannot return an incumbent.
    """
    from m_asro.runtime.execution import isolated_call

    if not math.isfinite(time_limit) or time_limit <= 0:
        raise ValueError("MIP time_limit must be positive and finite")
    if not math.isfinite(wall_timeout) or wall_timeout <= 0 or not 0 <= mip_gap <= 1:
        raise ValueError("invalid MIP wall_timeout or mip_gap")
    started = time.perf_counter()
    payload = isolated_call("reference_mip", (data, solver_type, time_limit, mip_gap, False),
                            timeout=wall_timeout)
    result = payload["result"]
    objective, checks = validate_solution(data, result)
    gap = result.get("mip_gap")
    diagnostics = {"solver_used": result["solver_used"], "requested_solver": solver_type,
                   "solver_status": result["status"], "solver_message": result.get("message"),
                   "mip_gap": gap, "solver_converged": result.get("is_optimal", False),
                   "is_optimal": bool(result.get("is_optimal") and gap is not None and gap <= 1e-9),
                   "fallback_reason": result.get("fallback_reason"),
                   "terminated_by_budget": result["status"] == "time_limit",
                   "solve_time": time.perf_counter() - started}
    if not checks["feasible"] or result.get("objective_value") is None:
        raise BaselineFailure("MIP returned no independently validated feasible incumbent", **diagnostics)
    return {**result, **diagnostics, "objective": objective, "objective_value": objective,
            "constraint_satisfaction": checks}
