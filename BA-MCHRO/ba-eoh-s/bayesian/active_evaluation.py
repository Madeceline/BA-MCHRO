"""Budgeted, paired active evaluation, independent of the M-PSRO training loop."""
from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
import math
import time
from typing import Any, Protocol

import numpy as np

from ba_eoh_s.bayesian.bayesian_cpm import BayesianCPMConfig, _exclusion, _key, bayesian_cpm_select
from ba_eoh_s.bayesian.features import instance_features
from ba_eoh_s.bayesian.linear_model import BayesianPerformanceModel
from ba_eoh_s.bayesian.observations import EvaluationObservation, ObservationStore, _hash_instance, observation_from_evaluation
from ba_eoh_s.bayesian.posterior import target_weights
from ba_eoh_s.bayesian.failure_policy import penalize_failures, rejection_reason, selection_evidence
from m_asro.runtime.progress import terminal_heartbeat


@dataclass(frozen=True)
class ActiveInstance:
    data: Any
    instance_hash: str
    features: dict[str, float]
    seed: int
    sample_index: int = 0
    evaluation_split: str = "training"

    def __post_init__(self) -> None:
        if (not self.instance_hash or type(self.seed) is not int
                or type(self.sample_index) is not int or self.sample_index < 0
                or self.evaluation_split not in {"training", "validation", "test"}
                or not self.features or any(not math.isfinite(v) for v in self.features.values())):
            raise ValueError("invalid active instance metadata")
        if self.data is not None:
            self.validate_contents()

    def validate_contents(self) -> None:
        if self.data is not None and (_hash_instance(self.data) != self.instance_hash
                                      or instance_features(self.data) != self.features):
            raise ValueError("instance contents do not match hash/features")

    @classmethod
    def from_snd(cls, data: Any, *, seed: int, sample_index: int = 0) -> "ActiveInstance":
        data = data.to_dict() if hasattr(data, "to_dict") else data
        return cls(data, _hash_instance(data), instance_features(data), seed, sample_index)

    def model_input(self) -> dict:
        return {"instance_hash": self.instance_hash, "instance_features": self.features,
                "evaluation_split": self.evaluation_split}


@dataclass(frozen=True)
class EvaluationOutcome:
    observation: EvaluationObservation
    executed: bool
    cache_hit: bool


class EvaluationBackend(Protocol):
    configuration_signature: str

    def peek(self, program: Any, instance: ActiveInstance) -> EvaluationObservation | None: ...

    def evaluate(self, program: Any, instance: ActiveInstance) -> EvaluationOutcome: ...


class SNDEvaluationBackend:
    """Adapter for an EXCLUSIVELY owned SNDEvaluator, no evaluator changes.

    Real cached results can be restored once as evidence with cache_hit=True.
    Their origin is still real execution, never posterior prediction. Current
    run costs are counted from receipts, not from the historical ledger.
    Do not share this evaluator with unrelated concurrently running searches.
    """

    def __init__(self, evaluator: Any) -> None:
        from snd.evaluation import digest
        self.evaluator = evaluator
        self.configuration_signature = digest({
            "reference_signature": evaluator.reference_signature, "timeout": evaluator.timeout,
            "utility": asdict(evaluator.utility), "seed": evaluator.seed,
        })

    def _observation(self, program: Any, instance: ActiveInstance, row: Mapping,
                     cache_hit: bool) -> EvaluationObservation:
        return observation_from_evaluation(
            program, instance.data, row, evaluation_split="training", seed=instance.seed,
            sample_index=instance.sample_index, configuration_signature=self.configuration_signature,
            cache_hit=cache_hit, observation_source="real_execution")

    def peek(self, program: Any, instance: ActiveInstance) -> EvaluationObservation | None:
        instance.validate_contents()
        with self.evaluator._cache_lock:
            row = self.evaluator.evaluations.get((_key(program), instance.instance_hash))
        return None if row is None else self._observation(program, instance, row, True)

    def evaluate(self, program: Any, instance: ActiveInstance) -> EvaluationOutcome:
        existing = self.peek(program, instance)
        if existing is not None:
            return EvaluationOutcome(existing, False, True)
        # Failures (including a failed reference attempt) spend one reserved
        # evaluation. There is no free retry outside the budget.
        started = time.perf_counter()
        try:
            row = self.evaluator.evaluate(program, instance.data)
        except Exception as exc:
            row = {"status": "solver_failure", "gap": None,
                   "failure_type": getattr(exc, "failure_type", "evaluation_error"),
                   "error": str(exc), "wall_time": time.perf_counter() - started}
        return EvaluationOutcome(self._observation(program, instance, row, False), True, False)


@dataclass(frozen=True)
class ActiveEvaluationConfig:
    method: str = "ucb"
    total_budget: int = 24  # INCLUDES initial runs AND all paired controls.
    initial_per_program: int = 2
    batch_size: int = 1  # Paired actions, not individual free evaluations.
    min_exploration_fraction: float = .2
    kappa: float = 1.0
    max_wall_time: float = 300.0
    information_threshold: float = 0.0  # Decision-score/time proxy, not EVI.
    default_evaluation_cost: float = 1.0
    random_seed: int = 42
    progress_output: bool = True
    cost_mode: str = "observed"  # "fixed" gives load-independent prompt replay.

    def __post_init__(self) -> None:
        if self.method not in {"ucb", "random", "uniform"}:
            raise ValueError("method must be ucb, random or uniform (no nested EVI)")
        if self.cost_mode not in {"observed", "fixed"}:
            raise ValueError("cost_mode must be observed or fixed")
        for name in ("total_budget", "initial_per_program", "batch_size", "random_seed"):
            value = getattr(self, name)
            minimum = 0 if name in {"total_budget", "random_seed"} else 1
            if type(value) is not int or value < minimum:
                raise ValueError(f"invalid {name}")
        if not math.isfinite(self.min_exploration_fraction) or not 0 <= self.min_exploration_fraction <= 1:
            raise ValueError("min_exploration_fraction must be in [0,1]")
        for name in ("kappa", "information_threshold"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        for name in ("max_wall_time", "default_evaluation_cost"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if type(self.progress_output) is not bool:
            raise ValueError("progress_output must be bool")


class BayesianActiveEvaluator:
    """Serial controller: a full candidate/incumbent bundle is budget-admitted.

    The wall limit prevents admission of further calls; an already running
    isolated call retains the evaluator's original timeout and can overrun
    this wall boundary. No execution restriction is relaxed or bypassed.
    """

    def __init__(self, model: BayesianPerformanceModel, backend: EvaluationBackend, *,
                 config: ActiveEvaluationConfig | None = None,
                 cpm_config: BayesianCPMConfig | None = None,
                 store: ObservationStore | None = None,
                 event_callback: Callable[[dict], None] | None = None) -> None:
        self.model, self.backend = model, backend
        self.config = config or ActiveEvaluationConfig()
        self.cpm_config = cpm_config or BayesianCPMConfig(n_draws=512)
        self.store = store if store is not None else ObservationStore()
        self.event_callback = event_callback
        if model.training_split != "training":
            raise ValueError("active evaluation requires a training model")
        if self.config.initial_per_program < self.cpm_config.min_observations:
            raise ValueError("initial_per_program must cover the CPM cold-start gate")
        self.events: list[dict] = []
        self._used = False

    def _emit(self, event: str, **details) -> None:
        row = {"event": event, "wall_seconds": time.perf_counter() - self.started,
               "actual_evaluation_count": self.count, "budget": self.config.total_budget,
               "remaining_budget": self.config.total_budget - self.count, **details}
        self.events.append(row)
        if self.event_callback:
            self.event_callback(row)
        if self.config.progress_output:
            text = f"[BA评估/{self.config.method}] {event} 真实={self.count}/{self.config.total_budget} 缓存={self.cache_hits} 耗时={row['wall_seconds']:.1f}s"
            if "program_id" in details:
                text += f" 程序={details['program_id'][:12]}"
            if "status" in details:
                text += f" 状态={details['status']}"
            print(text, flush=True)

    def _accept(self, row: EvaluationObservation, pid: str, instance: ActiveInstance) -> None:
        if (row.code_hash != pid or row.instance_hash != instance.instance_hash
                or row.evaluation_split != "training" or row.observation_source != "real_execution"
                or row.configuration_signature != self.backend.configuration_signature
                or row.instance_features != instance.features):
            raise ValueError("backend returned mismatched, non-training or non-real observation")
        previous = self.known.get((pid, instance.instance_hash))
        if previous is not None and (previous.observed_performance != row.observed_performance
                                    or previous.feasibility_status != row.feasibility_status):
            raise ValueError("conflicting deterministic observations")
        self.known[(pid, instance.instance_hash)] = row
        # Deduplicate across generator/sample seeds as well as cache reads.
        if not any(r.code_hash == pid and r.instance_hash == row.instance_hash
                   and r.configuration_signature == row.configuration_signature
                   for r in self.store.real_observations(split="training")):
            self.store.add(row)

    def _fit(self) -> None:
        started = time.perf_counter()
        self.model.fit([row for row in self.store.real_observations(split="training")
                        if row.configuration_signature == self.backend.configuration_signature])
        self.inference_time += time.perf_counter() - started

    def _expired(self) -> bool:
        return time.perf_counter() - self.started >= self.config.max_wall_time

    def _evaluate_bundle(self, ids: list[str], index: int, phase: str) -> bool:
        instance = self.instances[index]
        missing = [pid for pid in dict.fromkeys(ids) if (pid, instance.instance_hash) not in self.known]
        if len(missing) > self.config.total_budget - self.count:
            return False
        for pid in missing:
            if self._expired():
                return False
            self._emit("evaluation_start", program_id=pid, instance_hash=instance.instance_hash, phase=phase)
            started = time.perf_counter()
            # Reserve before dispatch. A backend exception/timeout consumes the
            # same budget as any other attempted real evaluation.
            self.count += 1
            try:
                with terminal_heartbeat(f"BA evaluation {pid[:12]}", enabled=self.config.progress_output):
                    outcome = self.backend.evaluate(self.programs[pid], instance)
            except Exception as exc:
                if getattr(exc, "fatal_evaluation", False):
                    raise
                row = EvaluationObservation(
                    program_id=pid, code_hash=pid, player="solver", generator_id=None,
                    generator_direction=None, instance_hash=instance.instance_hash,
                    instance_features=instance.features, seed=instance.seed, sample_index=instance.sample_index,
                    evaluation_split="training", observed_performance=None, solver_utility=None,
                    feasibility_status="evaluation_error", failure_reason=str(exc), reference_kind=None,
                    reference_value=None, evaluation_time=time.perf_counter()-started, cache_hit=False,
                    observation_source="real_execution", configuration_signature=self.backend.configuration_signature)
                outcome = EvaluationOutcome(row, True, False)
            finally:
                self.evaluation_time += time.perf_counter() - started
            if not outcome.executed and not outcome.cache_hit:
                raise ValueError("evaluation receipt must identify execution or cache reuse")
            if outcome.executed and outcome.cache_hit:
                raise ValueError("an evaluation cannot be both new execution and cache hit")
            self.count -= int(not outcome.executed)
            self.cache_hits += int(outcome.cache_hit)
            self._accept(outcome.observation, pid, instance)
            self._emit("evaluation_done", program_id=pid, instance_hash=instance.instance_hash,
                       phase=phase, status=outcome.observation.feasibility_status,
                       cache_hit=outcome.cache_hit)
        return True

    def _cost(self, ids: list[str], index: int) -> float:
        total = 0.
        for pid in ids:
            if (pid, self.instances[index].instance_hash) in self.known:
                continue
            times = [row.evaluation_time for (key, _), row in self.known.items()
                     if key == pid and row.evaluation_time is not None and row.evaluation_time > 0]
            total += (float(np.mean(times)) if times and self.config.cost_mode == "observed"
                      else self.config.default_evaluation_cost)
        return max(total, 1e-9)

    def _evidence(self, pid: str) -> dict:
        return selection_evidence(self.model, pid, [x.model_input() for x in self.instances], self.weights)

    def _failure_followups(self) -> None:
        """Supplement early failures on DISTINCT instances within existing budget.

        This does not retry cached failures, reserve a larger budget, or retain
        candidates across stages. An available control is paired on the same
        instance; each missing control also spends one real evaluation.
        """
        while not self._expired() and self.count < self.config.total_budget:
            evidence = {pid: self._evidence(pid) for pid in self.programs}
            pending = [pid for pid, row in evidence.items()
                       if rejection_reason(row, self.cpm_config) == "insufficient_reliability_observations"]
            actions = []
            for pid in pending:
                controls = [p for p, row in evidence.items()
                            if p != pid and rejection_reason(row, self.cpm_config) is None]
                control = min(controls, key=lambda p: (-evidence[p]["successful_observations"], p)) if controls else None
                for j, instance in enumerate(self.instances):
                    if (pid, instance.instance_hash) in self.known or self.weights[j] <= 0:
                        continue
                    ids = [pid] if control is None else [control, pid]
                    missing = sum((p, instance.instance_hash) not in self.known for p in ids)
                    if missing <= self.config.total_budget - self.count:
                        actions.append((evidence[pid]["distinct_instances"], missing, j, pid, ids))
            if not actions:
                return
            _, _, index, pid, ids = min(actions)
            self._emit("failure_followup", candidate=pid,
                       instance_hash=self.instances[index].instance_hash,
                       controls=[p for p in ids if p != pid])
            if not self._evaluate_bundle(ids, index, "failure_followup"):
                return
            self._fit()

    def _actions(self, selected: list[str], bank) -> list[dict]:
        actions = []
        for pid in bank.program_ids:
            if rejection_reason(self._evidence(pid), self.cpm_config) is not None:
                continue
            controls = [p for p in selected if p != pid]
            if not controls:
                others = [p for p in bank.program_ids if p != pid and
                          rejection_reason(self._evidence(p), self.cpm_config) is None]
                if others:
                    controls = [min(others, key=lambda p: (bank.cpi_samples([p], self.weights).mean(), p))]
            h = bank.losses[:, bank.program_ids.index(pid), :]
            if controls:
                incumbent = bank.losses[:, [bank.program_ids.index(p) for p in controls], :].min(axis=1)
                local_gain = np.maximum(incumbent - h, 0.)
            else:
                local_gain = -h  # Singleton-only evaluation fallback.
            total_gain = local_gain @ self.weights
            ucb = float(total_gain.mean() + self.config.kappa * total_gain.std(ddof=1))
            for j, instance in enumerate(self.instances):
                ids = list(dict.fromkeys([*controls, pid]))
                missing = [p for p in ids if (p, instance.instance_hash) not in self.known]
                if not missing or len(missing) > self.config.total_budget - self.count:
                    continue
                mean, std = float(local_gain[:, j].mean()), float(local_gain[:, j].std(ddof=1))
                # Decision-value proxy: portfolio marginal UCB multiplied by
                # local complementarity uncertainty/impact, NOT raw cell var.
                influence = self.weights[j] * (abs(mean) + self.config.kappa * std)
                cost = self._cost(ids, j)
                actions.append({"candidate": pid, "controls": controls, "ids": ids, "index": j,
                                "missing": missing, "program_ucb": ucb,
                                "local_gain_mean": mean, "local_gain_std": std,
                                "expected_cost": cost, "score": max(ucb, 0.) * influence / cost})
        return actions

    def run(self, candidates: Sequence[Any], instances: Sequence[ActiveInstance], population_size: int,
            weights: Sequence[float] | None = None) -> tuple[list[Any], dict]:
        if self._used:
            raise RuntimeError("controllers are single-use; create a fresh one for a new budget")
        self._used = True
        self.started = time.perf_counter()
        self.count = self.cache_hits = self.pair_count = self.exploration_count = 0
        self.inference_time = self.evaluation_time = self.acquisition_time = 0.
        self.known: dict[tuple[str, str], EvaluationObservation] = {}
        unique = {_key(p): p for p in candidates}
        self.programs = {pid: p for pid, p in unique.items() if not _exclusion(p, set())}
        if not self.programs or not instances or type(population_size) is not int or population_size < 1:
            raise ValueError("need eligible pure candidates, training instances and positive population size")
        if len({x.instance_hash for x in instances}) != len(instances):
            raise ValueError("deduplicate instance contents before assigning target weights")
        self.instances = list(instances)
        for instance in instances:
            instance.validate_contents()
        if any(x.evaluation_split != "training" for x in instances):
            raise ValueError("active pool cannot contain validation/test instances")
        if any(row.evaluation_split != "training" and row.instance_hash in {x.instance_hash for x in instances}
               for row in self.store.real_observations()):
            raise ValueError("active training pool overlaps a validation/test observation")
        self.weights = target_weights(weights, len(instances))
        rng = np.random.default_rng(self.config.random_seed)
        self._emit("start", method=self.config.method, candidate_count=len(self.programs))
        # Restore verified real caches, never posterior labels. Reading them
        # changes no run budget and at most one effective observation per cell.
        historical = {(r.code_hash, r.instance_hash): r for r in self.store.real_observations(split="training")
                      if r.configuration_signature == self.backend.configuration_signature}
        for pid in self.programs:
            for instance in instances:
                row = historical.get((pid, instance.instance_hash))
                if row is None:
                    row = self.backend.peek(self.programs[pid], instance)
                    self.cache_hits += int(row is not None)
                if row is not None:
                    self._accept(row, pid, instance)
        reason = "budget_exhausted"
        # Shared, evenly spaced columns ensure every new candidate obtains a
        # small real paired start. All initial runs consume the SAME budget.
        initial_columns = np.linspace(0, len(instances)-1,
                                      min(len(instances), self.config.initial_per_program), dtype=int)
        initial_complete = True
        for j in initial_columns:
            if not self._evaluate_bundle(list(self.programs), int(j), "initial_uniform"):
                reason = "wall_time_limit" if self._expired() else "initial_bundle_does_not_fit"
                initial_complete = False
                break
        self._fit()
        if initial_complete:
            self._failure_followups()
        while initial_complete and self.count < self.config.total_budget and not self._expired():
            fit_start = time.perf_counter()
            evidence = {pid: self._evidence(pid) for pid in self.programs}
            usable = [pid for pid in self.programs if rejection_reason(evidence[pid], self.cpm_config) is None]
            if not usable:
                reason = "no_eligible_candidates"
                break
            selected, _ = bayesian_cpm_select([self.programs[p] for p in usable], self.model,
                [x.model_input() for x in instances], population_size, self.weights, config=self.cpm_config)
            bank = self.model.joint_posterior_predictive(usable, [x.model_input() for x in instances],
                self.cpm_config.n_draws, random_seed=self.config.random_seed + self.pair_count,
                include_noise=self.cpm_config.include_noise)
            bank = penalize_failures(bank, evidence, self.cpm_config.failure_margin)
            self.inference_time += time.perf_counter() - fit_start
            selected_ids = [_key(p) for p in selected]
            made_progress = False
            for _ in range(self.config.batch_size):
                started = time.perf_counter()
                actions = self._actions(selected_ids, bank)
                if not actions:
                    reason = "no_complete_pair_fits_remaining_budget"
                    self.acquisition_time += time.perf_counter() - started
                    break
                explore = (self.config.method == "ucb" and
                           self.exploration_count < math.ceil(self.config.min_exploration_fraction * (self.pair_count+1)))
                if self.config.method == "random":
                    action = actions[int(rng.integers(len(actions)))]
                elif self.config.method == "uniform" or explore:
                    # Balance candidate exposure first, then common instances.
                    action = min(actions, key=lambda a: (
                        sum(p == a["candidate"] for p, x in self.known),
                        sum(x == instances[a["index"]].instance_hash for p, x in self.known),
                        a["index"], a["candidate"]))
                else:
                    action = max(actions, key=lambda a: (a["score"], a["candidate"], -a["index"]))
                    if action["score"] <= self.config.information_threshold:
                        reason = "information_value_threshold"
                        self.acquisition_time += time.perf_counter() - started
                        break
                self.acquisition_time += time.perf_counter() - started
                self._emit("paired_action", **action, exploration=explore, selected_keys=selected_ids)
                if not self._evaluate_bundle(action["ids"], action["index"], "paired_active"):
                    reason = "wall_time_limit" if self._expired() else "budget_exhausted"
                    break
                self.pair_count += 1
                self.exploration_count += int(explore)
                made_progress = True
                self._emit("paired_complete", candidate=action["candidate"], controls=action["controls"],
                           instance_hash=instances[action["index"]].instance_hash,
                           all_members_observed=all((p, instances[action["index"]].instance_hash) in self.known
                                                    for p in action["ids"]))
            self._fit()
            self._failure_followups()
            if not made_progress:
                break
        if self._expired():
            reason = "wall_time_limit"
        elif self.count == self.config.total_budget:
            reason = "budget_exhausted"
        started = time.perf_counter()
        selected, cpm_log = bayesian_cpm_select(list(self.programs.values()), self.model,
            [x.model_input() for x in instances], population_size, self.weights, config=self.cpm_config)
        self.inference_time += time.perf_counter() - started
        self._emit("finished", status=reason, selected_keys=[_key(p) for p in selected])
        return selected, {"method": self.config.method, "stop_reason": reason,
            "actual_evaluation_count": self.count, "cache_hit_count": self.cache_hits,
            "effective_observation_count": self.model.observation_count,
            "remaining_budget": self.config.total_budget-self.count, "config": asdict(self.config),
            "total_wall_time": time.perf_counter()-self.started, "evaluation_time": self.evaluation_time,
            "bayesian_inference_time": self.inference_time, "acquisition_time": self.acquisition_time,
            "paired_action_count": self.pair_count, "exploration_action_count": self.exploration_count,
            "posterior_cpi": cpm_log["expected_cpi"], "cpm": cpm_log,
            "selected_keys": [_key(p) for p in selected], "events": self.events,
            "wall_limit_semantics": "admission_limit_existing_call_keeps_original_timeout"}
