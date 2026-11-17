"""Opt-in Solver Oracle. All game payoffs still come from SNDEvaluator.pair.

This module owns a private evaluator during response search. Snapshot caches
may be reused, but Generator threads cannot spend this Oracle's budget.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import math
import time
from typing import Any, NoReturn, TYPE_CHECKING
from ba_eoh_s.eoh_s import cpm_select, cpi, finite_failure_vectors
from m_asro.problems.snd.evaluation import SNDEvaluator, digest
from m_asro.experiments.storage import Program
from ba_eoh_s.bayesian.active_evaluation import (ActiveEvaluationConfig, ActiveInstance, BayesianActiveEvaluator,
                                EvaluationOutcome, SNDEvaluationBackend)
from ba_eoh_s.bayesian.bayesian_cpm import BayesianCPMConfig, bayesian_cpm_select
from ba_eoh_s.bayesian.linear_model import BayesianPerformanceModel
from ba_eoh_s.bayesian.observations import EvaluationObservation
from ba_eoh_s.bayesian.failure_policy import failure_kind, rejection_reason, selection_evidence

if TYPE_CHECKING:
    from m_asro.core.solver import MultiAgentSNDSolver


@dataclass(frozen=True)
class BayesianOracleConfig:
    cpm_method: str = "original"
    active_evaluation: bool = False
    evaluation_budget: int = 256  # Per outer response, ALL stages + real audits.
    initial_per_program: int = 2
    acquisition_method: str = "ucb"
    batch_size: int = 1
    min_exploration_fraction: float = .2
    kappa: float = 1.
    max_wall_time: float = 600.
    information_threshold: float = 0.
    cost_mode: str = "fixed"  # Fixed is reproducible for exact prompt replay.
    default_evaluation_cost: float = 1.
    n_draws: int = 256
    min_observations: int = 2
    epsilon: float = .01
    observation_noise: float = .0025
    credible_level: float = .95
    include_noise: bool = True
    failure_rate_min_instances: int = 10
    max_failure_rate: float = .05

    @property
    def enabled(self) -> bool:
        return self.cpm_method != "original" or self.active_evaluation

    def __post_init__(self) -> None:
        if self.cpm_method not in {"original", "bayesian"}:
            raise ValueError("cpm_method must be original or bayesian")
        if type(self.active_evaluation) is not bool:
            raise ValueError("active_evaluation must be bool")
        if self.cost_mode not in {"fixed", "observed"}:
            raise ValueError("cost_mode must be fixed or observed")
        if not math.isfinite(self.observation_noise) or self.observation_noise <= 0:
            raise ValueError("observation_noise must be finite and positive")
        self.active_config(0, self.evaluation_budget, False)
        self.cpm_config(0)
        if self.initial_per_program < self.min_observations:
            raise ValueError("initial_per_program must cover min_observations")

    def active_config(self, seed: int, budget: int, progress: bool) -> ActiveEvaluationConfig:
        return ActiveEvaluationConfig(method=self.acquisition_method, total_budget=budget,
            initial_per_program=self.initial_per_program, batch_size=self.batch_size,
            min_exploration_fraction=self.min_exploration_fraction, kappa=self.kappa,
            max_wall_time=self.max_wall_time, information_threshold=self.information_threshold,
            default_evaluation_cost=self.default_evaluation_cost, random_seed=seed, progress_output=progress,
            cost_mode=self.cost_mode)

    def cpm_config(self, seed: int) -> BayesianCPMConfig:
        return BayesianCPMConfig(n_draws=self.n_draws, min_observations=self.min_observations,
            epsilon=self.epsilon, random_seed=seed, credible_level=self.credible_level,
            include_noise=self.include_noise, failure_rate_min_instances=self.failure_rate_min_instances,
            max_failure_rate=self.max_failure_rate)


class OracleBudgetError(RuntimeError):
    fatal_evaluation = True


class _LedgerBackend:
    def __init__(self, session: SolverOracleSession, evaluator: SNDEvaluator) -> None:
        self.session = session
        self.adapter = SNDEvaluationBackend(evaluator)
        self.configuration_signature = self.adapter.configuration_signature

    def peek(self, program: Program, instance: ActiveInstance) -> EvaluationObservation | None:
        # Only actual matching training evidence may restore a failed attempt
        # that raised before the evaluator populated its own cache.
        key = (program.code_hash, instance.instance_hash)
        row = self.session.records.get(key)
        if row is not None:
            return replace(row, cache_hit=True)
        row = self.adapter.peek(program, instance)
        if row is not None:
            self.session.restored_cache_count += 1
            return self.session.accept(row, instance)
        return None

    def evaluate(self, program: Program, instance: ActiveInstance) -> EvaluationOutcome:
        cached = self.peek(program, instance)
        if cached is not None:
            self.session.cache_hits += 1
            return EvaluationOutcome(cached, False, True)
        if self.session.actual >= self.session.settings.evaluation_budget:
            self.session.fail("Solver Oracle real evaluation budget exhausted")
        if time.perf_counter()-self.session.started >= self.session.settings.max_wall_time:
            self.session.fail("Solver Oracle wall admission limit reached")
        # Reserve before dispatch, including reference/candidate failures.
        self.session.actual += 1
        self.session.counts[self.session.phase] += 1
        started = time.perf_counter()
        outcome = self.adapter.evaluate(program, instance)
        self.session.evaluation_seconds += time.perf_counter()-started
        return replace(outcome, observation=self.session.accept(outcome.observation, instance))


class SolverOracleSession:
    """One shared budget across all EoH-S selections within an outer response."""
    def __init__(self, solver: MultiAgentSNDSolver, iteration: int, scenarios: list[dict],
                 weights: list[float] | None, labels: list[dict]) -> None:
        self.solver, self.iteration = solver, iteration
        self.settings = solver.bayesian_config
        self.started = time.perf_counter()
        self.actual = self.cache_hits = 0
        self.restored_cache_count = 0
        self.evaluation_seconds = self.inference_seconds = self.acquisition_seconds = 0.
        self.counts = {"selection": 0, "audit": 0}
        self.phase = "selection"
        self.records, self.selections = {}, []
        self.labels = {row["hash"]: row for row in labels}
        self.pool = []
        for j, x in enumerate(scenarios):
            label = self.labels[digest(x)]
            generated = solver.evaluator.generated.get((
                solver.evaluator.program_index[label["member_id"]].code_hash, label["source"], label["sample"])
                ) if label.get("member_id") else None
            source_seed = generated["seed"] if generated else x.get("seed")
            source_seed = source_seed if type(source_seed) is int else solver.seed
            self.pool.append(ActiveInstance.from_snd(x, seed=source_seed,
                sample_index=label.get("sample", j)))
        forbidden = {digest(x) for x in [*solver.validation, *solver.tests]}
        if forbidden & {x.instance_hash for x in self.pool}:
            raise ValueError("Bayesian Oracle training instances overlap validation/test contents")
        if len(self.pool) < self.settings.min_observations:
            raise ValueError("训练实例数小于 Bayesian min_observations；扩大训练池或显式设 min_observations=1")
        self.weights = weights
        self.store = solver._bayesian_store
        self.model = BayesianPerformanceModel(observation_noise=self.settings.observation_noise,
                                               random_seed=solver.seed+iteration)
        evaluator = solver.evaluator
        # Private evaluator: same safety/reference settings, same archive and
        # TimingRecorder, disjoint mutable caches during parallel responses.
        self.evaluator = SNDEvaluator(solver.base, solver.archive, solver.utility, *solver.reference_args,
            training_bases=solver.training_bases, reference_config=solver.reference_config,
            reference_cache_dir=solver.reference_cache_dir, parallel_workers=solver.parallel_workers,
            timing=solver.timing)
        self.evaluator._process_slots = evaluator._process_slots
        hashes = {x.instance_hash for x in self.pool}
        with evaluator._cache_lock:
            references, evaluations = getattr(solver, "_bayesian_cache_snapshot",
                                               (evaluator.references, evaluator.evaluations))
            self.evaluator.references.update({k: v for k, v in references.items() if k in hashes})
            self.evaluator.evaluations.update({k: v for k, v in evaluations.items() if k[1] in hashes})
        self.backend = _LedgerBackend(self, self.evaluator)
        self.cpm_config = replace(self.settings.cpm_config(solver.seed+iteration),
                                  failure_margin=solver.cpi_failure_margin)

    def accept(self, row: EvaluationObservation, instance: ActiveInstance) -> EvaluationObservation:
        label = self.labels[instance.instance_hash]
        metrics_row = self.evaluator.evaluations.get((row.code_hash, row.instance_hash))
        utility = (self.solver.utility.pair({**metrics_row, "distortion": 0.}, self.solver.directions[0])[0]
                   if metrics_row is not None else None)
        row = replace(row, generator_id=label.get("member_id", label.get("generator_id")),
                      generator_direction=label.get("source") if label.get("source") != "base" else None,
                      solver_utility=utility)
        key = (row.code_hash, row.instance_hash)
        if key not in self.records:
            self.records[key] = row
            previous = [r for r in self.store.real_observations(split="training")
                        if (r.code_hash, r.instance_hash, r.configuration_signature) ==
                        (row.code_hash, row.instance_hash, row.configuration_signature)]
            if previous and (previous[0].observed_performance != row.observed_performance
                             or previous[0].feasibility_status != row.feasibility_status):
                raise ValueError("conflicting real observations for deterministic Oracle cell")
            if not previous:
                self.store.add(row)
                self.solver.archive.append("bayesian_observations.jsonl", row.to_dict())
        return row

    def _fit(self) -> None:
        started = time.perf_counter()
        self.model.fit([r for r in self.store.real_observations(split="training")
                        if r.configuration_signature == self.backend.configuration_signature])
        self.inference_seconds += time.perf_counter()-started

    def select(self, candidates: list[Program], stage: int) -> tuple[list[Program], dict[str, list[float]]]:
        # Temporary training parents are separate from the warm-start-free
        # Bayesian portfolio. Do not retire on source count before real audit.
        candidates = list({p.code_hash: p for p in candidates
                           if p.error is None and p.member_ids is None}.values())
        eligible = self.solver._without_warm_start(candidates)
        warm = [p for p in candidates if self.solver._is_warm_start(p)]
        if getattr(self.solver, "_warm_start_retired", False):
            warm = []
        # Keep every configured warm-start control. They occupy temporary
        # training slots only; none is eligible for Bayesian CPM or response.
        if not eligible:
            if warm:
                return self._training_population([], {}, {}, None, warm, stage)
            self.fail("没有合法非暖启动 Bayesian 候选，且无可暂留的暖启动父代")
        # Divide remaining budget over remaining EoH stages, reserving a worst
        # case full selected-portfolio audit BEFORE active evaluation starts.
        remaining = self.settings.evaluation_budget-self.actual
        stages_left = self.solver.R-stage+1
        stage_allowance = remaining // stages_left
        reserve = min(self.solver.K, len(eligible))*len(self.pool)
        reserve += sum(self.backend.peek(p, x) is None for p in warm for x in self.pool)
        self.phase = "selection"
        active_log = None
        if self.settings.active_evaluation:
            budget = max(0, stage_allowance-reserve)
            config = self.settings.active_config(self.solver.seed+self.iteration*1000+stage, budget,
                                                  self.solver.progress)
            # Global admission time includes generation pauses between stages.
            config = replace(config, max_wall_time=max(.000001,
                self.settings.max_wall_time-(time.perf_counter()-self.started)))
            active = BayesianActiveEvaluator(self.model, self.backend, config=config,
                cpm_config=self.cpm_config, store=self.store,
                event_callback=lambda row: self.solver.archive.append("bayesian_active_events.jsonl",
                    {**row, "iteration": self.iteration, "eoh_iteration": stage}))
            selected, active_log = active.run(eligible, self.pool, self.solver.K, self.weights)
            self.inference_seconds += active_log["bayesian_inference_time"]
            self.acquisition_seconds += active_log["acquisition_time"]
        else:
            for p in eligible:
                for instance in self.pool:
                    self.backend.evaluate(p, instance)
            self._fit()
            selected = []
        if self.settings.cpm_method == "bayesian":
            started = time.perf_counter()
            selected, diagnostic = bayesian_cpm_select(eligible, self.model,
                [x.model_input() for x in self.pool], self.solver.K, self.weights, config=self.cpm_config)
            self.inference_seconds += time.perf_counter()-started
        else:
            # Active-only ablation: ORIGINAL CPM on genuinely paired common
            # columns, never a dense matrix filled with posterior means.
            evidence = [p for p in eligible if rejection_reason(selection_evidence(self.model, p,
                [x.model_input() for x in self.pool], self.weights), self.cpm_config) is None]
            common = [j for j, x in enumerate(self.pool) if all(
                self.backend.peek(p, x) is not None for p in evidence)]
            rows = {p.code_hash: [self.backend.peek(p, self.pool[j]) for j in common] for p in evidence}
            vectors = finite_failure_vectors(
                {key: [r.observed_performance if r.feasible else float("inf") for r in values]
                 for key, values in rows.items()}, self.solver.cpi_failure_margin,
                {key: [failure_kind(r.failure_reason) if not r.feasible else None for r in values]
                 for key, values in rows.items()}) if common and evidence else {}
            selected, diagnostic = cpm_select(evidence, vectors, self.solver.K,
                [self.weights[j] for j in common] if self.weights is not None else None) if common and evidence else ([], {})
            diagnostic["observed_common_columns"] = common
        if not selected:
            if warm:
                return self._training_population([], {}, diagnostic, active_log, warm, stage)
            self.fail("真实初始观测不足或所有候选失败；不允许用暖启动/预测替代响应，请增加 Oracle 预算")
        # Remember the selected set before auditing. Interrupted export will
        # admit only members with complete REAL audit rows, never predictions.
        self.solver._remember_portfolio(selected, [x.data for x in self.pool],
                                       self.weights, self.iteration, stage)
        # The original CS/LS rules receive complete audited vectors, with
        # explicitly labelled penalties for failed cells. Payoff rows stay raw.
        self.phase = "audit"
        missing = [(p, x) for p in selected for x in self.pool if self.backend.peek(p, x) is None]
        if len(missing) > self.settings.evaluation_budget-self.actual:
            self.fail("预算不足以完成真实 portfolio / payoff audit")
        for p, x in missing:
            self.backend.evaluate(p, x)
        # A newly discovered failure changes reliability evidence, not a
        # successful loss observation. Apply the SAME rate gate after audit.
        audited = list(selected)
        self._fit()
        audit_evidence = {p.code_hash: selection_evidence(self.model, p,
                          [x.model_input() for x in self.pool], self.weights) for p in selected}
        selected = [p for p in selected if rejection_reason(audit_evidence[p.code_hash], self.cpm_config) is None]
        if not selected and not warm:
            self.fail("候选在真实 audit 中均失败，禁止回退暖启动")
        rows = {p.code_hash: [self.backend.peek(p, x) for x in self.pool] for p in selected}
        vectors = finite_failure_vectors(
            {key: [r.observed_performance if r.feasible else float("inf") for r in values]
             for key, values in rows.items()}, self.solver.cpi_failure_margin,
            {key: [failure_kind(r.failure_reason) if not r.feasible else None for r in values]
             for key, values in rows.items()})
        diagnostic["audit_excluded_ids"] = [p.id for p in audited if p.code_hash not in vectors]
        diagnostic["audit_failure_evidence"] = audit_evidence
        diagnostic["audit_exclusion_reasons"] = {p.id: rejection_reason(audit_evidence[p.code_hash], self.cpm_config)
                                                 for p in audited if p not in selected}
        return self._training_population(selected, vectors, diagnostic, active_log, warm, stage)

    def _training_population(self, selected: list[Program], vectors: dict[str, list[float]],
                             diagnostic: dict, active_log: dict | None, warm: list[Program],
                             stage: int) -> tuple[list[Program], dict[str, list[float]]]:
        """Keep a temporary real control without admitting it to a response.

        Only fully audited non-baseline members count towards retirement.
        Failed control cells receive the original finite parent-search penalty;
        these penalties are never Bayesian observations or portfolio payoffs.
        """
        self.solver._training_candidates(selected, "solver")
        if getattr(self.solver, "_warm_start_retired", False):
            warm = []
        # Reserve temporary local population slots for all configured controls
        # that fit after the genuinely selected portfolio.
        warm = warm[:max(0, self.solver.K-len(selected))]
        training = selected + warm
        training_vectors = dict(vectors)
        self.phase = "audit"
        missing = [(p, x) for p in warm for x in self.pool if self.backend.peek(p, x) is None]
        if len(missing) > self.settings.evaluation_budget-self.actual:
            self.fail("预算不足以完成暖启动训练对照评估")
        for p in warm:
            observations = [self.backend.evaluate(p, x).observation for x in self.pool]
            training_vectors[p.code_hash] = [r.observed_performance if r.feasible else float("inf")
                                             for r in observations]
        parent_vectors = finite_failure_vectors(training_vectors, self.solver.cpi_failure_margin)
        empirical = cpi(vectors, list(vectors), self.weights) if selected else None
        has_failure = any(not self.backend.peek(p, x).feasible for p in selected for x in self.pool)
        record = {"iteration": self.iteration, "eoh_iteration": stage,
            "cpm_method": self.settings.cpm_method, "selection": diagnostic,
            "active": active_log, "portfolio_cpi": empirical,
            "portfolio_cpi_source": ("real_training_audit_with_failure_penalty" if has_failure
                                     else "real_complete_training_audit"), "selected_ids": [p.id for p in selected],
            "audit_excluded_ids": diagnostic.get("audit_excluded_ids", []),
            "training_failure_count": {p.id: sum(not self.backend.peek(p, x).feasible for x in self.pool)
                                       for p in selected},
            "failure_penalties": "selection_only_not_successful_observations",
            "training_parent_ids": [p.id for p in training],
            "temporary_warm_start_ids": [p.id for p in warm],
            "warm_start_retired": getattr(self.solver, "_warm_start_retired", False),
            "qualified_non_warm_count": len(selected),
            "training_population_cpi": cpi(parent_vectors, list(parent_vectors), self.weights),
            "training_population_cpi_source": "real_training_with_finite_control_failure_penalty",
            "warm_start_control_cpi": {p.id: cpi(parent_vectors, [p.code_hash], self.weights) for p in warm},
            "warm_start_control_failure_count": {p.id: sum(
                not self.backend.peek(p, x).feasible for x in self.pool) for p in warm},
            "status": "selected" if selected else "temporary_warm_start_training_only",
            "actual_evaluation_count": self.actual, "budget": self.settings.evaluation_budget}
        self.selections.append(record)
        self.solver.archive.append("bayesian_selections.jsonl", record)
        portfolio_text = f"{empirical:.6f}" if empirical is not None else "暂无合格新程序"
        self.solver._report(f"  solver: BA EoH-S {stage}/{self.solver.R}，实测 portfolio CPI={portfolio_text}，"
                            f"真实评估 {self.actual}/{self.settings.evaluation_budget}")
        if warm:
            self.solver._report("  solver：暖启动仅暂留为训练父代/对照，继续生成新程序；禁止进入响应与最终输出")
        return training, parent_vectors

    def fail(self, message: str) -> NoReturn:
        self.solver.archive.append("bayesian_failures.jsonl", {"iteration": self.iteration,
            "actual_evaluation_count": self.actual, "budget": self.settings.evaluation_budget, "error": message})
        self.solver._write_selection_status(error=message,
            candidates=[p for p in self.solver.archive.programs if p.player == "solver"])
        self.solver._report(f"  solver: Oracle 停止，真实评估 {self.actual}/{self.settings.evaluation_budget}；{message}")
        raise OracleBudgetError(message)

    def publish(self) -> None:
        # Publish only actual cache rows AFTER the solver search. Main evaluator
        # single-flight locks remain authoritative during simultaneous searches.
        with self.solver.evaluator._cache_lock:
            for key, row in self.evaluator.evaluations.items():
                self.solver.evaluator.evaluations.setdefault(key, row)
            for key, row in self.evaluator.references.items():
                self.solver.evaluator.references.setdefault(key, row)

    def finish(self) -> dict[str, Any]:
        self.publish()
        return {"iteration": self.iteration, "config": asdict(self.settings),
            "actual_evaluation_count": self.actual, "cache_hit_count": self.cache_hits,
            "restored_cache_observation_count": self.restored_cache_count,
            "evaluation_count_by_phase": self.counts, "total_wall_time": time.perf_counter()-self.started,
            "evaluation_time": self.evaluation_seconds, "bayesian_inference_time": self.inference_seconds,
            "acquisition_time": self.acquisition_seconds, "selections": self.selections,
            "budget_scope": "solver_oracle_all_selection_and_complete_member_audits",
            "outside_budget": "generator_search_payoff_expansion_validation_holdout_counted_by_global_timing",
            "wall_limit_semantics": "admission_limit_existing_calls_keep_original_timeouts"}
