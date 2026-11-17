"""Standalone Bayesian complementary population management; no game integration."""
from __future__ import annotations

import ast
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
import math
import time
from typing import Any, Protocol

from m_asro.problems.snd.baselines import BASE_SOLVER
from ba_eoh_s.eoh_s import cpm_select
from ba_eoh_s.bayesian.posterior import JointLossSamples, posterior_summary, target_weights
from ba_eoh_s.bayesian.failure_policy import penalize_failures, rejection_reason, selection_evidence


class PerformancePosterior(Protocol):
    training_split: str

    def selection_evidence(self, program: Any, *,
                           instance_weights: Mapping[str, float] | None = None) -> dict[str, Any]: ...

    def joint_posterior_predictive(
        self, programs: Iterable[Any], instances: Iterable[Any], n_draws: int = 256,
        *, random_seed: int | None = None, include_noise: bool = True,
    ) -> JointLossSamples: ...


@dataclass(frozen=True)
class BayesianCPMConfig:
    n_draws: int = 2048
    credible_level: float = 0.95
    epsilon: float = 0.0
    min_observations: int = 2
    random_seed: int = 42
    include_noise: bool = True
    failure_rate_min_instances: int = 10
    max_failure_rate: float = 0.05
    failure_margin: float = 1.0

    def __post_init__(self) -> None:
        if type(self.n_draws) is not int or self.n_draws < 2:
            raise ValueError("n_draws must be an integer >= 2")
        if type(self.min_observations) is not int or self.min_observations < 1:
            raise ValueError("min_observations must be a positive integer")
        if not math.isfinite(self.credible_level) or not 0 < self.credible_level < 1:
            raise ValueError("credible_level must be in (0, 1)")
        if not math.isfinite(self.epsilon) or self.epsilon < 0:
            raise ValueError("epsilon must be nonnegative and finite")
        if type(self.random_seed) is not int or self.random_seed < 0:
            raise ValueError("random_seed must be a nonnegative integer")
        if type(self.include_noise) is not bool:
            raise ValueError("include_noise must be bool")
        if type(self.failure_rate_min_instances) is not int or self.failure_rate_min_instances < 1:
            raise ValueError("failure_rate_min_instances must be a positive integer")
        if not math.isfinite(self.max_failure_rate) or not 0 <= self.max_failure_rate <= 1:
            raise ValueError("max_failure_rate must lie in [0, 1]")
        if not math.isfinite(self.failure_margin) or self.failure_margin <= 0:
            raise ValueError("failure_margin must be positive and finite")


def _field(program: Any, name: str, default: Any = None) -> Any:
    return program.get(name, default) if isinstance(program, Mapping) else getattr(program, name, default)


def _key(program: Any) -> str:
    return program if isinstance(program, str) else str(_field(program, "code_hash", _field(program, "id", "")))


_WARM_START_AST = ast.dump(ast.parse(BASE_SOLVER), include_attributes=False)


def _exclusion(program: Any, excluded_ids: set[str]) -> str | None:
    """Conservative pure-program guard; never export or execute candidates here.

    Synthetic string ids have no source to check: callers must supply excluded
    ids for them. Real Program sources are checked against the current baseline
    AST; ensembles/mixed policies are outside this pure-candidate interface.
    Existing solver guards remain authoritative and entirely unchanged.
    """
    if _key(program) in excluded_ids or str(_field(program, "id", "")) in excluded_ids:
        return "explicitly_excluded"
    if _field(program, "player", "solver") != "solver":
        return "not_solver"
    if _field(program, "member_ids") is not None or _field(program, "mixture_weights") is not None:
        return "requires_pure_candidate"
    if _field(program, "operator") == "base":
        return "warm_start"
    if _field(program, "status", "parsed") in {"rejected", "failed", "invalid"} or _field(program, "error"):
        return "rejected_program"
    code = _field(program, "code")
    if code:
        try:
            if ast.dump(ast.parse(code), include_attributes=False) == _WARM_START_AST:
                return "warm_start_copy"
        except SyntaxError:
            return "invalid_source"
    return None


def bayesian_cpm_select(
    candidates: Sequence[Any], model: PerformancePosterior, instances: Iterable[Any],
    population_size: int, weights: Sequence[float] | None = None, *,
    config: BayesianCPMConfig | None = None, excluded_program_ids: Iterable[str] = (),
) -> tuple[list[Any], dict[str, Any]]:
    """Greedy posterior-expected complementarity using one joint loss bank.

    The first member minimises expected singleton CPI; every subsequent member
    maximises E[C(H)-C(H+{h})|D]. Probability(gain > epsilon) is a diagnostic,
    not an extra selection threshold or an empirical mixed-strategy utility.
    Failures require follow-up evidence, then a weighted failure-rate gate.
    Known failed cells receive decision penalties, never fabricated losses.
    Unknown future feasibility is NOT certified by this rule.
    """
    started = time.perf_counter()
    settings = config or BayesianCPMConfig()
    if type(population_size) is not int or population_size < 1:
        raise ValueError("population_size must be a positive integer")
    if model.training_split != "training":
        raise ValueError("Bayesian-CPM selection requires a training-only posterior")
    items = list(instances)
    importance = target_weights(weights, len(items))
    for item in items:
        if (isinstance(item, Mapping) and item.get("evaluation_split", "training") != "training"):
            raise ValueError("selection instances must be training representatives")
    unique = {_key(p): p for p in candidates}
    if "" in unique:
        raise ValueError("candidates need stable program ids")
    excluded_ids = set(excluded_program_ids)
    eligible: dict[str, Any] = {}
    rejected: dict[str, Any] = {}
    evidence: dict[str, Any] = {}
    for pid, program in unique.items():
        reason = _exclusion(program, excluded_ids)
        row = selection_evidence(model, program, items, importance)
        evidence[pid] = row
        if reason is None:
            reason = rejection_reason(row, settings)
        if reason is None:
            eligible[pid] = program
        else:
            rejected[pid] = {"reason": reason, "evidence": row}
    diagnostic: dict[str, Any] = {
        "method": "bayesian_cpm", "selection_objective": "posterior_expected_instance_minimum_loss",
        "observation_source": "posterior_prediction", "empirical_cpi": None,
        "selected_program_ids": [], "selected_keys": [], "expected_cpi": None,
        "cpi_uncertainty": None, "cpi_interval": None, "posterior_cpi": None,
        "requested_population_size": population_size, "population_size": 0,
        "eligible_program_ids": list(eligible), "excluded_candidates": rejected,
        "candidate_evidence": evidence, "steps": [], "random_seed": settings.random_seed,
        "n_draws": settings.n_draws, "epsilon": settings.epsilon,
        "min_observations": settings.min_observations, "include_noise": settings.include_noise,
        "failure_rate_min_instances": settings.failure_rate_min_instances,
        "max_failure_rate": settings.max_failure_rate,
        "failure_penalties": "selection_only_not_successful_observations",
        "target_instance_count": len(items), "target_weights": importance.tolist(),
        "target_scope": "fixed_training_representatives_not_full_population_posterior",
        "joint_structure": "independent_program_blocks_shared_beta_across_instances",
        "bayesian_inference_time": 0.0,
    }
    if not eligible:
        diagnostic.update({"status": "no_eligible_candidates", "selection_time": time.perf_counter() - started})
        return [], diagnostic
    # A SINGLE draw bank prevents incoherent redraws when H grows or when a
    # candidate is compared to different incumbents.
    bank = model.joint_posterior_predictive(
        list(eligible), items, settings.n_draws, random_seed=settings.random_seed,
        include_noise=settings.include_noise)
    bank = penalize_failures(bank, evidence, settings.failure_margin)
    if (set(bank.program_ids) != set(eligible) or len(bank.instance_hashes) != len(items)
            or bank.losses.shape[0] != settings.n_draws or bank.evaluation_split != "training"):
        raise ValueError("posterior bank does not match candidates and common instances")
    diagnostic.update({"target_instance_hashes": list(bank.instance_hashes),
                       "model_schema": bank.model_schema, "bayesian_inference_time": bank.inference_seconds,
                       "conditioned_observed_cells": bank.conditioned_observed_cells})
    singleton = {pid: bank.cpi_samples([pid], importance) for pid in eligible}
    selected: list[str] = []
    k = min(population_size, len(eligible))
    while len(selected) < k:
        remaining = [pid for pid in eligible if pid not in selected]
        summaries = {}
        for pid in remaining:
            singleton_stat = posterior_summary(singleton[pid], settings.credible_level)
            gain = (posterior_summary(bank.marginal_gain_samples(pid, selected, importance),
                                      settings.credible_level, settings.epsilon) if selected else None)
            summaries[pid] = {"singleton_cpi": singleton_stat, "marginal_gain": gain}
        if not selected:
            winner = min(remaining, key=lambda pid: (summaries[pid]["singleton_cpi"]["mean"], pid))
        else:
            # Match the original CPM's deterministic secondary/tie rules.
            winner = max(remaining, key=lambda pid: (
                summaries[pid]["marginal_gain"]["mean"], -summaries[pid]["singleton_cpi"]["mean"], pid))
        gain = summaries[winner]["marginal_gain"]
        selected.append(winner)
        portfolio_stat = posterior_summary(bank.cpi_samples(selected, importance), settings.credible_level)
        diagnostic["steps"].append({
            "step": len(selected), "selected_key": winner,
            "criterion": "expected_singleton_cpi" if gain is None else "expected_marginal_gain",
            "marginal_gain_mean": gain["mean"] if gain else None,
            "marginal_gain_interval": gain["credible_interval"] if gain else None,
            "improvement_probability": gain["improvement_probability"] if gain else None,
            "marginal_gain": gain, "posterior_cpi": portfolio_stat, "candidate_scores": summaries})
    final = diagnostic["steps"][-1]["posterior_cpi"]
    diagnostic.update({
        "status": "selected" if k == population_size else "insufficient_eligible_candidates",
        "selected_keys": selected, "selected_program_ids": [str(_field(eligible[pid], "id", pid)) for pid in selected],
        "population_size": k, "expected_cpi": final["mean"], "cpi_uncertainty": final["std"],
        "cpi_interval": final["credible_interval"], "posterior_cpi": final,
        "selection_time": time.perf_counter() - started})
    return [eligible[pid] for pid in selected], diagnostic


def select_portfolio(candidates: Sequence[Any], population_size: int, *,
                     cpm_method: str = "original", vectors: Mapping[str, Sequence[float]] | None = None,
                     weights: Sequence[float] | None = None, model: PerformancePosterior | None = None,
                     instances: Iterable[Any] | None = None, config: BayesianCPMConfig | None = None,
                     excluded_program_ids: Iterable[str] = ()) -> tuple[list[Any], dict[str, Any]]:
    """Offline opt-in dispatcher. The original branch is an unmodified call."""
    if cpm_method == "original":
        if vectors is None:
            raise ValueError("original CPM requires observed performance vectors")
        return cpm_select(candidates, vectors, population_size, weights)
    if cpm_method != "bayesian":
        raise ValueError("cpm_method must be original or bayesian")
    if model is None or instances is None:
        raise ValueError("bayesian CPM requires a model and training representatives")
    return bayesian_cpm_select(candidates, model, instances, population_size, weights,
                               config=config, excluded_program_ids=excluded_program_ids)
