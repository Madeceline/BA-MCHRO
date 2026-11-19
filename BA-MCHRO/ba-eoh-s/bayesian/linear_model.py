"""Small conjugate Bayesian performance model used by the offline BA-MCHRO layer.

The model deliberately has no dependency on the solver or on the evaluator.  A
separate Bayesian linear regression is maintained for every program.  This is
the inexpensive first-stage model: observations are real successful executions
and failures are retained for feasibility accounting, but are not fabricated
as continuous losses.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
import hashlib
import json
import math
import time
from typing import Any

import numpy as np

from ba_eoh_s.bayesian.features import FEATURE_NAMES
from ba_eoh_s.bayesian.failure_policy import failure_kind
from ba_eoh_s.bayesian.observations import EvaluationObservation
from ba_eoh_s.bayesian.posterior import JointLossSamples


def _program_id(program: Any) -> str:
    if isinstance(program, str):
        return program
    if isinstance(program, Mapping):
        return str(program.get("code_hash", program.get("program_id", program.get("id", ""))))
    return str(getattr(program, "code_hash", getattr(program, "id", "")))


def _as_mapping(row: Any) -> Mapping[str, Any]:
    if isinstance(row, EvaluationObservation):
        return row.to_dict()
    if isinstance(row, Mapping):
        return row
    raise TypeError("observations must contain EvaluationObservation or mappings")


def _instance_features(instance: Any, feature_names: Sequence[str]) -> np.ndarray:
    if isinstance(instance, np.ndarray):
        vector = np.asarray(instance, dtype=float)
        if vector.shape != (len(feature_names),) or not np.all(np.isfinite(vector)):
            raise ValueError("instance feature array has the wrong shape or contains non-finite values")
        return vector
    if isinstance(instance, (list, tuple)) and len(instance) == len(feature_names):
        vector = np.asarray(instance, dtype=float)
        if np.all(np.isfinite(vector)):
            return vector
    if isinstance(instance, Mapping) and "instance_features" in instance:
        values = instance["instance_features"]
    elif isinstance(instance, Mapping) and all(name in instance for name in feature_names):
        values = instance
    elif isinstance(instance, Mapping):
        from ba_eoh_s.bayesian.features import instance_features
        values = instance_features(instance)
    else:
        # Keep the import lazy so importing the model does not require a data
        # generator when callers provide already extracted feature vectors.
        from ba_eoh_s.bayesian.features import instance_features

        values = instance_features(instance)
    try:
        vector = np.asarray([float(values[name]) for name in feature_names], dtype=float)
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("instance features do not match the model schema") from exc
    if vector.ndim != 1 or not np.all(np.isfinite(vector)):
        raise ValueError("instance features must be finite")
    return vector


class BayesianPerformanceModel:
    """Independent conjugate Bayesian linear regressions for program losses.

    For a program ``h`` and standardized instance features ``phi(x)`` the
    model is ``y = [1, phi(x)] beta_h + eps`` with Gaussian noise of known
    variance.  A Gaussian prior gives the closed-form update

    ``Lambda_n = Lambda_0 + X'X/sigma2`` and
    ``mu_n = Lambda_n^-1 (Lambda_0 mu_0 + X'y/sigma2)``.

    The model is intentionally independent of CPM and M-PSRO.  Posterior
    predictions are evidence for offline selection only; they never become
    empirical payoff entries automatically.
    """

    MODEL_SCHEMA = "bayesian-linear-performance-v1"

    def __init__(
        self,
        feature_names: Sequence[str] = FEATURE_NAMES,
        prior_mean: Sequence[float] | None = None,
        prior_precision: Sequence[Sequence[float]] | None = None,
        observation_noise: float = 1.0,
        random_seed: int = 42,
        ridge: float = 0.0,
    ) -> None:
        self.feature_names = tuple(feature_names)
        if not self.feature_names:
            raise ValueError("feature_names cannot be empty")
        self.dimension = len(self.feature_names) + 1  # intercept
        self.observation_noise = float(observation_noise)
        self.ridge = float(ridge)
        if not math.isfinite(self.observation_noise) or self.observation_noise <= 0:
            raise ValueError("observation_noise must be positive and finite")
        if not math.isfinite(self.ridge) or self.ridge < 0:
            raise ValueError("ridge must be non-negative and finite")
        self.prior_mean = (
            np.zeros(self.dimension, dtype=float)
            if prior_mean is None
            else np.asarray(prior_mean, dtype=float)
        )
        if self.prior_mean.shape != (self.dimension,) or not np.all(np.isfinite(self.prior_mean)):
            raise ValueError("prior_mean has the wrong shape or contains non-finite values")
        if prior_precision is None:
            self.prior_precision = np.eye(self.dimension, dtype=float)
        else:
            self.prior_precision = np.asarray(prior_precision, dtype=float)
        if self.prior_precision.shape != (self.dimension, self.dimension):
            raise ValueError("prior_precision has the wrong shape")
        if not np.all(np.isfinite(self.prior_precision)):
            raise ValueError("prior_precision contains non-finite values")
        if not np.allclose(self.prior_precision, self.prior_precision.T, atol=1e-12):
            raise ValueError("prior_precision must be symmetric")
        try:
            np.linalg.cholesky(self.prior_precision)
        except np.linalg.LinAlgError as exc:
            raise ValueError("prior_precision must be positive definite") from exc

        self.random_seed = int(random_seed)
        self._rng = np.random.default_rng(self.random_seed)
        self._observations: list[dict[str, Any]] = []
        self._records: dict[tuple[Any, ...], dict[str, Any]] = {}
        self._excluded_test_count = 0
        self._excluded_prediction_count = 0
        self.training_split = "training"
        self._means = np.zeros(len(self.feature_names), dtype=float)
        self._scales = np.ones(len(self.feature_names), dtype=float)
        self._posteriors: dict[str, tuple[np.ndarray, np.ndarray, int]] = {}
        self._feasibility: dict[str, dict[str, int]] = {}
        self._fitted = False

    @property
    def observation_count(self) -> int:
        return len(self._observations)

    @property
    def program_ids(self) -> tuple[str, ...]:
        return tuple(sorted(self._posteriors))

    @property
    def excluded_test_count(self) -> int:
        return self._excluded_test_count

    @property
    def excluded_prediction_count(self) -> int:
        return self._excluded_prediction_count

    @property
    def real_observation_count(self) -> int:
        return len(self._observations)

    @property
    def feature_means(self) -> dict[str, float]:
        return dict(zip(self.feature_names, self._means.tolist()))

    @property
    def feature_scales(self) -> dict[str, float]:
        return dict(zip(self.feature_names, self._scales.tolist()))

    def _eligible(self, row: Mapping[str, Any]) -> bool:
        split = row.get("evaluation_split", "training")
        source = row.get("observation_source", "real_execution")
        if split != self.training_split:
            return False
        if source == "posterior_prediction":
            return False
        value = row.get("observed_performance")
        status = str(row.get("feasibility_status", "feasible"))
        return source == "real_execution" and value is not None and status in {
            "ok", "feasible", "success"
        }

    def _normalise_rows(self, observations: Iterable[Any]) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for item in observations:
            row = dict(_as_mapping(item))
            if not row.get("code_hash"):
                row["code_hash"] = str(row.get("program_id", ""))
            if not row.get("code_hash"):
                raise ValueError("each observation needs code_hash or program_id")
            if self._eligible(row):
                value = float(row["observed_performance"])
                if not math.isfinite(value):
                    raise ValueError("observed_performance must be finite")
                # Copy the feature vector now: callers can safely reuse/mutate
                # their observation objects after an update.
                row["_x"] = _instance_features(row, self.feature_names)
            rows.append(row)
        return rows

    def fit(self, observations: Iterable[Any], *, evaluation_split: str = "training") -> "BayesianPerformanceModel":
        """Fit from one split; validation and test rows never leak by default."""
        if evaluation_split not in {"training", "validation"}:
            raise ValueError("evaluation_split must be training or validation")
        self.training_split = evaluation_split
        self._observations = []
        self._records = {}
        self._feasibility = {}
        self._excluded_test_count = 0
        self._excluded_prediction_count = 0
        self._posteriors = {}
        return self.update(observations)

    def update(self, new_observations: Iterable[Any], *, evaluation_split: str | None = None) -> "BayesianPerformanceModel":
        """Add real observations and recompute closed-form posteriors.

        Recomputing from the compact ledger makes sequential updates numerically
        identical to a one-shot fit and permits the feature scaler to remain
        leakage-free (it is learned from the observations supplied to the
        model, never from test instances).
        """
        if evaluation_split is not None:
            if evaluation_split not in {"training", "validation"}:
                raise ValueError("evaluation_split must be training or validation")
            self.training_split = evaluation_split
        for row in self._normalise_rows(new_observations):
            key = self._row_key(row)
            previous = self._records.get(key)
            if previous is None:
                self._records[key] = row
            elif (previous.get("observation_source") == "real_execution"
                  and row.get("observation_source") == "real_execution"
                  and previous.get("observed_performance") != row.get("observed_performance")):
                raise ValueError(f"conflicting real observations for key={key}")
            elif (previous.get("observation_source") != "real_execution"
                  and row.get("observation_source") == "real_execution"):
                self._records[key] = row
        self._excluded_test_count = sum(
            row.get("evaluation_split", "training") == "test" for row in self._records.values()
        )
        self._excluded_prediction_count = sum(
            row.get("observation_source") == "posterior_prediction" for row in self._records.values()
        )
        self._feasibility = {}
        for row in self._records.values():
            pid = str(row["code_hash"])
            status = str(row.get("feasibility_status", "unknown"))
            bucket = self._feasibility.setdefault(pid, {"feasible": 0, "infeasible": 0})
            if status in {"ok", "feasible", "success"} and row.get("observed_performance") is not None:
                bucket["feasible"] += 1
            else:
                bucket["infeasible"] += 1
        self._observations = [
            row for row in self._records.values()
            if self._eligible(row)
        ]
        self._rebuild()
        return self

    @staticmethod
    def _row_key(row: Mapping[str, Any]) -> tuple[Any, ...]:
        """A deterministic program/instance/config is one effective sample.

        Generator seeds locate instances, not independent noisy reruns.  Keep
        them in the observation ledger for provenance, not in the likelihood.
        """
        return (
            str(row.get("code_hash", row.get("program_id", ""))),
            str(row.get("instance_hash", "")),
            str(row.get("evaluation_split", "training")),
            str(row.get("configuration_signature", "default")),
        )

    def _rebuild(self) -> None:
        if self._observations:
            raw = np.vstack([row["_x"] for row in self._observations])
            self._means = raw.mean(axis=0)
            self._scales = raw.std(axis=0)
            self._scales = np.where(self._scales > 1e-12, self._scales, 1.0)
        else:
            self._means = np.zeros(len(self.feature_names), dtype=float)
            self._scales = np.ones(len(self.feature_names), dtype=float)
        grouped: dict[str, list[dict[str, Any]]] = {}
        for row in self._observations:
            grouped.setdefault(str(row["code_hash"]), []).append(row)
        self._posteriors = {}
        for pid, rows in grouped.items():
            x = np.vstack([self._design(row["_x"]) for row in rows])
            y = np.asarray([float(row["observed_performance"]) for row in rows])
            precision = self.prior_precision + (x.T @ x) / self.observation_noise
            precision = (precision + precision.T) / 2
            covariance = np.linalg.inv(precision + self.ridge * np.eye(self.dimension))
            rhs = self.prior_precision @ self.prior_mean + (x.T @ y) / self.observation_noise
            mean = covariance @ rhs
            self._posteriors[pid] = (mean, (covariance + covariance.T) / 2, len(rows))
        self._fitted = True

    def _design(self, raw_features: np.ndarray) -> np.ndarray:
        scaled = (raw_features - self._means) / self._scales
        return np.concatenate(([1.0], scaled))

    def _posterior(self, pid: str) -> tuple[np.ndarray, np.ndarray, int]:
        return self._posteriors.get(
            pid,
            (self.prior_mean.copy(), np.linalg.inv(self.prior_precision), 0),
        )

    def posterior_parameters(self, program: Any) -> dict[str, Any]:
        """Return a copy of posterior mean/covariance and effective count."""
        pid = _program_id(program)
        mean, covariance, count = self._posterior(pid)
        return {
            "program_id": pid,
            "mean": mean.copy(),
            "covariance": covariance.copy(),
            "observation_count": count,
            "cold_start": count == 0,
            "feasibility": dict(self._feasibility.get(pid, {"feasible": 0, "infeasible": 0})),
            "model_schema": self.MODEL_SCHEMA,
        }

    def selection_evidence(self, program: Any, *,
                           instance_weights: Mapping[str, float] | None = None) -> dict[str, Any]:
        """Training-only real-run evidence for an offline selection gate.

        Do not use the legacy feasibility counters here: those counters also
        contain excluded validation/test records and are only an audit ledger.
        """
        pid = _program_id(program)
        records = [row for row in self._records.values()
                   if str(row["code_hash"]) == pid
                   and row.get("evaluation_split", "training") == self.training_split
                   and row.get("observation_source", "real_execution") == "real_execution"]
        # Several seeds/configs for identical content never create extra
        # reliability trials. A conflicting failure remains a failed instance.
        grouped: dict[str, list[dict]] = {}
        for row in records:
            key = str(row.get("instance_hash", ""))
            if instance_weights is None or instance_weights.get(key, 0.) > 0:
                grouped.setdefault(key, []).append(row)
        failed = {key: next(r for r in rows if not self._eligible(r))
                  for key, rows in grouped.items() if any(not self._eligible(r) for r in rows)}
        failures = list(failed.values())
        reasons: dict[str, int] = {}
        for row in failures:
            reason = str(row.get("failure_reason") or row.get("feasibility_status", "unknown"))
            reasons[reason] = reasons.get(reason, 0) + 1
        kinds = {key: failure_kind(row.get("failure_reason") or row.get("feasibility_status"))
                 for key, row in failed.items()}
        counts = {kind: sum(value == kind for value in kinds.values()) for kind in set(kinds.values())}
        mass = sum(1. if instance_weights is None else instance_weights[key] for key in grouped)
        failed_mass = sum(1. if instance_weights is None else instance_weights[key] for key in failed)
        return {"program_id": pid, "evaluation_split": self.training_split,
                "successful_observations": len(grouped) - len(failed),
                "failed_observations": len(failed), "failure_reasons": reasons,
                "distinct_instances": len(grouped), "failure_types": counts,
                "failure_instances": kinds,
                "weighted_failure_rate": failed_mass / mass if mass else None,
                "observed_target_weight": mass,
                "failure_rate_scope": "current_observed_training_target" if instance_weights is not None
                                      else "distinct_real_training_instances"}

    def joint_posterior_predictive(
        self, programs: Iterable[Any], instances: Iterable[Any], n_draws: int = 256,
        *, random_seed: int | None = None, include_noise: bool = True,
    ) -> JointLossSamples:
        """Draw one beta per program/draw, then predict all common instances.

        Cov(Y_h(x), Y_h(x')) = phi(x)' Sigma_h phi(x'), plus independent
        model-discrepancy variance on distinct instances when include_noise is
        true.  The current independent-program model has zero CROSS-program
        covariance; matching draw indices does not justify inventing any.
        Repeated content hashes share their residual as well as their beta.
        """
        started = time.perf_counter()
        if type(n_draws) is not int or n_draws < 1:
            raise ValueError("n_draws must be a positive integer")
        ids = tuple(dict.fromkeys(_program_id(p) for p in programs))
        items = list(instances)
        if not ids or not all(ids) or not items:
            raise ValueError("programs and instances must be nonempty")
        for item in items:
            if (isinstance(item, Mapping) and item.get("evaluation_split") is not None
                    and item["evaluation_split"] != self.training_split):
                raise ValueError("joint selection instances must belong to the fitted split")
        designs = np.vstack([self._design(_instance_features(x, self.feature_names)) for x in items])
        hashes = tuple(self._hash_features(x) for x in items)
        unique_hashes = list(dict.fromkeys(hashes))
        indices = np.asarray([unique_hashes.index(h) for h in hashes])
        for value in unique_hashes:
            positions = [i for i, h in enumerate(hashes) if h == value]
            if not np.allclose(designs[positions], designs[positions[0]], rtol=0, atol=0):
                raise ValueError("the same instance hash has conflicting features")
        seed = self.random_seed if random_seed is None else random_seed
        rng = np.random.default_rng(seed)
        values = np.empty((n_draws, len(ids), len(items)))
        known: dict[tuple[str, str], float] = {}
        for row in self._observations:
            key = (str(row["code_hash"]), str(row.get("instance_hash", "")))
            value = float(row["observed_performance"])
            if key in known and known[key] != value:
                raise ValueError("joint loss prediction requires one consistent evaluation configuration")
            known[key] = value
        conditioned = 0
        # Sort for candidate-order invariance, restore the requested axis order.
        for pid in sorted(ids):
            mean, covariance, _ = self._posterior(pid)
            root = np.linalg.cholesky(covariance)
            beta = mean + rng.standard_normal((n_draws, self.dimension)) @ root.T
            predicted = beta @ designs.T
            if include_noise:
                discrepancy = rng.normal(0, math.sqrt(self.observation_noise),
                                         (n_draws, len(unique_hashes)))
                predicted += discrepancy[:, indices]
            # A deterministic loss already observed on this exact content is
            # known, not another noisy rerun. Preserve real observations at
            # these cells; uncertainty belongs to the missing/future cells.
            for j, instance_hash in enumerate(hashes):
                if (pid, instance_hash) in known:
                    predicted[:, j] = known[pid, instance_hash]
                    conditioned += 1
            values[:, ids.index(pid), :] = predicted
        return JointLossSamples(ids, hashes, values, int(seed), self.MODEL_SCHEMA,
                                self.training_split, time.perf_counter() - started, include_noise,
                                conditioned_observed_cells=conditioned)

    def _predict(self, pid: str, instance: Any, n_draws: int,
                 standard_normals: np.ndarray | None = None) -> tuple[float, float, np.ndarray]:
        raw = _instance_features(instance, self.feature_names)
        design = self._design(raw)
        mean, covariance, _ = self._posterior(pid)
        predictive_mean = float(design @ mean)
        variance = float(self.observation_noise + design @ covariance @ design)
        variance = max(variance, 0.0)
        if standard_normals is None:
            standard_normals = np.random.default_rng(
                self.random_seed + self._stable_int(pid)
            ).normal(size=n_draws)
        if standard_normals.shape != (n_draws,):
            raise ValueError("standard_normals must have shape (n_draws,)")
        draws = predictive_mean + math.sqrt(variance) * standard_normals
        return predictive_mean, math.sqrt(variance), draws

    @staticmethod
    def _stable_int(value: str) -> int:
        return int.from_bytes(hashlib.sha256(value.encode("utf-8")).digest()[:4], "little")

    def posterior_predictive(
        self,
        programs: Iterable[Any] | Any,
        instances: Iterable[Any] | Any,
        n_draws: int = 256,
    ) -> dict[str, list[dict[str, Any]]]:
        """Return per-program, per-instance posterior predictive summaries."""
        if isinstance(programs, (str, Mapping)) or not isinstance(programs, Iterable):
            programs = [programs]
        is_feature_vector = (
            isinstance(instances, (list, tuple))
            and len(instances) == len(self.feature_names)
            and all(isinstance(value, (int, float, np.number)) for value in instances)
        )
        if isinstance(instances, (np.ndarray, Mapping)) or is_feature_vector or not isinstance(instances, Iterable):
            instances = [instances]
        programs = list(programs)
        instances = list(instances)
        n_draws = int(n_draws)
        if n_draws <= 0:
            raise ValueError("n_draws must be positive")
        output: dict[str, list[dict[str, Any]]] = {}
        for program in programs:
            pid = _program_id(program)
            if not pid:
                raise ValueError("program needs an id or code_hash")
            rows: list[dict[str, Any]] = []
            count = self._posterior(pid)[2]
            for instance in instances:
                mean, std, draws = self._predict(pid, instance, n_draws)
                rows.append({
                    "program_id": pid,
                    "instance_hash": self._hash_features(instance),
                    "mean": mean,
                    "std": std,
                    "draws": draws,
                    "observation_count": count,
                    "cold_start": count == 0,
                    "model_schema": self.MODEL_SCHEMA,
                })
            output[pid] = rows
        return output

    def expected_loss(self, program: Any, instance: Any | None = None) -> float:
        """Posterior mean loss for one instance, or average observed-design loss."""
        pid = _program_id(program)
        if instance is not None:
            return self._predict(pid, instance, 1)[0]
        rows = [row for row in self._observations if str(row["code_hash"]) == pid]
        if not rows:
            return float(self.prior_mean[0])
        return float(np.mean([self._predict(pid, row["_x"], 1)[0] for row in rows]))

    def marginal_gain_samples(
        self,
        candidate: Any,
        selected: Iterable[Any],
        instances: Iterable[Any],
        n_draws: int = 256,
    ) -> np.ndarray:
        """Use the joint parameter posterior for paired per-instance improvement.

        For compatibility an empty incumbent returns minus singleton CPI, a
        first-member ranking score, not a finite empty-set marginal gain.
        Bayesian-CPM reports that first step explicitly as singleton selection.
        """
        selected_ids = [_program_id(item) for item in selected]
        candidate_id = _program_id(candidate)
        bank = self.joint_posterior_predictive([*selected_ids, candidate_id], instances, n_draws)
        if not selected_ids:
            return -bank.cpi_samples([candidate_id])
        return bank.marginal_gain_samples(candidate_id, selected_ids)

    def _hash_features(self, instance: Any) -> str:
        if isinstance(instance, Mapping) and "instance_hash" in instance:
            return str(instance["instance_hash"])
        data = instance.to_dict() if hasattr(instance, "to_dict") else instance
        if isinstance(data, Mapping) and "arcs" in data and "commodities" in data:
            return hashlib.sha256(json.dumps(data, sort_keys=True, ensure_ascii=False,
                                            allow_nan=False).encode("utf-8")).hexdigest()
        vector = _instance_features(instance, self.feature_names)
        return hashlib.sha256(json.dumps(vector.tolist()).encode("utf-8")).hexdigest()

