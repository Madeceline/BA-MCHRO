"""Joint loss draws and finite-target portfolio statistics (lower is better)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


def target_weights(weights: Sequence[float] | None, width: int) -> np.ndarray:
    """Normalised quadrature weights for a fixed representative instance set."""
    values = np.ones(width) if weights is None else np.asarray(weights, dtype=float)
    if (width < 1 or values.shape != (width,) or not np.all(np.isfinite(values))
            or np.any(values < 0) or not np.isfinite(values.sum()) or values.sum() <= 0):
        raise ValueError("weights must be finite, nonnegative, aligned and have positive sum")
    return values / values.sum()


@dataclass(frozen=True)
class JointLossSamples:
    """One joint posterior bank, axes = (draw, program, common instance).

    The bank is reused throughout selection.  It is always labelled as a
    prediction, never an evaluator row or an empirical payoff matrix.
    """

    program_ids: tuple[str, ...]
    instance_hashes: tuple[str, ...]
    losses: np.ndarray
    random_seed: int
    model_schema: str
    evaluation_split: str = "training"
    inference_seconds: float = 0.0
    include_noise: bool = True
    observation_source: str = "posterior_prediction"
    conditioned_observed_cells: int = 0

    def __post_init__(self) -> None:
        values = np.array(self.losses, dtype=float, copy=True)
        if (values.ndim != 3 or values.shape[0] < 1 or not self.program_ids
                or not self.instance_hashes or values.shape[1:] != (
                    len(self.program_ids), len(self.instance_hashes))):
            raise ValueError("joint losses must have shape (draws, programs, instances)")
        if len(set(self.program_ids)) != len(self.program_ids):
            raise ValueError("joint bank must contain unique program ids")
        if not np.all(np.isfinite(values)):
            raise ValueError("joint posterior losses must be finite")
        if self.observation_source != "posterior_prediction":
            raise ValueError("joint bank is a prediction, not an observation")
        values.setflags(write=False)
        object.__setattr__(self, "losses", values)

    def cpi_samples(self, selected_ids: Sequence[str],
                    weights: Sequence[float] | None = None) -> np.ndarray:
        if not selected_ids:
            raise ValueError("CPI of the empty portfolio is not finite")
        indices = [self.program_ids.index(pid) for pid in dict.fromkeys(selected_ids)]
        importance = target_weights(weights, len(self.instance_hashes))
        # The minimum is across programs on EACH common instance, before the
        # expectation over instances or posterior draws.
        return self.losses[:, indices, :].min(axis=1) @ importance

    def marginal_gain_samples(self, candidate_id: str, selected_ids: Sequence[str],
                              weights: Sequence[float] | None = None) -> np.ndarray:
        if not selected_ids:
            raise ValueError("use singleton CPI to choose the first member; empty-set gain is undefined")
        before = self.cpi_samples(selected_ids, weights)
        after = self.cpi_samples([*selected_ids, candidate_id], weights)
        return before - after


def posterior_summary(draws: np.ndarray, credible_level: float = 0.95,
                      epsilon: float | None = None) -> dict:
    """Equal-tailed posterior interval; MC error is reported separately."""
    values = np.asarray(draws, dtype=float)
    if values.ndim != 1 or not len(values) or not np.all(np.isfinite(values)):
        raise ValueError("summary needs a finite one-dimensional draw vector")
    if not np.isfinite(credible_level) or not 0 < credible_level < 1:
        raise ValueError("credible_level must lie in (0, 1)")
    if epsilon is not None and (not np.isfinite(epsilon) or epsilon < 0):
        raise ValueError("epsilon must be finite and nonnegative")
    alpha = (1 - credible_level) / 2
    std = float(values.std(ddof=1)) if len(values) > 1 else 0.0
    result = {
        "mean": float(values.mean()), "variance": std ** 2, "std": std,
        "credible_interval": np.quantile(values, [alpha, 1 - alpha]).tolist(),
        "credible_level": credible_level, "mc_standard_error": std / np.sqrt(len(values)),
        "n_draws": len(values), "observation_source": "posterior_prediction",
    }
    if epsilon is not None:
        probability = float(np.mean(values > epsilon))
        result.update({"epsilon": epsilon, "improvement_probability": probability,
                       "probability_mc_standard_error": float(np.sqrt(
                           probability * (1 - probability) / len(values)))})
    return result
