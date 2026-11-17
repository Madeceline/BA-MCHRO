import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np

from ba_eoh_s.bayesian import BayesianPerformanceModel, ObservationStore


def observation(program, instance, value, *, seed=1, split="training", source="real_execution",
                status="feasible"):
    return {
        "program_id": program,
        "code_hash": program,
        "instance_hash": f"instance-{instance}",
        "instance_features": {"x": float(instance)},
        "seed": seed,
        "sample_index": None,
        "evaluation_split": split,
        "observed_performance": value,
        "solver_utility": None,
        "feasibility_status": status,
        "failure_reason": None if status == "feasible" else "timeout",
        "reference_kind": "heuristic",
        "reference_value": 1.0,
        "evaluation_time": 0.1,
        "cache_hit": source == "cache_reuse",
        "observation_source": source,
        "configuration_signature": "default",
        "player": "solver",
        "generator_id": None,
        "generator_direction": None,
    }


class TestBayesianPerformanceModel(unittest.TestCase):
    def setUp(self):
        self.model = lambda seed=42: BayesianPerformanceModel(
            feature_names=("x",), observation_noise=0.25, random_seed=seed
        )

    def test_conjugate_update_matches_closed_form(self):
        rows = [observation("p", 0, 1.0), observation("p", 1, 3.0, seed=2)]
        model = self.model().fit(rows)
        raw = np.array([[0.0], [1.0]])
        # The model standardises x using the training sample, then adds an
        # intercept.  Reproduce the exact conjugate formula independently.
        x = (raw - raw.mean(0)) / raw.std(0)
        design = np.c_[np.ones(2), x]
        prior_precision = np.eye(2)
        sigma2 = 0.25
        precision = prior_precision + design.T @ design / sigma2
        expected_cov = np.linalg.inv(precision)
        expected_mean = expected_cov @ (design.T @ np.array([1.0, 3.0]) / sigma2)
        posterior = model.posterior_parameters("p")
        np.testing.assert_allclose(posterior["mean"], expected_mean)
        np.testing.assert_allclose(posterior["covariance"], expected_cov)

    def test_sequential_update_equals_fit_and_deduplicates(self):
        rows = [observation("p", 0, 1.0), observation("p", 1, 3.0, seed=2)]
        one_shot = self.model().fit(rows)
        sequential = self.model().fit(rows[:1]).update(rows[1:]).update(rows[1:])
        np.testing.assert_allclose(
            one_shot.posterior_parameters("p")["mean"],
            sequential.posterior_parameters("p")["mean"],
        )
        self.assertEqual(sequential.posterior_parameters("p")["observation_count"], 2)

    def test_sparse_and_cold_start_prediction(self):
        model = self.model().fit([observation("p", 0, 1.0)])
        result = model.posterior_predictive(["p", "cold"], [{"x": 2.0}], n_draws=64)
        self.assertFalse(result["p"][0]["cold_start"])
        self.assertTrue(result["cold"][0]["cold_start"])
        self.assertGreater(result["cold"][0]["std"], result["p"][0]["std"])
        self.assertEqual(result["cold"][0]["draws"].shape, (64,))

    def test_test_and_prediction_rows_do_not_update_posterior(self):
        model = self.model().fit([observation("p", 0, 1.0)])
        before = model.posterior_parameters("p")["mean"].copy()
        model.update([observation("p", 1, -100, split="test"),
                      observation("p", 2, -100, source="posterior_prediction")])
        np.testing.assert_allclose(before, model.posterior_parameters("p")["mean"])

    def test_failures_are_counted_but_not_regressed(self):
        model = self.model().fit([observation("p", 0, None, status="timeout")])
        self.assertTrue(model.posterior_parameters("p")["cold_start"])
        self.assertEqual(model.posterior_parameters("p")["feasibility"]["infeasible"], 1)

    def test_posterior_draws_are_reproducible(self):
        rows = [observation("p", 0, 1.0)]
        a = self.model(9).fit(rows).posterior_predictive("p", {"x": 1.5}, 32)
        b = self.model(9).fit(rows).posterior_predictive("p", {"x": 1.5}, 32)
        np.testing.assert_array_equal(a["p"][0]["draws"], b["p"][0]["draws"])

    def test_wide_predictive_interval_has_toy_coverage(self):
        # A deterministic regression check rather than a claim of real-data
        # calibration: a known low-noise linear process should be covered by
        # the deliberately wide 95% predictive intervals on most test points.
        rows = [observation("p", i, 1.0 + 2.0 * i, seed=i + 1) for i in range(8)]
        model = BayesianPerformanceModel(
            feature_names=("x",), observation_noise=0.01, random_seed=4
        ).fit(rows)
        covered = 0
        for i in range(8, 18):
            summary = model.posterior_predictive("p", {"x": float(i)}, n_draws=256)["p"][0]
            lower = summary["mean"] - 1.96 * summary["std"]
            upper = summary["mean"] + 1.96 * summary["std"]
            covered += lower <= 1.0 + 2.0 * i <= upper
        self.assertGreaterEqual(covered, 7)

    def test_paired_marginal_gain_and_empty_instances(self):
        model = self.model().fit([observation("p", 0, 1.0)])
        gains = model.marginal_gain_samples("q", ["p"], [{"x": 0.0}, {"x": 1.0}], 32)
        self.assertEqual(gains.shape, (32,))
        with self.assertRaises(ValueError):
            model.marginal_gain_samples("q", [], [], 4)


class TestObservationStorePersistence(unittest.TestCase):
    def test_dedup_cache_replacement_and_jsonl(self):
        cached = observation("p", 0, 1.0, source="cache_reuse")
        real = observation("p", 0, 1.0, source="real_execution")
        store = ObservationStore.from_dicts([cached, cached])
        self.assertEqual(store.unique_count, 1)
        self.assertEqual(store.actual_evaluation_count, 0)
        store.extend([real])
        self.assertEqual(store.actual_evaluation_count, 1)
        with TemporaryDirectory() as directory:
            path = Path(directory) / "observations.jsonl"
            store.to_jsonl(path)
            restored = ObservationStore.from_jsonl(path)
            self.assertEqual(restored.to_list(), store.to_list())
            first = json.loads(path.read_text(encoding="utf-8").splitlines()[0])
            self.assertEqual(first["code_hash"], "p")


if __name__ == "__main__":
    unittest.main()
