"""Budget, pairing and isolation invariants for the independent active oracle."""
from contextlib import redirect_stdout
from dataclasses import replace
import io
from pathlib import Path
from tempfile import TemporaryDirectory
import threading
import time
import unittest

import numpy as np

from ba_eoh_s.bayesian import (
    ActiveEvaluationConfig, ActiveInstance, BayesianActiveEvaluator, BayesianCPMConfig,
    BayesianPerformanceModel, EvaluationObservation, EvaluationOutcome, JointLossSamples,
    ObservationStore,
    SNDEvaluationBackend,
)
from m_asro.runtime.progress import terminal_heartbeat
from tests.test_bayesian import observation


class DeterministicBackend:
    """Analytic toy loss, explicitly NOT a measured SND performance claim."""
    configuration_signature = "default"

    def __init__(self, *, fail=None, raise_error=False, delay=0.):
        self.cache, self.calls = {}, []
        self.fail, self.raise_error, self.delay = fail, raise_error, delay

    def peek(self, program, instance):
        row = self.cache.get((program, instance.instance_hash))
        return None if row is None else replace(row, cache_hit=True)

    def evaluate(self, program, instance):
        existing = self.peek(program, instance)
        if existing:
            return EvaluationOutcome(existing, False, True)
        self.calls.append((program, instance.instance_hash))
        if self.delay:
            time.sleep(self.delay)
        if self.raise_error:
            raise TimeoutError("toy timeout")
        x = instance.features["x"]
        loss = {"a": 2+x, "b": 2-x, "c": 4.0}[program]
        row = EvaluationObservation.from_dict(observation(
            program, int(x), None if program == self.fail else loss,
            status="timeout" if program == self.fail else "feasible"))
        row = replace(row, instance_hash=instance.instance_hash, instance_features=instance.features,
                      seed=instance.seed, sample_index=instance.sample_index)
        self.cache[(program, instance.instance_hash)] = row
        return EvaluationOutcome(row, True, False)


def pool(n=8):
    return [ActiveInstance(None, f"instance-{i}", {"x": float(i)}, i, i) for i in range(n)]


def controller(backend=None, *, store=None, callback=None, **settings):
    defaults = dict(total_budget=15, progress_output=False, random_seed=9)
    defaults.update(settings)
    return BayesianActiveEvaluator(
        BayesianPerformanceModel(feature_names=("x",), observation_noise=.01),
        backend or DeterministicBackend(), config=ActiveEvaluationConfig(**defaults),
        cpm_config=BayesianCPMConfig(n_draws=64, random_seed=9), store=store, event_callback=callback)


class TestActiveEvaluation(unittest.TestCase):
    def test_initial_controls_and_new_runs_share_one_hard_budget(self):
        backend = DeterministicBackend()
        active = controller(backend, total_budget=13, batch_size=3)
        selected, log = active.run(["a", "b", "c"], pool(), 2)
        self.assertEqual(log["actual_evaluation_count"], len(backend.calls))
        self.assertLessEqual(len(backend.calls), 13)
        self.assertEqual(len(set(backend.calls)), len(backend.calls))
        self.assertEqual(sum(e["event"] == "evaluation_done" and e["phase"] == "initial_uniform"
                             for e in log["events"]), 6)
        known = set()
        for event in log["events"]:
            if event["event"] == "evaluation_done":
                known.add((event["program_id"], event["instance_hash"]))
            if event["event"] == "paired_complete":
                self.assertTrue(event["all_members_observed"])
                for pid in [event["candidate"], *event["controls"]]:
                    self.assertIn((pid, event["instance_hash"]), known)
        self.assertEqual(len(selected), 2)

    def test_cache_restoration_and_different_sample_seed_do_not_add_samples(self):
        backend = DeterministicBackend()
        _, first = controller(backend, total_budget=12).run(["a", "b", "c"], pool(), 2)
        count = len(backend.calls)
        _, second = controller(backend, total_budget=0).run(
            ["a", "b", "c"], [replace(x, seed=x.seed+1000) for x in pool()], 2)
        self.assertEqual(len(backend.calls), count)
        self.assertEqual(second["actual_evaluation_count"], 0)
        self.assertEqual(second["cache_hit_count"], count)
        self.assertEqual(second["effective_observation_count"], first["effective_observation_count"])

    def test_historical_observations_are_not_reexecuted(self):
        backend = DeterministicBackend()
        store = ObservationStore()
        controller(backend, store=store, total_budget=6).run(["a", "b", "c"], pool(), 2)
        _, log = controller(DeterministicBackend(), store=store, total_budget=0).run(["a", "b", "c"], pool(), 2)
        self.assertEqual(log["actual_evaluation_count"], 0)
        self.assertEqual(log["effective_observation_count"], 6)
        self.assertEqual(store.unique_count, 6)

    def test_small_budget_never_partially_admits_initial_bundle(self):
        backend = DeterministicBackend()
        selected, log = controller(backend, total_budget=2).run(["a", "b", "c"], pool(), 2)
        self.assertEqual(backend.calls, [])
        self.assertEqual(selected, [])
        self.assertEqual(log["stop_reason"], "initial_bundle_does_not_fit")

    def test_new_program_requires_real_cold_start(self):
        selected, log = controller(total_budget=3).run(["a", "b", "c"], pool(), 2)
        self.assertEqual(selected, [])
        self.assertEqual(log["effective_observation_count"], 3)
        self.assertTrue(all(r["reason"] == "insufficient_real_observations"
                            for r in log["cpm"]["excluded_candidates"].values()))

    def test_failed_or_raising_backend_attempts_are_counted_and_not_selected(self):
        backend = DeterministicBackend(fail="b")
        selected, log = controller(backend, total_budget=10).run(["a", "b", "c"], pool(), 2)
        self.assertNotIn("b", selected)
        self.assertEqual(log["actual_evaluation_count"], len(backend.calls))
        self.assertEqual(log["cpm"]["excluded_candidates"]["b"]["reason"], "insufficient_reliability_observations")
        broken = DeterministicBackend(raise_error=True)
        selected, log = controller(broken, total_budget=7).run(["a", "b", "c"], pool(), 2)
        self.assertEqual(selected, [])
        self.assertEqual(log["actual_evaluation_count"], 7)
        self.assertEqual(len(broken.calls), 7)
        self.assertEqual(log["effective_observation_count"], 0)

    def test_all_methods_are_reproducible_with_identical_start(self):
        starts = []
        for method in ("ucb", "random", "uniform"):
            outputs = []
            for _ in range(2):
                backend = DeterministicBackend()
                selected, log = controller(backend, method=method, batch_size=2).run(["a", "b", "c"], pool(), 2)
                outputs.append((selected, backend.calls, log["posterior_cpi"]))
            self.assertEqual(outputs[0], outputs[1])
            starts.append(outputs[0][1][:6])
        self.assertEqual(starts[0], starts[1])
        self.assertEqual(starts[1], starts[2])

    def test_split_source_feature_and_configuration_mismatch_rejected(self):
        for key, value in (("evaluation_split", "test"), ("observation_source", "posterior_prediction"),
                           ("configuration_signature", "wrong"), ("instance_features", {"x": 999.})):
            backend = DeterministicBackend()
            real_evaluate = backend.evaluate

            def invalid(program, instance):
                receipt = real_evaluate(program, instance)
                return replace(receipt, observation=replace(receipt.observation, **{key: value}))

            backend.evaluate = invalid
            with self.subTest(key=key), self.assertRaises(ValueError):
                controller(backend).run(["a", "b"], pool(), 1)
        with self.assertRaises(ValueError):
            controller().run(["a"], [replace(pool()[0], evaluation_split="test")], 1)

    def test_held_out_content_overlap_and_duplicate_target_rejected(self):
        row = EvaluationObservation.from_dict(observation("a", 0, 2., split="test"))
        with self.assertRaises(ValueError):
            controller(store=ObservationStore([row])).run(["a"], pool(), 1)
        with self.assertRaises(ValueError):
            controller().run(["a"], [pool()[0], pool()[0]], 1)

    def test_content_identity_cannot_be_changed_by_metadata(self):
        from m_asro.problems.snd.generator import SNDInstance
        instance = ActiveInstance.from_snd(SNDInstance(3, .7, .5, seed=7), seed=7)
        with self.assertRaises(ValueError):
            replace(instance, instance_hash="forged-training-identity")
        instance.data["arcs"][0]["capacity"] += 1
        with self.assertRaises(ValueError):
            controller().run(["a"], [instance], 1)

    def test_stop_threshold_and_minimum_exploration(self):
        _, log = controller(total_budget=24, information_threshold=1e100,
                            min_exploration_fraction=.25).run(["a", "b", "c"], pool(12), 2)
        self.assertEqual(log["stop_reason"], "information_value_threshold")
        self.assertGreaterEqual(log["exploration_action_count"], 1)
        self.assertLess(log["actual_evaluation_count"], 24)

    def test_wall_admission_limit_retains_underlying_timeout(self):
        backend = DeterministicBackend(delay=.03)
        _, log = controller(backend, max_wall_time=.01).run(["a", "b", "c"], pool(), 2)
        self.assertEqual(len(backend.calls), 1)
        self.assertEqual(log["actual_evaluation_count"], 1)
        self.assertEqual(log["stop_reason"], "wall_time_limit")

    def test_complementarity_not_raw_variance_and_cost_penalises_actions(self):
        # a has enormous variance but is always worse than the incumbent;
        # b has a modest, certain complementary improvement.
        active = controller()
        active.programs = {"a": "a", "b": "b", "c": "c"}
        active.instances = pool(1)
        active.weights = np.array([1.])
        active.known, active.count = {}, 0
        losses = np.array([[[20.], [0.], [1.]], [[2000.], [0.], [1.]]])
        bank = JointLossSamples(("a", "b", "c"), ("instance-0",), losses, 0, "toy")
        # Acquisition receives already qualified programs in the real loop.
        active._evidence = lambda pid: {"successful_observations": 2, "failed_observations": 0}
        actions = {a["candidate"]: a for a in active._actions(["c"], bank)}
        self.assertEqual(actions["a"]["score"], 0.)
        self.assertGreater(actions["b"]["score"], 0.)
        self.assertEqual(actions["b"]["missing"], ["c", "b"])
        base = actions["b"]["score"]
        active.config = replace(active.config, default_evaluation_cost=10.)
        slow = {a["candidate"]: a for a in active._actions(["c"], bank)}
        self.assertAlmostEqual(slow["b"]["score"], base/10)

    def test_progress_callback_and_single_use(self):
        events = []
        active = controller(callback=events.append, progress_output=True, total_budget=6)
        with redirect_stdout(io.StringIO()) as output:
            _, log = active.run(["a", "b", "c"], pool(), 2)
        self.assertIn("[BA评估/ucb]", output.getvalue())
        self.assertIn("6/6", output.getvalue())
        self.assertEqual(events, log["events"])
        with self.assertRaises(RuntimeError):
            active.run(["a"], pool(), 1)

    def test_config_rejects_evi_and_invalid_values(self):
        for kwargs in ({"method": "evi"}, {"total_budget": -1}, {"total_budget": True},
                       {"max_wall_time": float("nan")}, {"kappa": -1}, {"batch_size": 0},
                       {"min_exploration_fraction": 1.1}, {"progress_output": "yes"}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                ActiveEvaluationConfig(**kwargs)


class TestTerminalProgress(unittest.TestCase):
    def test_heartbeat_prints_and_stops_even_on_error(self):
        with redirect_stdout(io.StringIO()) as output:
            with self.assertRaises(RuntimeError):
                with terminal_heartbeat("training", interval=.005):
                    time.sleep(.025)
                    raise RuntimeError("stop")
        self.assertIn("training", output.getvalue())
        self.assertFalse(any(t.name == "snd-terminal-progress" for t in threading.enumerate()))

    def test_disabled_is_silent(self):
        with redirect_stdout(io.StringIO()) as output:
            with terminal_heartbeat("quiet", enabled=False, interval=.001):
                time.sleep(.005)
        self.assertEqual(output.getvalue(), "")


class TestSNDBackendIntegration(unittest.TestCase):
    def test_original_isolated_evaluator_and_real_cache_receipts(self):
        from m_asro.problems.snd.baselines import BASE_SOLVER
        from m_asro.problems.snd.evaluation import SNDEvaluator
        from m_asro.problems.snd.reference import ReferenceConfig
        from m_asro.experiments.storage import ProgramArchive
        from m_asro.core.utilities import UtilityConfig
        from m_asro.problems.snd.generator import SNDInstance

        with TemporaryDirectory() as directory:
            archive = ProgramArchive(Path(directory))
            program = archive.add(player="solver", role="solver", explanation="frozen test candidate",
                code=BASE_SOLVER.replace("1.0 * arc", "0.2 * arc"), parents=[], iteration=0,
                eoh_iteration=0, operator="I1")
            instances = [ActiveInstance.from_snd(SNDInstance(3, .7, .5, seed=seed), seed=seed)
                         for seed in (41, 42)]
            evaluator = SNDEvaluator(instances[0].data, archive, UtilityConfig(), "scipy", 3., 0.,
                15., 1, 41, reference_config=ReferenceConfig(method="mip"), parallel_workers=1)
            backend = SNDEvaluationBackend(evaluator)
            store = ObservationStore()

            def run(budget):
                return BayesianActiveEvaluator(BayesianPerformanceModel(), backend,
                    config=ActiveEvaluationConfig(total_budget=budget, progress_output=False),
                    cpm_config=BayesianCPMConfig(n_draws=32), store=store).run([program], instances, 1)

            selected, log = run(2)
            self.assertEqual([p.code_hash for p in selected], [program.code_hash])
            self.assertEqual(log["actual_evaluation_count"], 2)
            self.assertEqual(log["effective_observation_count"], 2)
            self.assertEqual(evaluator.timing.summary()["stages"]["candidate_process"]["calls"], 2)
            self.assertTrue(all(row.feasible for row in store.records()))
            self.assertTrue(all(row.reference_kind is not None for row in store.records()))
            # Reuse original evaluator caches in a completely fresh ledger.
            fresh = BayesianActiveEvaluator(BayesianPerformanceModel(), backend,
                config=ActiveEvaluationConfig(total_budget=0, progress_output=False),
                cpm_config=BayesianCPMConfig(n_draws=32))
            _, second = fresh.run([program], instances, 1)
            self.assertEqual(second["actual_evaluation_count"], 0)
            self.assertEqual(second["cache_hit_count"], 2)
            self.assertEqual(second["effective_observation_count"], 2)
            self.assertEqual(evaluator.timing.summary()["stages"]["candidate_process"]["calls"], 2)


if __name__ == "__main__":
    unittest.main()
