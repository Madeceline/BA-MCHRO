"""Handled interruption exports real portfolios without resume or holdout reads."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import time
import unittest
from unittest.mock import patch

from m_asro import MultiAgentSNDSolver
from m_asro.problems.snd.baselines import BASE_SOLVER
from ba_eoh_s.bayesian import OracleBudgetError
from ba_eoh_s.bayesian.oracle import SolverOracleSession
from m_asro.problems.snd.evaluation import SNDEvaluator, digest
from m_asro.core.interruption import export_interrupted
from m_asro.runtime.performance import TimingRecorder
from m_asro.experiments.storage import ProgramArchive
from m_asro.problems.snd.generator import SNDInstance
from tests.test_bayesian_integration import small_config


class TestInterruptionExport(unittest.TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.solver = MultiAgentSNDSolver(SNDInstance(3, .7, .5, seed=7),
            generator_directions=("cost",), population_size=2,
            progress_output=False, make_plots=False, eoh_framework="eoh-s",
            response_mode="mpsro", output_dir=self.tmp.name, test_instance_count=1,
            reference_method="heuristic", reference_restarts=1)
        s = self.solver
        s.root = Path(self.tmp.name) / "run"
        s.archive = ProgramArchive(s.root)
        s.timing = TimingRecorder(s.archive)
        s.evaluator = SNDEvaluator(s.base, s.archive, s.utility, *s.reference_args,
                                  reference_config=s.reference_config, timing=s.timing)
        s._solve_started = time.perf_counter()
        s.user_prompt = "interrupted unit contract"
        self.programs = [s.archive.add(player="solver", role="solver", explanation="test",
            code=BASE_SOLVER.replace("1.0 * arc", f"{factor} * arc"), parents=[],
            iteration=1, eoh_iteration=0, operator="I1") for factor in (.2, 2.)]
        self.warm = s.archive.add(player="solver", role="solver", explanation="control",
            code=BASE_SOLVER, parents=[], iteration=0, eoh_iteration=0, operator="base")
        for program in self.programs:
            s.evaluator.evaluate(program, s.base)

    def remember(self, members=None, scenarios=None, stage=0):
        self.solver._remember_portfolio(members or self.programs,
            scenarios or [self.solver.base], None, 1, stage)

    def exported(self, result):
        path = self.solver.root / result["artifacts"]["final_solver"]
        spec = importlib.util.spec_from_file_location("interrupted_export", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module.FinalSNDSolver().solve(self.solver.base)

    def test_latest_portfolio_exports_all_members_with_original_mixture_semantics(self):
        self.remember()
        with patch.object(self.solver.evaluator, "evaluate", side_effect=AssertionError("no new payoff")):
            result = export_interrupted(self.solver, OracleBudgetError("budget exhausted"))
        self.assertEqual(result["selected_solver_set"], [p.id for p in self.programs])
        self.assertEqual(result["selected_solver"]["mixture_weights"], [.5, .5])
        self.assertEqual(result["interruption"]["additional_execution_count"], 0)
        self.assertEqual(result["training_status"], "interrupted")
        self.assertIsNone(result["test_evaluations"])
        self.assertIsNone(result["mixed_response_utility"])
        self.assertFalse(result["interruption"]["resumable"])
        deployed = self.exported(result)
        self.assertTrue(deployed["constraint_satisfaction"]["feasible"])
        self.assertAlmostEqual(deployed["objective"], result["objective"])

    def test_warm_start_removed_and_unselected_candidates_not_added(self):
        self.remember([self.warm, self.programs[0]])
        result = export_interrupted(self.solver, RuntimeError("generator error"))
        self.assertEqual(result["selected_solver_set"], [self.programs[0].id])
        selected = self.solver.archive.programs[-1]
        self.assertFalse(self.solver._contains_warm_start(selected))

    def test_pending_audit_exports_only_completely_observed_members(self):
        other = SNDInstance(3, .7, .5, seed=97).to_dict()
        self.solver.evaluator.evaluate(self.programs[0], other)
        self.remember(scenarios=[self.solver.base, other], stage=1)
        result = export_interrupted(self.solver, OracleBudgetError("wall limit"))
        self.assertEqual(result["selected_solver_set"], [self.programs[0].id])
        self.assertIn("incomplete_or_failed_real_audit", [
            r["reason"] for r in result["interruption"]["excluded_members"]])

    def test_incomplete_latest_set_falls_back_to_previous_audited_set(self):
        self.remember([self.programs[1]])
        other = SNDInstance(3, .7, .5, seed=97).to_dict()
        self.remember([self.programs[0]], [self.solver.base, other], stage=1)
        result = export_interrupted(self.solver, OracleBudgetError("wall limit"))
        self.assertEqual(result["selected_solver_set"], [self.programs[1].id])
        self.assertEqual(result["interruption"]["source_eoh_iteration"], 0)

    def test_no_portfolio_does_not_export_or_hide_error(self):
        self.solver.started = False
        def interrupted(prompt):
            raise OracleBudgetError("no eligible candidate")
        with patch.object(self.solver, "_solve", side_effect=interrupted):
            with self.assertRaisesRegex(OracleBudgetError, "no eligible"):
                self.solver.solve("prompt")
        self.assertFalse((self.solver.root / "final_solver").exists())
        self.assertFalse((self.solver.root / "result.json").exists())
        status = json.loads((self.solver.root / "interruption.json").read_text(encoding="utf-8"))
        self.assertEqual(status["export_status"], "no_validated_non_warm_portfolio")

    def test_export_wrapper_handles_keyboard_interrupt(self):
        self.remember()
        with patch.object(self.solver, "_solve", side_effect=KeyboardInterrupt):
            result = self.solver.solve("prompt")
        self.assertEqual(result.data["interruption"]["exception_type"], "KeyboardInterrupt")
        self.assertTrue(result.data["constraint_satisfaction"]["feasible"])

    def test_failed_validation_member_is_removed_without_retry_or_holdout_access(self):
        data = SNDInstance(3, .7, .5, seed=97).to_dict()
        self.solver.validation = [data]
        self.solver.tests = object()  # Accessing/iterating holdout must fail.
        for p in self.programs:
            self.solver.evaluator.evaluate(p, data)
        self.solver.evaluator.evaluations[self.programs[0].code_hash, digest(data)] = {
            "status": "solver_failure", "error": "timeout"}
        self.remember()
        with patch("m_asro.core.interruption.execute", side_effect=AssertionError("must not rerun")):
            result = export_interrupted(self.solver, RuntimeError("stopped"))
        self.assertEqual(result["selected_solver_set"], [self.programs[1].id])
        self.assertEqual(result["interruption"]["additional_execution_count"], 0)

    def test_forged_feasible_flag_still_requires_independent_solution_validation(self):
        self.remember([self.programs[0]])
        row = self.solver.evaluator.evaluations[self.programs[0].code_hash, digest(self.solver.base)]
        row["flows"] = {}
        row["constraint_satisfaction"] = {"feasible": True}
        self.assertIsNone(export_interrupted(self.solver, RuntimeError("stop")))
        self.assertFalse((self.solver.root / "final_solver").exists())

    def test_missing_validation_runs_without_reference_and_is_counted_separately(self):
        self.solver.validation = [SNDInstance(3, .7, .5, seed=97).to_dict()]
        self.remember([self.programs[0]])
        with patch.object(self.solver.evaluator, "reference", side_effect=AssertionError("no reference")):
            result = export_interrupted(self.solver, RuntimeError("reference timeout"))
        self.assertEqual(result["interruption"]["additional_execution_count"], 1)
        self.assertEqual(len(self.solver.evaluator.evaluations), 2)
        self.assertEqual(result["interruption"]["deployment_checks"][-1]["evaluation_split"], "validation")

    def test_ensemble_mode_retains_best_of_members_deployment(self):
        self.solver.response_mode = "pure"
        self.remember()
        result = export_interrupted(self.solver, RuntimeError("stop"))
        self.assertEqual(result["deployment_kind"], "ensemble")
        self.assertIsNone(result["selected_solver"]["mixture_weights"])
        self.assertAlmostEqual(self.exported(result)["objective"], result["objective"])

    def test_previous_outputs_are_preserved(self):
        self.remember()
        folder = self.solver.root / "final_solver"
        folder.mkdir()
        sentinel = folder / "partial.txt"
        sentinel.write_text("keep", encoding="utf-8")
        result = export_interrupted(self.solver, RuntimeError("failed original export"))
        self.assertEqual(sentinel.read_text(encoding="utf-8"), "keep")
        self.assertNotEqual(result["artifacts"]["final_solver"], "final_solver/solver.py")
        self.assertTrue(self.exported(result)["constraint_satisfaction"]["feasible"])

    def test_complete_final_preserved_when_reporting_fails(self):
        self.remember()
        first = export_interrupted(self.solver, RuntimeError("training stop"))
        source = (self.solver.root / "final_solver/solver.py").read_bytes()
        second = export_interrupted(self.solver, RuntimeError("plot failed"))
        self.assertEqual(second["selected_solver_set"], first["selected_solver_set"])
        self.assertEqual(second["interruption"]["export_status"], "existing_final_preserved")
        self.assertEqual((self.solver.root / "final_solver/solver.py").read_bytes(), source)


class TestInterruptedTraining(unittest.TestCase):
    def test_offline_wall_budget_interruption_exports_latest_audited_set(self):
        with TemporaryDirectory() as directory:
            solver = MultiAgentSNDSolver(SNDInstance(3, .7, .5, seed=7),
                generator_directions=("cost",), outer_iterations=1, eoh_iterations=1,
                population_size=2, n_candidates=2, instances_per_generator=1, seed=7,
                eoh_framework="eoh-s", response_mode="mpsro", parallel_workers=1,
                response_workers=1, generator_operators=("M1",), test_instance_count=1,
                validation_instance_count=1, progress_output=False, make_plots=False,
                reference_method="heuristic", reference_restarts=1, output_dir=directory,
                bayesian_config=small_config())
            original = SolverOracleSession.select
            def expire(session, candidates, stage):
                if stage == 1:
                    session.started -= session.settings.max_wall_time + 1
                    session.fail("Solver Oracle wall admission limit reached")
                return original(session, candidates, stage)
            with patch.object(SolverOracleSession, "select", expire):
                result = solver.solve("small offline interrupted training")
            self.assertEqual(result.data["training_status"], "interrupted")
            self.assertEqual(result.data["interruption"]["exception_type"], "OracleBudgetError")
            self.assertTrue(result.data["constraint_satisfaction"]["feasible"])
            self.assertIsNone(result.data["test_evaluations"])
            self.assertTrue((result.output_dir / "final_solver/solver.py").is_file())
            self.assertTrue(all(not solver._contains_warm_start(p)
                for p in solver.archive.programs if p.id in result.data["selected_solver_set"]))
            expected = solver._bayesian_reports[0]["selections"][0]["selected_ids"]
            self.assertEqual(result.data["selected_solver_set"], expected)

    def test_original_eohs_and_parallel_generator_failure_export_without_changing_game(self):
        with TemporaryDirectory() as directory:
            solver = MultiAgentSNDSolver(SNDInstance(3, .7, .5, seed=7),
                generator_directions=("cost",), outer_iterations=2, eoh_iterations=1,
                population_size=2, n_candidates=2, instances_per_generator=1, seed=7,
                eoh_framework="eoh-s", response_mode="pure", parallel_workers=2,
                response_workers=2, generator_operators=("M1",), test_instance_count=1,
                validation_instance_count=1, progress_output=False, make_plots=False,
                reference_method="heuristic", reference_restarts=1, output_dir=directory)
            original = solver._search
            def fail_generator(player, iteration, pools, meta):
                if iteration == 2 and player == "cost":
                    raise RuntimeError("injected generator failure")
                return original(player, iteration, pools, meta)
            with patch.object(solver, "_search", side_effect=fail_generator):
                result = solver.solve("parallel legacy interrupted training")
            self.assertEqual(result.data["training_status"], "interrupted")
            self.assertEqual(result.data["deployment_kind"], "ensemble")
            self.assertNotIn("bayesian", result.data)
            self.assertEqual(result.data["interruption"]["source_iteration"], 2)
            self.assertEqual(len(result.data["iterations"]), 2)  # Initial + completed first round.
            self.assertIn("injected generator failure", result.data["interruption"]["reason"])
            self.assertTrue(result.data["constraint_satisfaction"]["feasible"])


if __name__ == "__main__":
    unittest.main()
