"""MIP fallback never bypasses feasibility validation or fabricates references."""
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from m_asro.problems.snd.evaluation import SNDEvaluator
from m_asro.runtime.execution import ExecutionError
from m_asro.problems.snd.reference import ReferenceConfig
from m_asro.experiments.storage import ProgramArchive
from m_asro.core.utilities import UtilityConfig


class ReferenceFallbackTests(unittest.TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.data = {"nodes": [0, 1], "arcs": [{"id": "a", "tail": 0, "head": 1,
            "capacity": 5., "fixed_cost": 2., "variable_cost": 1.}],
            "commodities": [{"id": "k", "origin": 0, "destination": 1, "demand": 3.}]}
        self.solution = {"kind": "heuristic", "is_optimal": False,
            "arc_open": {"a": 1}, "flows": {"k": {"a": 3.}}}

    def evaluator(self, enabled=True):
        return SNDEvaluator(self.data, ProgramArchive(self.root), UtilityConfig(),
            "scipy", 1., .05, 5., 1, 42,
            reference_config=ReferenceConfig(method="auto", mip_fallback=enabled),
            reference_cache_dir=self.root / "cache")

    def test_timeout_fallback_and_disk_cache(self):
        evaluator = self.evaluator()
        with patch("m_asro.runtime.execution.isolated_call", side_effect=[
                ExecutionError("timeout", failure_type="timeout"),
                {"result": self.solution, "wall_time": .1}]) as call:
            row = evaluator.reference(self.data)
        self.assertEqual([c.args[0] for c in call.call_args_list],
                         ["reference_mip", "reference_heuristic"])
        self.assertEqual(row["selected_method"], "heuristic")
        self.assertEqual(row["requested_method"], "mip")
        self.assertEqual(row["value"], 5.)
        self.assertIn("timeout", row["mip_fallback_reason"])
        with patch("m_asro.runtime.execution.isolated_call") as call:
            self.assertEqual(self.evaluator().reference(self.data)["value"], 5.)
            call.assert_not_called()

    def test_disabled_preserves_exception(self):
        with patch("m_asro.runtime.execution.isolated_call", side_effect=ExecutionError("timeout")) as call:
            with self.assertRaises(ExecutionError):
                self.evaluator(False).reference(self.data)
            self.assertEqual(call.call_count, 1)

    def test_invalid_mip_and_invalid_fallback_rejected(self):
        invalid = {"result": {**self.solution, "flows": {}}, "wall_time": .1}
        evaluator = self.evaluator()
        with patch("m_asro.runtime.execution.isolated_call", side_effect=[invalid, invalid]):
            with self.assertRaisesRegex(RuntimeError, "启发式回退也失败"):
                evaluator.reference(self.data)
        self.assertEqual(evaluator.references, {})

    def test_successful_mip_does_not_fallback(self):
        with patch("m_asro.runtime.execution.isolated_call",
                   return_value={"result": self.solution, "wall_time": .1}) as call:
            self.assertEqual(self.evaluator().reference(self.data)["selected_method"], "mip")
            self.assertEqual(call.call_count, 1)
