"""Decorators/default expressions use Python semantics inside existing limits."""
from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from m_asro.problems.snd.deployment import ProgramSNDSolver
from m_asro.runtime.execution import ExecutionError, execute, validate_code


def candidate(default: str, *, kind: str = "function", keyword_only: bool = False) -> str:
    parameter = f"value={default}"
    if keyword_only:
        parameter = "*, " + parameter
    if kind == "lambda":
        body = f"    helper = lambda {parameter}: value\n"
    else:
        prefix = "async " if kind == "async" else ""
        body = f"    {prefix}def helper({parameter}):\n        return value\n"
    return f"def solve_snd(instance):\n{body}    return helper()\n"


class TestCandidateDefaults(unittest.TestCase):
    def test_scalar_defaults_allowed_in_function_and_lambda(self):
        for value in ("None", "True", "False", "0", "2", "-3", "+4", "1.25", "-0.5",
                      "1e-9", "2j", "-2j", "''", "'path'", "b'bytes'"):
            for kind in ("function", "lambda", "async"):
                for keyword_only in (False, True):
                    with self.subTest(value=value, kind=kind, keyword_only=keyword_only):
                        validate_code(candidate(value, kind=kind, keyword_only=keyword_only),
                                      "solve_snd(instance)")

    def test_mutable_defaults_allowed(self):
        for value in ("[]", "{}", "{1, 2}", "list()", "dict()", "set()",
                      "[x for x in range(3)]"):
            for kind in ("function", "lambda", "async"):
                for keyword_only in (False, True):
                    with self.subTest(value=value, kind=kind, keyword_only=keyword_only):
                        validate_code(candidate(value, kind=kind, keyword_only=keyword_only),
                                      "solve_snd(instance)")

    def test_calls_references_and_expressions_allowed(self):
        for value in ("instance", "instance.get('x')", "float('inf')", "1 + 2", "(1, 2)",
                      "()", "...", "(lambda: 2)()", "[1][0]", "(value := 1)",
                      "'a' * 100", "True if instance else False", "not False", "1e309", "-1e309"):
            with self.subTest(value=value):
                validate_code(candidate(value), "solve_snd(instance)")

    def test_required_keyword_only_argument_is_not_a_default(self):
        code = ("def solve_snd(instance):\n"
                "    def helper(*, value, retries=2):\n"
                "        return value + retries\n"
                "    return helper(value=3)\n")
        validate_code(code, "solve_snd(instance)")

    def test_entry_optional_arguments_allowed(self):
        validate_code("def solve_snd(instance=None):\n    return instance\n", "solve_snd(instance)")
        validate_code("def solve_snd(instance, retries=2):\n    return {}\n", "solve_snd(instance)")
        validate_code("def solve_snd(instance, *, retries=2):\n    return {}\n", "solve_snd(instance)")

    def test_decorators_allowed(self):
        code = ("def solve_snd(instance):\n"
                "    @dict\n"
                "    def helper(value=None):\n"
                "        return value\n"
                "    return {}\n")
        validate_code(code, "solve_snd(instance)")

    def test_default_and_decorator_operations_are_not_statically_filtered(self):
        for expression in ("open('blocked')", "eval('1')", "exec('pass')", "__import__('os')",
                           "os.getcwd()", "(1).__class__"):
            for code in (candidate(expression),
                         f"@{expression}\ndef solve_snd(instance):\n    return {{}}\n"):
                with self.subTest(expression=expression, code=code):
                    with patch("m_asro.runtime.execution.isolated_call") as dispatch:
                        execute(code, "solve_snd", {})
                        dispatch.assert_called_once()

    def test_source_limits_still_enforced(self):
        for limits in ({"max_code_lines": 1}, {"max_code_chars": 5}, {"max_ast_nodes": 1}):
            with self.subTest(limits=limits), self.assertRaisesRegex(ValueError, "超过"):
                validate_code(candidate("None"), "solve_snd(instance)", **limits)

    def test_solver_defaults_execute_deterministically_in_isolation(self):
        code = ("def solve_snd(instance):\n"
                "    def helper(value=None, weight=-0.5, *, retries=2, enabled=True, label='ok'):\n"
                "        amount = instance['amount'] if value is None else value\n"
                "        return dict(value=amount * weight + retries, enabled=enabled, label=label)\n"
                "    return helper()\n")
        first = execute(code, "solve_snd", {"amount": 6}, timeout=10.)
        second = execute(code, "solve_snd", {"amount": 6}, timeout=10.)
        self.assertEqual(first["result"], {"value": -1., "enabled": True, "label": "ok"})
        self.assertEqual(first["result"], second["result"])
        self.assertEqual(first["lines"], second["lines"])

    def test_generator_defaults_execute_with_original_interface(self):
        code = ("def generate_snd(base_instance, seed):\n"
                "    def helper(scale=2, offset=0):\n"
                "        return dict(value=base_instance['value'] * scale + offset + seed)\n"
                "    return helper()\n")
        result = execute(code, "generate_snd", {"value": 3}, 7, timeout=10.)
        self.assertEqual(result["result"], {"value": 13})

    def test_mutable_defaults_share_within_call_and_reset_across_executions(self):
        code = ("def solve_snd(instance):\n"
                "    def helper(items=[], counts={}, seen=set()):\n"
                "        items.append(instance['value'])\n"
                "        counts['n'] = counts.get('n', 0) + 1\n"
                "        seen.add(len(items))\n"
                "        return [len(items), counts['n'], len(seen)]\n"
                "    return [helper(), helper()]\n")
        first = execute(code, "solve_snd", {"value": 7}, timeout=10.)
        second = execute(code, "solve_snd", {"value": 7}, timeout=10.)
        self.assertEqual(first["result"], [[1, 1, 1], [2, 2, 2]])
        self.assertEqual(first["result"], second["result"])
        self.assertEqual(first["lines"], second["lines"])

    def test_custom_function_default_runs_once_at_definition(self):
        code = ("def solve_snd(instance):\n"
                "    events = []\n"
                "    def build():\n"
                "        events.append('built')\n"
                "        return dict(value=instance['value'])\n"
                "    def helper(value=build()):\n"
                "        value['value'] += 1\n"
                "        return value['value']\n"
                "    result = [helper(), helper()]\n"
                "    return dict(result=result, events=events)\n")
        result = execute(code, "solve_snd", {"value": 4}, timeout=10.)
        self.assertEqual(result["result"], {"result": [5, 6], "events": ["built"]})

    def test_lambda_mutable_and_keyword_only_call_defaults_execute(self):
        code = ("def solve_snd(instance):\n"
                "    helper = lambda items=[], *, scale=sum([1, 2]): (items.append(scale), len(items))[1]\n"
                "    return [helper(), helper()]\n")
        self.assertEqual(execute(code, "solve_snd", {}, timeout=10.)["result"], [1, 2])

    def test_nested_decorator_factory_and_stacked_decorators_execute(self):
        code = ("def solve_snd(instance):\n"
                "    def multiply(factor):\n"
                "        def decorate(fn):\n"
                "            def wrapped(value):\n"
                "                return factor * fn(value)\n"
                "            return wrapped\n"
                "        return decorate\n"
                "    @multiply(2)\n"
                "    @multiply(3)\n"
                "    def helper(value):\n"
                "        return value + 1\n"
                "    return helper(instance['value'])\n")
        self.assertEqual(execute(code, "solve_snd", {"value": 4}, timeout=10.)["result"], 30)

    def test_entry_decorator_executes(self):
        code = ("@(lambda fn: (lambda instance: fn(instance) + 2))\n"
                "def solve_snd(instance):\n"
                "    return instance['value']\n")
        self.assertEqual(execute(code, "solve_snd", {"value": 4}, timeout=10.)["result"], 6)

    def test_generator_decorated_helper_with_mutable_default_executes(self):
        code = ("def generate_snd(base_instance, seed):\n"
                "    def decorate(fn):\n"
                "        return fn\n"
                "    @decorate\n"
                "    def helper(values=list()):\n"
                "        values.append(base_instance['value'] + seed)\n"
                "        return dict(values=values)\n"
                "    return helper()\n")
        self.assertEqual(execute(code, "generate_snd", {"value": 3}, 7, timeout=10.)["result"],
                         {"values": [10]})

    def test_invalid_decorator_and_default_errors_are_runtime_failures(self):
        for code in ("@0\ndef solve_snd(instance):\n    return {}\n", candidate("1 / 0")):
            with self.subTest(code=code):
                validate_code(code, "solve_snd(instance)")
                with self.assertRaises(ExecutionError) as ctx:
                    execute(code, "solve_snd", {}, timeout=10.)
                self.assertEqual(ctx.exception.failure_type, "runtime")

    def test_default_and_decorator_execution_spend_line_budget(self):
        prefix = ("def solve_snd(instance):\n"
                  "    def spin():\n"
                  "        while True:\n"
                  "            pass\n")
        definitions = ("    def helper(value=spin()):\n        return value\n",
                       "    @spin()\n    def helper():\n        return {}\n")
        for definition in definitions:
            with self.subTest(definition=definition), self.assertRaises(ExecutionError) as ctx:
                execute(prefix+definition+"    return helper()\n", "solve_snd", {},
                        timeout=10., line_limit=100)
            self.assertEqual(ctx.exception.failure_type, "line_limit")

    def test_default_execution_keeps_wall_timeout(self):
        code = ("def solve_snd(instance):\n"
                "    def spin():\n"
                "        while True:\n"
                "            pass\n"
                "    def helper(value=spin()):\n"
                "        return value\n"
                "    return helper()\n")
        with self.assertRaises(ExecutionError) as ctx:
            execute(code, "solve_snd", {}, timeout=.5, line_limit=10**12)
        self.assertEqual(ctx.exception.failure_type, "timeout")

    def test_decorated_solver_defaults_still_require_feasible_deployment_result(self):
        data = {"nodes": [0, 1], "arcs": [{"id": "a", "tail": 0, "head": 1,
            "capacity": 3., "fixed_cost": 2., "variable_cost": 1.}],
            "commodities": [{"id": "k", "origin": 0, "destination": 1, "demand": 2.}]}
        template = ("@(lambda fn: (lambda instance: fn(instance)))\n"
                    "def solve_snd(instance):\n"
                    "    def route(flows=dict(k=dict(a=AMOUNT))):\n"
                    "        return dict(arc_open=dict(a=1), flows=flows)\n"
                    "    return route()\n")
        with TemporaryDirectory() as directory:
            source = Path(directory)/"candidate.py"
            source.write_text(template.replace("AMOUNT", "2."), encoding="utf-8")
            result = ProgramSNDSolver(source, timeout=10.).solve(data)
            self.assertTrue(result["constraint_satisfaction"]["feasible"])
            self.assertEqual(result["objective"], 4.)
            source.write_text(template.replace("AMOUNT", "0."), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "不可行方案"):
                ProgramSNDSolver(source, timeout=10.).solve(data)


if __name__ == "__main__":
    unittest.main()
