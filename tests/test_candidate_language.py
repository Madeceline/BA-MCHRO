"""Candidates use Python semantics; only syntax and source size gate dispatch."""
from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from m_asro import MultiAgentSNDSolver, UtilityConfig
from m_asro.problems.snd.baselines import BASE_SOLVER
from m_asro.llm.code_parser import CodeParser
from m_asro.problems.snd.deployment import ProgramSNDSolver
from m_asro.runtime.execution import ExecutionError, execute, validate_code
from m_asro.llm.operators import ALL_OPERATOR_CODES, OPERATOR_TEMPLATES
from m_asro.llm.prompts import build_prompt
from m_asro.experiments.storage import ProgramArchive
from m_asro.problems.snd.generator import SNDInstance


class RawCodeClient:
    """Offline generated program with imports/helpers, without explanation/fences."""

    def generate(self, prompt, temperature=1., n=1):
        code = ("import math\n"
                "def scale(value):\n    return math.sqrt(value)\n" +
                BASE_SOLVER.replace("1.0 * arc", "scale(0.25) * arc"))
        return [code] * n


class TestCandidateLanguage(unittest.TestCase):
    def test_optional_explanation_and_unfenced_source(self):
        parser = CodeParser()
        source = "def solve_snd(data, extra=2):\n    return data\n"
        for raw in (source, f"```python\n{source}```",
                    f"EXPLANATION:\n```python\n{source}```"):
            explanation, code = parser.parse(raw, "solve_snd(instance)")
            self.assertEqual(explanation, "")
            self.assertEqual(code, source.strip())
        explanation, code = parser.parse("EXPLANATION: 可选说明\n" + source, "unused")
        self.assertIn("可选说明", explanation)
        validate_code(code, "unused")

    def test_multiple_code_blocks_are_combined_in_order(self):
        raw = ("说明\n```python\nimport math\n```\n辅助代码\n"
               "```py\ndef helper(x):\n    return math.sqrt(x)\n```\n"
               "```python3\nsolve_snd = lambda data: helper(data['x'])\n```")
        _, code = CodeParser().parse(raw, "solve_snd(instance)")
        self.assertEqual(execute(code, "solve_snd", {"x": 9.})["result"], 3.)
        with self.assertRaisesRegex(ValueError, "源码超过行数"):
            validate_code(code, "unused", max_code_lines=2)

    def test_only_python_syntax_is_statically_rejected(self):
        for source in ("def f(:\n    pass", "return 1", "nonlocal unbound",
                       "break", "await f()"):
            with self.subTest(source=source):
                with self.assertRaises(SyntaxError):
                    validate_code(source, "unused")
                with self.assertRaises(SyntaxError):
                    CodeParser().parse(source, "unused")
        # Missing/wrong entry names and required extra parameters are runtime issues.
        for source in ("x = 1", "def other(x):\n    return x",
                       "def solve_snd(x, y):\n    return x + y"):
            validate_code(source, "solve_snd(instance)")
            with self.assertRaises(ExecutionError) as caught:
                execute(source, "solve_snd", {})
            self.assertEqual(caught.exception.failure_type, "runtime")

    def test_imports_builtins_classes_global_nonlocal_and_flexible_entry(self):
        source = '''from __future__ import annotations
import math
from dataclasses import dataclass
from collections import deque
counter = 0
@dataclass
class Box:
    value: int
def helper(value):
    return Box(value).__dict__['value']
def solve_snd(data, /, extra=2, *args, scale=3, **kwargs):
    global counter
    counter += 1
    total = 0
    def accumulate():
        nonlocal total
        total += helper(data['value'])
    accumulate()
    values = deque(map(abs, [-1, -2]))
    try:
        raise ValueError('handled')
    except Exception:
        pass
    return dict(total=total * scale + extra, counter=counter,
                reverse=list(reversed(values)), first=next(iter(values)),
                power=pow(2, 3), root=math.isqrt(16), name=__name__)
'''
        expected = dict(total=14, counter=1, reverse=[2, 1], first=1,
                        power=8, root=4, name="__candidate__")
        first = execute(source, "solve_snd", {"value": 4})
        second = execute(source, "solve_snd", {"value": 4})
        self.assertEqual(first["result"], expected)
        self.assertEqual(first["result"], second["result"])
        self.assertEqual(first["lines"], second["lines"])

    def test_io_and_dynamic_execution_have_normal_python_semantics(self):
        source = '''import pathlib
import sys
def solve_snd(instance):
    namespace = {}
    exec(compile('value = 2 + 3', '<dynamic>', 'exec'), namespace)
    value = eval('value * 2', namespace)
    with open(instance['path'], 'w', encoding='utf-8') as output:
        output.write(str(value))
    print('candidate output')
    return dict(value=int(pathlib.Path(instance['path']).read_text()),
                imported=__import__('math').isqrt(25),
                locals_ok='namespace' in locals(), globals_ok='sys' in globals())
'''
        with TemporaryDirectory() as directory:
            path = str(Path(directory) / "candidate.txt")
            result = execute(source, "solve_snd", {"path": path})
            self.assertEqual(result["result"], dict(value=10, imported=5,
                                                   locals_ok=True, globals_ok=True))

    def test_async_entry_is_executed_and_still_line_limited(self):
        code = ("import asyncio\nasync def solve_snd(data, *, extra=2):\n"
                "    await asyncio.sleep(0)\n    return data['x'] + extra\n")
        self.assertEqual(execute(code, "solve_snd", {"x": 3})["result"], 5)
        with self.assertRaises(ExecutionError) as caught:
            execute("async def solve_snd(data):\n    while True:\n        pass\n",
                    "solve_snd", {}, line_limit=100)
        self.assertEqual(caught.exception.failure_type, "line_limit")

    def test_all_operators_have_new_instruction_without_old_prohibitions(self):
        requirement = "禁止直接套用简单遍历算法与目前已有的成熟启发式算法"
        for operator in ALL_OPERATOR_CODES:
            parents = [dict(id=str(i), explanation="parent", code="")
                       for i in range(OPERATOR_TEMPLATES[operator].parent_count)]
            self.assertIn(requirement, OPERATOR_TEMPLATES[operator].prompt)
            for player in ("solver", "cost"):
                prompt = build_prompt(player, operator, parents, "", UtilityConfig(), {})
                self.assertIn(requirement, prompt)
                for old in ("禁止 import", "禁止 eval", "只能有一个", "仅输出一个",
                            "可用内置函数：abs/all"):
                    self.assertNotIn(old, prompt)

    def test_generated_unfenced_program_is_accepted_and_deployable(self):
        with TemporaryDirectory() as directory:
            solver = MultiAgentSNDSolver(SNDInstance(3, .7, .5, seed=7),
                generator_directions=("cost",), llm_client=RawCodeClient(),
                make_plots=False, progress_output=False)
            solver.root = Path(directory)
            solver.archive = ProgramArchive(solver.root)
            solver.user_prompt = "Python check"
            candidates = solver._generate("solver", "I1", [], 1, 0,
                                          {player: [] for player in solver.players},
                                          {"marginals": [[] for _ in solver.players]})
            self.assertEqual(len(candidates), 1)
            program = candidates[0]
            self.assertEqual(program.explanation, "")
            self.assertIsNone(program.error)
            self.assertFalse(solver._is_warm_start(program))
            path = solver.root / "candidate.py"
            path.write_text(program.code, encoding="utf-8")
            deployed = ProgramSNDSolver(path).solve(solver.base)
            self.assertTrue(deployed["constraint_satisfaction"]["feasible"])


if __name__ == "__main__":
    unittest.main()
