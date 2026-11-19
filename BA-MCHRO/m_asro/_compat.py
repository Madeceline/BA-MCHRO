"""Import aliases for archived programs and external integrations."""
from __future__ import annotations

import importlib
import importlib.abc
import importlib.util
import sys
from pathlib import Path


MODULE_ALIASES = {
    "multi_agent_snd": "m_asro",
    "snd_generator": "snd.generator",
    "snd_solver": "snd.mip",
    "snd_visualization": "experiments.tools.visualization",
    "multi_agent_snd.solver": "m_asro.core.solver",
    "multi_agent_snd.game": "m_asro.core.game",
    "multi_agent_snd.utilities": "m_asro.core.utilities",
    "multi_agent_snd.interruption": "m_asro.core.interruption",
    "multi_agent_snd.execution": "m_asro.runtime.execution",
    "multi_agent_snd.performance": "m_asro.runtime.performance",
    "multi_agent_snd.progress": "m_asro.runtime.progress",
    "multi_agent_snd.llm_client": "m_asro.llm.client",
    "multi_agent_snd.replay": "m_asro.llm.replay",
    "multi_agent_snd.code_parser": "m_asro.llm.code_parser",
    "multi_agent_snd.operators": "m_asro.llm.operators",
    "multi_agent_snd.prompts": "m_asro.llm.prompts",
    "multi_agent_snd.storage": "m_asro.runtime.storage",
    "multi_agent_snd.diagnostics": "m_asro.runtime.diagnostics",
    "multi_agent_snd.evaluation": "snd.evaluation",
    "multi_agent_snd.constraints": "snd.constraints",
    "multi_agent_snd.reference": "snd.reference",
    "multi_agent_snd.baselines": "snd.baselines",
    "multi_agent_snd.benchmark_baselines": "snd.benchmark_baselines",
    "multi_agent_snd.benchmarks": "snd.benchmarks",
    "multi_agent_snd.deployment": "snd.deployment",
    "multi_agent_snd.eoh_s": "ba_eoh_s.eoh_s",
    "multi_agent_snd.clustering": "eoh.clustering",
    "multi_agent_snd.bayesian": "ba_eoh_s.bayesian",
    "snd_generator.instance": "snd.generator.instance",
    "snd_solver.solver": "snd.mip.solver",
    "snd_visualization.multi_agent": "experiments.tools.visualization.multi_agent",
    "m_asro.problems.snd": "snd",
    "m_asro.problems.snd.visualization": "experiments.tools.visualization",
    "m_asro.experiments": "experiments.tools",
    "m_asro.experiments.storage": "m_asro.runtime.storage",
    "m_asro.experiments.diagnostics": "m_asro.runtime.diagnostics",
    "m_asro.llm": "eoh",
    "ba_eoh_s.clustering": "eoh.clustering",
}


class _AliasLoader(importlib.abc.Loader):
    def __init__(self, target: str):
        self.target = target

    def create_module(self, spec):
        return importlib.import_module(self.target)

    def exec_module(self, module):
        pass


class _AliasFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "ba_eoh_s":
            folder = Path(__file__).resolve().parents[1] / "ba-eoh-s"
            if folder.is_dir():
                return importlib.util.spec_from_file_location(
                    fullname, folder / "__init__.py", submodule_search_locations=[str(folder)])
        canonical = MODULE_ALIASES.get(fullname)
        if canonical is None:
            for old in sorted(MODULE_ALIASES, key=len, reverse=True):
                if fullname.startswith(old + "."):
                    canonical = MODULE_ALIASES[old] + fullname[len(old):]
                    break
        if canonical is None:
            return None
        spec = importlib.util.find_spec(canonical)
        if spec is None:
            return None
        return importlib.util.spec_from_loader(fullname, _AliasLoader(canonical),
            is_package=spec.submodule_search_locations is not None)


def install_aliases() -> None:
    if not any(isinstance(finder, _AliasFinder) for finder in sys.meta_path):
        sys.meta_path.insert(0, _AliasFinder())
