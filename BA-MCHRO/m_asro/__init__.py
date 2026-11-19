"""多主体 SND 共演化公共接口；延迟导入避免每个候选子进程加载 SciPy。"""
from __future__ import annotations

from m_asro._compat import install_aliases

install_aliases()

__all__ = ["MultiAgentSNDSolver", "MultiAgentResult", "UtilityConfig", "CCEMetaSolver", "OfflineMultiAgentClient", "LiteLLMClient", "EoHSConfig"]


def __getattr__(name: str):
    if name == "LiteLLMClient":
        from eoh.client import LiteLLMClient
        return LiteLLMClient
    if name in ("MultiAgentSNDSolver", "MultiAgentResult"):
        from m_asro.core.solver import MultiAgentSNDSolver, MultiAgentResult
        return {"MultiAgentSNDSolver": MultiAgentSNDSolver, "MultiAgentResult": MultiAgentResult}[name]
    if name == "UtilityConfig":
        from m_asro.core.utilities import UtilityConfig
        return UtilityConfig
    if name == "CCEMetaSolver":
        from m_asro.core.game import CCEMetaSolver
        return CCEMetaSolver
    if name == "OfflineMultiAgentClient":
        from snd.baselines import OfflineMultiAgentClient
        return OfflineMultiAgentClient
    if name == "EoHSConfig":
        from ba_eoh_s.eoh_s import EoHSConfig
        return EoHSConfig
    raise AttributeError(name)
