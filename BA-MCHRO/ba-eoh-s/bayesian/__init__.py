"""观测管理、共轭性能模型、主动评估与可选 Solver Oracle。

这些组件可独立用于离线分析；默认不会改变 ``SNDEvaluator``、CPM 或 M-PSRO
的执行路径。Oracle 必须显式启用。真实执行结果可通过 :func:`observation_from_evaluation` 转换后，
再交给 :class:`BayesianPerformanceModel`。
"""

from ba_eoh_s.bayesian.features import FEATURE_NAMES, FEATURE_SCHEMA_VERSION, instance_features
from ba_eoh_s.bayesian.observations import (
    EvaluationObservation,
    ObservationStore,
    observation_from_evaluation,
)
from ba_eoh_s.bayesian.linear_model import BayesianPerformanceModel
from ba_eoh_s.bayesian.posterior import JointLossSamples, posterior_summary
from ba_eoh_s.bayesian.bayesian_cpm import BayesianCPMConfig, bayesian_cpm_select, select_portfolio
from ba_eoh_s.bayesian.active_evaluation import (ActiveEvaluationConfig, ActiveInstance, BayesianActiveEvaluator,
                                EvaluationBackend, EvaluationOutcome, SNDEvaluationBackend)
from ba_eoh_s.bayesian.oracle import BayesianOracleConfig, OracleBudgetError

__all__ = [
    "FEATURE_NAMES",
    "FEATURE_SCHEMA_VERSION",
    "instance_features",
    "EvaluationObservation",
    "ObservationStore",
    "observation_from_evaluation",
    "BayesianPerformanceModel",
    "JointLossSamples",
    "posterior_summary",
    "BayesianCPMConfig",
    "bayesian_cpm_select",
    "select_portfolio",
    "ActiveEvaluationConfig",
    "ActiveInstance",
    "BayesianActiveEvaluator",
    "EvaluationBackend",
    "EvaluationOutcome",
    "SNDEvaluationBackend",
    "BayesianOracleConfig",
    "OracleBudgetError",
]
