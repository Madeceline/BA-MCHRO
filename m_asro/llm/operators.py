"""六种程序变异算子的 SND 专用提示词模板。

论文给出了统一的五段式 Prompt 结构、六种算子的语义，以及父代使用方式，
但没有公开六套可逐字复制的完整 Prompt。这里将这些公开定义具体化为静态多商品
SND 的可执行模板；模板中的 ``thought-code pair`` 与论文的程序策略表示一致。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence


# Figure 5 所示五段结构。保持为常量，便于测试和后续实验核对 Prompt。
PROMPT_SECTIONS = (
    "Task Description",
    "Operator-Specific Prompt",
    "Expected Output",
    "Note",
    "Parent Heuristics",
)


@dataclass(frozen=True, slots=True)
class OperatorPromptTemplate:
    """一种程序变异算子的元数据与提示词正文。"""

    code: str
    name: str
    parent_count: int
    prompt: str


INITIALIZATION_OPERATOR = "I1"
EVOLUTION_OPERATOR_CODES = ("E1", "E2", "M1", "M2", "M3")
ALL_OPERATOR_CODES = (INITIALIZATION_OPERATOR, *EVOLUTION_OPERATOR_CODES)
# EoH-S 的两种模因搜索算子：互补感知搜索（CS）与局部搜索（LS）。
EOH_S_OPERATOR_CODES = ("CS", "LS")
ALL_OPERATOR_CODES = (*ALL_OPERATOR_CODES, *EOH_S_OPERATOR_CODES)

# 提示词正文单独保存在 UTF-8 文本文件中，研究者可直接编辑而无需修改 Python。
OPERATOR_PROMPT_FILE = Path(__file__).with_name("operator_prompts.txt")

# 名称和父代数量属于算法控制元数据，仍由代码管理；自然语言正文从 txt 加载。
_OPERATOR_METADATA = {
    "I1": ("Initialization / 从零初始化", 0),
    "E1": ("Diverse Exploration / 多样性探索", 2),
    "E2": ("Shared-Idea Recombination / 共享思想重组", 2),
    "M1": ("Semantic Refinement / 语义精炼", 1),
    "M2": ("Parameterized Adjustment / 参数化调整", 1),
    "M3": ("Simplification and Pruning / 简化剪枝", 1),
    "CS": ("Complementary-aware Search / 互补感知搜索", 2),
    "LS": ("Local Search / 局部搜索", 1),
}


def load_operator_prompts(path: str | Path = OPERATOR_PROMPT_FILE) -> dict[str, str]:
    """读取 ``[I1]`` 等分节的 UTF-8 提示词，并校验六个算子完整且唯一。"""
    prompt_path = Path(path)
    try:
        text = prompt_path.read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"算子提示词文件不存在：{prompt_path}") from exc

    prompts: dict[str, str] = {}
    current_code: str | None = None
    current_lines: list[str] = []

    def save_current() -> None:
        """保存当前分节，同时拒绝空正文与重复分节。"""
        if current_code is None:
            return
        body = "\n".join(current_lines).strip()
        if not body:
            raise ValueError(f"{prompt_path} 中 [{current_code}] 的提示词为空")
        if current_code in prompts:
            raise ValueError(f"{prompt_path} 中 [{current_code}] 重复定义")
        prompts[current_code] = body

    for line_number, line in enumerate(text.splitlines(), start=1):
        stripped = line.strip()
        if stripped.startswith("[") and stripped.endswith("]"):
            save_current()
            current_code = stripped[1:-1].strip()
            if current_code not in ALL_OPERATOR_CODES:
                raise ValueError(
                    f"{prompt_path}:{line_number} 包含未知算子 [{current_code}]"
                )
            current_lines = []
        elif current_code is None:
            if stripped:
                raise ValueError(
                    f"{prompt_path}:{line_number} 的文本不属于任何算子分节"
                )
        else:
            current_lines.append(line)
    save_current()

    missing = set(ALL_OPERATOR_CODES) - prompts.keys()
    if missing:
        raise ValueError(f"{prompt_path} 缺少算子：{', '.join(sorted(missing))}")
    return prompts


_OPERATOR_PROMPTS = load_operator_prompts()

# Appendix B.1/B.2 中 I/E/M 算子的父代关系和从 txt 加载的 SND 提示词。
OPERATOR_TEMPLATES: dict[str, OperatorPromptTemplate] = {
    code: OperatorPromptTemplate(code, name, parent_count, _OPERATOR_PROMPTS[code])
    for code, (name, parent_count) in _OPERATOR_METADATA.items()
}


_SOLVER_EXPECTED_OUTPUT = """建议以 ``EXPLANATION:`` 开头，用中文说明启发式的核心思想、
构造步骤及可行性机制，再输出完整 Python 源码。说明可选，源码可直接输出或置于代码块中。
程序应提供可通过 ``solve_snd(instance)`` 调用的入口，允许导入、顶层辅助函数、类和可选参数。

返回字典必须包含 ``objective_value``、``arc_open``、``flows`` 和
``constraint_satisfaction``。``arc_open`` 以弧 id 为键；``flows`` 以商品 id 为第一层键、
弧 id 为第二层键。必须返回实际弧开启与流量方案，评估器会独立重算目标并检查约束。"""


_GENERATOR_EXPECTED_OUTPUT = """建议以 ``EXPLANATION:`` 开头，用中文说明实例扰动思想、
目标弱点及可行性保持机制，再输出完整 Python 源码。说明可选，源码可直接输出或置于代码块中。
程序应提供可通过 ``generate_snd(base_instance, seed)`` 调用的入口，
允许导入、顶层辅助函数、类和可选参数。

函数必须返回可直接交给 SND 求解端口的完整标准实例字典；不能返回评分、差异补丁或
解释文字。"""


_COMMON_NOTE = """源码只检查 Python 语法及行数、字符数和 AST 节点上限，不限制 import、
内置函数、装饰器、默认值、global、nonlocal、类、辅助函数或双下划线名称。
执行环境提供完整 Python 内置函数；导入的第三方库需在当前环境中已安装。
程序在独立子进程执行，仍受墙钟及 Python 执行行数预算约束，返回结果须通过独立校验。
请使相同输入的行为可复现；使用随机方法时显式固定种子，生成器使用传入的 seed。"""


def render_parents(operator: str, parents: Sequence[Any]) -> str:
    """渲染父代思想/源码对；保持原格式以兼容已有 prompt 日志重放。"""
    if operator == INITIALIZATION_OPERATOR:
        return "无父代。I1 必须完全从零生成候选。"
    rendered = []
    for index, parent in enumerate(parents, start=1):
        def field(name: str, default: str) -> str:
            return str(parent.get(name, default) if isinstance(parent, dict) else getattr(parent, name, default))
        rendered.append(
            f"### Thought-Code Pair {index}\n"
            f"ID: {field('id', f'parent_{index}')}\n"
            f"THOUGHT:\n{field('explanation', '')}\n"
            f"CODE:\n```python\n{field('code', '')}\n```"
        )
    return "\n\n".join(rendered)
