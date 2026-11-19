"""提取候选源码与可选原理说明，不限制 Python 语法结构或可用名称。"""
from __future__ import annotations

import re


class CodeParser:
    """兼容带说明的代码块、多个代码块和直接返回的 Python 源码。"""

    def parse(self, raw_output: str, signature: str) -> tuple[str, str]:
        """返回 ``(explanation, code)``；保留 signature 参数供旧调用方使用。

        入口是否可调用及返回结果是否正确，由真实执行和结果校验判断。
        compile 还会检查 ast.parse 不拒绝的顶层 return、无绑定 nonlocal 等。
        """
        explanation = self._extract_explanation(raw_output)
        code = self._extract_code(raw_output)
        compile(code, "<candidate>", "exec", dont_inherit=True)
        return explanation, code

    @staticmethod
    def _extract_explanation(raw_output: str) -> str:
        prefix = raw_output.split("```", 1)[0]
        match = re.search(r"(?:EXPLANATION|原理说明)\s*[:：]\s*(.*)", prefix,
                          flags=re.IGNORECASE | re.DOTALL)
        if not match:
            return ""
        explanation = match.group(1).strip()
        return explanation if "```" in raw_output else explanation.split("\n", 1)[0]

    @staticmethod
    def _extract_code(raw_output: str) -> str:
        # 按顺序合并代码块，允许 import、辅助函数、类与入口分别放置。
        # 优先 Python/无语言标签的代码块，避免把说明性 Markdown 当成源码。
        blocks = re.findall(r"```([^\n`]*)\n(.*?)```", raw_output, flags=re.DOTALL)
        if blocks:
            python_blocks = [body.strip() for language, body in blocks
                             if language.strip().lower() in {"", "python", "py", "python3"}]
            selected = python_blocks if python_blocks else [body.strip() for _, body in blocks]
            return "\n\n".join(block for block in selected if block)
        # 没有代码围栏时接受源码；显式说明标题保留为 Python 注释。
        # 其余内容交给编译器，不猜测、修复或丢弃无效代码。
        return re.sub(r"\A[ \t]*(EXPLANATION|原理说明)[ \t]*[:：]([^\n]*)", r"# \1:\2",
                      raw_output, count=1, flags=re.IGNORECASE).strip()
