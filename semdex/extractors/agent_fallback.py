"""Conservative LLM fallback for unknown, decodable text formats.

It intentionally has no shell or filesystem tools: it can only inspect the
already selected file's bounded bytes, which avoids turning indexing into an
arbitrary command execution surface.  Binary formats still require a dedicated
extractor or a user-configured script rule.
"""
from __future__ import annotations

from pathlib import Path

from ..models import ExtractError
from .base import ExtractContext, Extractor
from .text import decode_text_best_effort

PROMPT = """下面是一份未知文本格式文件的内容。请给出便于文件检索的简短中文摘要，
说明它看起来是什么、包含哪些主题、人物、项目或日期。不要臆测未出现的信息。"""


class AgentFallbackExtractor(Extractor):
    name = "agent_fallback"

    def extract(self, path: Path, ctx: ExtractContext) -> str:
        try:
            with path.open("rb") as source:
                raw = source.read(ctx.config.agent_fallback.max_bytes)
        except OSError as e:
            raise ExtractError(f"读取未知格式文件失败: {e}") from e
        if b"\0" in raw:
            raise ExtractError("未知格式看起来是二进制文件；请配置专用提取器或自定义脚本")
        text = decode_text_best_effort(raw).strip()
        if not text:
            raise ExtractError("未知文本格式中没有可索引的内容")
        summary = ctx.llm.chat([
            {"role": "system", "content": PROMPT},
            {"role": "user", "content": f"文件名: {path.name}\n\n{text}"},
        ], temperature=0)
        return f"[未知文本格式原文]\n{text}\n\n[本地 LLM 摘要]\n{summary.strip()}"
