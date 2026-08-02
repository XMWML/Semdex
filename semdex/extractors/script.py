"""自定义脚本提取器：执行 `script <安全快照路径>`，stdout 即索引文本。

快照保留原文件名和扩展名，但不暴露原目录；退出码 0 表示成功，
非 0 时 stderr 作为错误信息记录。
这是最通用的扩展口子——任何格式只要能写个脚本转文字就能进索引。
"""
from __future__ import annotations

import subprocess
from pathlib import Path

from ..models import ExtractError
from .base import ExtractContext, Extractor

SCRIPT_TIMEOUT = 180


class ScriptExtractor(Extractor):
    name = "script"

    def __init__(self, script: str):
        self.script = script

    def extract(self, path: Path, ctx: ExtractContext) -> str:
        try:
            proc = subprocess.run(
                [self.script, str(path)],
                capture_output=True, text=True, errors="replace", timeout=SCRIPT_TIMEOUT,
            )
        except subprocess.TimeoutExpired:
            raise ExtractError(f"脚本超时（>{SCRIPT_TIMEOUT}s）: {self.script}")
        except OSError as e:
            raise ExtractError(f"脚本无法执行: {e}") from e
        if proc.returncode != 0:
            tail = (proc.stderr or "").strip()[-500:]
            raise ExtractError(f"脚本退出码 {proc.returncode}: {tail}")
        return proc.stdout
