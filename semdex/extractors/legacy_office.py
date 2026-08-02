"""Legacy Office extraction through an explicitly local LibreOffice conversion."""
from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path

from ..models import CapabilityUnavailable, ExtractError
from .base import ExtractContext, Extractor

CONVERTER_TIMEOUT = 180


class LegacyOfficeExtractor(Extractor):
    name = "legacy_office"
    exts = (".doc", ".xls", ".ppt")

    def _converter(self) -> str:
        for candidate in ("libreoffice", "soffice"):
            found = shutil.which(candidate)
            if found:
                return found
        raise CapabilityUnavailable(
            "legacy Office 文件需要 LibreOffice。安装 LibreOffice 后重新运行 `semdex index` 即可补索引"
        )

    def extract(self, path: Path, ctx: ExtractContext) -> str:
        suffix = path.suffix.lower()
        target_suffix = {".doc": ".docx", ".xls": ".xlsx", ".ppt": ".pptx"}[suffix]
        converter = self._converter()
        with tempfile.TemporaryDirectory(prefix="semdex-office-") as tmp:
            outdir = Path(tmp)
            try:
                proc = subprocess.run(
                    [converter, "--headless", "--convert-to", target_suffix[1:], "--outdir", str(outdir), str(path)],
                    capture_output=True,
                    text=True,
                    errors="replace",
                    timeout=CONVERTER_TIMEOUT,
                )
            except subprocess.TimeoutExpired as e:
                raise ExtractError(f"legacy Office 转换超时（>{CONVERTER_TIMEOUT}s）") from e
            except OSError as e:
                raise CapabilityUnavailable(f"无法启动 LibreOffice: {e}") from e
            if proc.returncode != 0:
                detail = (proc.stderr or proc.stdout or "未知错误").strip()[-800:]
                raise ExtractError(f"legacy Office 转换失败: {detail}")
            converted = outdir / f"{path.stem}{target_suffix}"
            if not converted.exists():
                raise ExtractError("LibreOffice 没有生成可读取的转换文件")
            from . import resolve  # local import avoids the registration cycle

            extractor = resolve(converted, ctx.config)
            if extractor is None or isinstance(extractor, LegacyOfficeExtractor):
                raise ExtractError("没有可用于转换后文件的提取器")
            return extractor.extract(converted, ctx)
