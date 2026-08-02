"""OCR adapters for images and scanned PDFs.

Tesseract remains optional and dependency-free; a configured local HTTP OCR
service can be used instead.  Missing local capabilities stay actionable and
recoverable in the indexing queue.
"""
from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path

from .config import OcrCfg
from .models import CapabilityNotConfigured, CapabilityUnavailable, ExtractError
from .remote import RemoteRequestError, RemoteResponseError, post_multipart_json

OCR_TIMEOUT_SECONDS = 180


def _find_command(command: str, label: str) -> str:
    expanded = Path(command).expanduser()
    resolved = str(expanded) if expanded.parent != Path(".") and expanded.is_file() else shutil.which(command)
    if not resolved:
        raise CapabilityUnavailable(
            f"{label} 不可用: 找不到 `{command}`。安装后重新运行 `semdex index` 即可补索引"
        )
    return resolved


def _require_ocr(cfg: OcrCfg) -> None:
    if not cfg.enabled:
        raise CapabilityNotConfigured(
            "OCR 未启用（在配置中设置 [ocr] enabled = true，并选择 tesseract 或 local_http）"
        )
    if cfg.provider not in {"tesseract", "local_http"}:
        raise CapabilityNotConfigured(f"不支持的 OCR provider: {cfg.provider}")


def _run(args: list[str], label: str, timeout_sec: int = OCR_TIMEOUT_SECONDS) -> str:
    try:
        proc = subprocess.run(
            args,
            capture_output=True,
            text=True,
            errors="replace",
            timeout=timeout_sec,
        )
    except subprocess.TimeoutExpired as e:
        raise ExtractError(f"{label} 超时（>{timeout_sec}s）") from e
    except OSError as e:
        raise CapabilityUnavailable(f"无法启动 {label}: {e}") from e
    if proc.returncode != 0:
        detail = (proc.stderr or proc.stdout or "未知错误").strip()[-800:]
        raise ExtractError(f"{label} 失败: {detail}")
    return proc.stdout


def _response_text(payload: object, response_path: str) -> str:
    value = payload
    for part in response_path.split("."):
        if isinstance(value, dict):
            if part not in value:
                raise ExtractError(f"OCR 响应中找不到 response_path `{response_path}`")
            value = value[part]
        elif isinstance(value, list):
            try:
                value = value[int(part)]
            except (ValueError, IndexError) as e:
                raise ExtractError(f"OCR 响应中找不到 response_path `{response_path}`") from e
        else:
            raise ExtractError(f"OCR 响应中找不到 response_path `{response_path}`")
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return str(value)
    raise ExtractError(f"OCR response_path `{response_path}` 的值不是文本")


def _ocr_local_http(path: Path, cfg: OcrCfg) -> str:
    try:
        payload = post_multipart_json(
            cfg.endpoint,
            file_path=path,
            fields={"languages": cfg.languages},
            api_key=cfg.api_key,
            timeout_sec=cfg.timeout_sec,
            label="本地 OCR 服务",
        )
    except RemoteRequestError as e:
        raise CapabilityUnavailable(str(e)) from e
    except RemoteResponseError as e:
        raise ExtractError(str(e)) from e
    return _response_text(payload, cfg.response_path)


def ocr_image(path: Path, cfg: OcrCfg) -> str:
    """Return text from one image using the configured OCR provider."""
    _require_ocr(cfg)
    if cfg.provider == "tesseract":
        tesseract = _find_command(cfg.command, "OCR")
        return _run(
            [tesseract, str(path), "stdout", "-l", cfg.languages],
            "图片 OCR",
            cfg.timeout_sec,
        ).strip()
    if cfg.provider == "local_http":
        return _ocr_local_http(path, cfg)
    # Keep this defensive branch for Config objects created directly in Python.
    raise CapabilityNotConfigured(f"不支持的 OCR provider: {cfg.provider}")


def ocr_pdf(path: Path, cfg: OcrCfg) -> str:
    """Render a scanned PDF page-by-page and run OCR on the generated images."""
    _require_ocr(cfg)
    renderer = _find_command(cfg.pdf_renderer, "PDF 渲染器")
    with tempfile.TemporaryDirectory(prefix="semdex-ocr-") as tmp:
        prefix = Path(tmp) / "page"
        _run(
            [renderer, "-r", str(cfg.dpi), "-png", str(path), str(prefix)],
            "扫描 PDF 渲染",
            cfg.timeout_sec,
        )
        pages = sorted(Path(tmp).glob("page-*.png"))
        if not pages:
            raise ExtractError("扫描 PDF 渲染后没有得到页面图像")
        texts = [ocr_image(page, cfg) for page in pages]
    return "\n\n".join(text for text in texts if text).strip()
