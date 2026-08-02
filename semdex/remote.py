"""Small dependency-free transport helpers for configured local HTTP models."""
from __future__ import annotations

import json
import mimetypes
import secrets
from collections.abc import Mapping
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


class RemoteRequestError(RuntimeError):
    """The configured service could not accept a request."""


class RemoteResponseError(RuntimeError):
    """The service returned a response Semdex cannot use."""


def _safe_filename(path: Path) -> str:
    # MIME headers are ASCII.  Keep an extension when possible and avoid a
    # filename-derived header injection if a user has an unusual local name.
    name = path.name.replace("\\", "_").replace('"', "_").replace("\r", "_").replace("\n", "_")
    return name.encode("ascii", "replace").decode("ascii") or "upload"


def _multipart_body(
    file_path: Path,
    fields: Mapping[str, str],
    *,
    file_field: str,
) -> tuple[str, bytes]:
    boundary = "----Semdex" + secrets.token_hex(16)
    chunks: list[bytes] = []

    def add(value: str | bytes) -> None:
        chunks.append(value.encode("utf-8") if isinstance(value, str) else value)

    for name, value in fields.items():
        add(f"--{boundary}\r\n")
        add(f'Content-Disposition: form-data; name="{name}"\r\n\r\n')
        add(str(value))
        add("\r\n")

    mime = mimetypes.guess_type(file_path.name)[0] or "application/octet-stream"
    add(f"--{boundary}\r\n")
    add(
        f'Content-Disposition: form-data; name="{file_field}"; '
        f'filename="{_safe_filename(file_path)}"\r\n'
    )
    add(f"Content-Type: {mime}\r\n\r\n")
    try:
        chunks.append(file_path.read_bytes())
    except OSError as e:
        raise RemoteRequestError(f"无法读取待上传文件 {file_path.name}: {e}") from e
    add("\r\n")
    add(f"--{boundary}--\r\n")
    return boundary, b"".join(chunks)


def _http_error_detail(error: HTTPError) -> str:
    try:
        body = error.read().decode("utf-8", errors="replace").strip()
    except Exception:
        body = ""
    detail = body[-1_000:] if body else str(error.reason or "未知错误")
    return f"HTTP {error.code}: {detail}"


def post_multipart_json(
    endpoint: str,
    *,
    file_path: Path,
    fields: Mapping[str, str] | None = None,
    api_key: str = "",
    timeout_sec: float = 180,
    label: str,
    file_field: str = "file",
) -> object:
    """Upload one file and return a JSON response object.

    This deliberately uses ``urllib`` rather than another runtime dependency;
    services implementing local OCR/ASR APIs only need standard multipart form
    handling.  Callers map connection failures to their own capability states.
    """
    target = endpoint.strip()
    if not target:
        raise RemoteRequestError(f"{label} 未配置 endpoint")
    try:
        timeout = max(1.0, float(timeout_sec))
    except (TypeError, ValueError) as e:
        raise RemoteRequestError(f"{label} timeout_sec 必须是正数") from e

    boundary, body = _multipart_body(file_path, fields or {}, file_field=file_field)
    headers = {
        "Accept": "application/json",
        "Content-Type": f"multipart/form-data; boundary={boundary}",
    }
    if api_key.strip():
        headers["Authorization"] = f"Bearer {api_key.strip()}"
    request = Request(target, data=body, headers=headers, method="POST")
    try:
        with urlopen(request, timeout=timeout) as response:
            raw = response.read()
    except HTTPError as e:
        raise RemoteRequestError(f"{label} 请求失败（{_http_error_detail(e)}）") from e
    except (URLError, OSError, TimeoutError, ValueError) as e:
        raise RemoteRequestError(f"{label} 服务不可用: {e}") from e

    try:
        return json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as e:
        preview = raw[:300].decode("utf-8", errors="replace").strip()
        suffix = f"（响应片段: {preview}）" if preview else ""
        raise RemoteResponseError(f"{label} 返回的不是 JSON{suffix}") from e
