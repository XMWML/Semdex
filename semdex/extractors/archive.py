"""Safe, bounded text extraction from ZIP archives."""
from __future__ import annotations

import tempfile
import zipfile
from pathlib import Path

from ..models import (
    CapabilityNotConfigured, CapabilityUnavailable, ExtractError,
    ModelNotConfigured, ModelUnavailable,
)
from ..paths import ensure_private_directory
from .base import ExtractContext, Extractor

MAX_MEMBERS = 2_000
MAX_MEMBER_BYTES = 20 * 1024 * 1024
MAX_TOTAL_BYTES = 100 * 1024 * 1024
MAX_ARCHIVE_DEPTH = 3


class ZipExtractor(Extractor):
    name = "zip"
    exts = (".zip", ".cbz")

    def extract(self, path: Path, ctx: ExtractContext) -> str:
        budget = {"members": 0, "bytes": 0}
        return self._extract_archive(path, ctx, depth=0, budget=budget)

    def _extract_archive(
        self,
        path: Path,
        ctx: ExtractContext,
        *,
        depth: int,
        budget: dict[str, int],
    ) -> str:
        try:
            archive = zipfile.ZipFile(path)
        except (OSError, zipfile.BadZipFile) as e:
            raise ExtractError(f"ZIP 解析失败: {e}") from e

        parts: list[str] = []
        try:
            infos = [info for info in archive.infolist() if not info.is_dir()]
            with tempfile.TemporaryDirectory(
                prefix="semdex-zip-",
                dir=str(ensure_private_directory(ctx.config.temp_dir)),
            ) as tmp:
                temp_root = Path(tmp)
                for index, info in enumerate(infos, 1):
                    budget["members"] += 1
                    if budget["members"] > MAX_MEMBERS:
                        parts.append(f"[压缩包成员超过全局上限 {MAX_MEMBERS}，其余跳过]")
                        break
                    if info.flag_bits & 0x1:
                        parts.append(f"# {info.filename}\n（加密文件，跳过）")
                        continue
                    if (
                        info.file_size > MAX_MEMBER_BYTES
                        or budget["bytes"] + info.file_size > MAX_TOTAL_BYTES
                    ):
                        parts.append(f"# {info.filename}\n（超过压缩包安全大小限制，跳过）")
                        continue
                    # Never extract archive paths as supplied: use a fresh local
                    # filename, keeping only the suffix needed by the router.
                    suffix = Path(info.filename).suffix.lower()
                    member_path = temp_root / f"member-{depth}-{index}{suffix}"
                    try:
                        data = archive.read(info)
                    except (OSError, RuntimeError, zipfile.BadZipFile) as e:
                        parts.append(f"# {info.filename}\n（读取失败: {e}）")
                        continue
                    if budget["bytes"] + len(data) > MAX_TOTAL_BYTES:
                        parts.append(f"# {info.filename}\n（超过压缩包安全大小限制，跳过）")
                        continue
                    budget["bytes"] += len(data)
                    member_path.write_bytes(data)

                    from . import resolve  # local import avoids registration cycle

                    extractor = resolve(member_path, ctx.config)
                    if extractor is None:
                        parts.append(f"# {info.filename}\n（无适用提取器）")
                        continue
                    try:
                        if isinstance(extractor, ZipExtractor):
                            if depth >= MAX_ARCHIVE_DEPTH:
                                parts.append(
                                    f"# {info.filename}\n"
                                    f"（压缩包嵌套超过 {MAX_ARCHIVE_DEPTH} 层，跳过）"
                                )
                                continue
                            text = self._extract_archive(
                                member_path, ctx, depth=depth + 1, budget=budget
                            ).strip()
                        else:
                            text = extractor.extract(member_path, ctx).strip()
                    except (ModelNotConfigured, ModelUnavailable, CapabilityNotConfigured, CapabilityUnavailable):
                        # A member waiting for a model or local capability means
                        # the archive itself is not complete.  Let index_pending()
                        # record the outer archive as retryable instead of
                        # indexing a partial warning message as its content.
                        raise
                    except ExtractError as e:
                        parts.append(f"# {info.filename}\n（提取失败: {e}）")
                        continue
                    if text:
                        parts.append(f"# {info.filename}\n{text}")
        finally:
            archive.close()

        if not parts:
            raise ExtractError("压缩包中没有可提取的内容")
        return "\n\n".join(parts)
