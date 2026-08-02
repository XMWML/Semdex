"""Mail extractors for RFC 822 messages and mbox archives using the stdlib."""
from __future__ import annotations

import mailbox
from email import policy
from email.header import decode_header, make_header
from email.message import Message
from email.parser import BytesParser
from html.parser import HTMLParser
from pathlib import Path

from ..models import ExtractError
from .base import ExtractContext, Extractor

MAX_MBOX_MESSAGES = 2_000


class _HtmlText(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.parts: list[str] = []

    def handle_data(self, data: str) -> None:
        if data.strip():
            self.parts.append(data.strip())


def _header(message: Message, name: str) -> str:
    value = message.get(name, "")
    try:
        return str(make_header(decode_header(value)))
    except Exception:
        return str(value)


def _part_text(part: Message) -> str:
    try:
        content = part.get_content()
    except Exception:
        payload = part.get_payload(decode=True) or b""
        charset = part.get_content_charset() or "utf-8"
        content = payload.decode(charset, errors="replace")
    if not isinstance(content, str):
        return ""
    if part.get_content_type() == "text/html":
        parser = _HtmlText()
        parser.feed(content)
        return "\n".join(parser.parts)
    return content


def message_to_text(message: Message) -> str:
    lines = []
    for header, label in (("subject", "主题"), ("from", "发件人"), ("to", "收件人"), ("cc", "抄送"), ("date", "日期")):
        value = _header(message, header)
        if value:
            lines.append(f"{label}: {value}")

    plain_parts: list[str] = []
    html_parts: list[str] = []
    attachments: list[str] = []
    for part in message.walk():
        if part.is_multipart():
            continue
        filename = part.get_filename()
        if filename:
            attachments.append(_header(part, "content-disposition") or filename)
        if part.get_content_disposition() == "attachment":
            continue
        content_type = part.get_content_type()
        if content_type == "text/plain":
            plain_parts.append(_part_text(part))
        elif content_type == "text/html":
            html_parts.append(_part_text(part))

    body = "\n\n".join(part.strip() for part in (plain_parts or html_parts) if part.strip())
    if body:
        lines.extend(("", body))
    if attachments:
        lines.extend(("", "附件:", *attachments))
    return "\n".join(lines).strip()


class EmlExtractor(Extractor):
    name = "eml"
    exts = (".eml",)

    def extract(self, path: Path, ctx: ExtractContext) -> str:
        try:
            message = BytesParser(policy=policy.default).parsebytes(path.read_bytes())
        except OSError as e:
            raise ExtractError(f"邮件读取失败: {e}") from e
        except Exception as e:
            raise ExtractError(f"邮件解析失败: {e}") from e
        return message_to_text(message)


class MboxExtractor(Extractor):
    name = "mbox"
    exts = (".mbox",)

    def extract(self, path: Path, ctx: ExtractContext) -> str:
        try:
            box = mailbox.mbox(path, create=False)
            parts = []
            for index, message in enumerate(box, 1):
                if index > MAX_MBOX_MESSAGES:
                    parts.append(f"…（其余邮件省略，共超过 {MAX_MBOX_MESSAGES} 封）")
                    break
                text = message_to_text(message)
                if text:
                    parts.append(f"# 邮件 {index}\n{text}")
            return "\n\n".join(parts)
        except OSError as e:
            raise ExtractError(f"mbox 读取失败: {e}") from e
        except Exception as e:
            raise ExtractError(f"mbox 解析失败: {e}") from e
