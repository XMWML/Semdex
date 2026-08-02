"""OpenAI 兼容接口的薄封装（LM Studio / Ollama / 云端均可）。

各模型用途（agent / entities / fallback / vision / embedding）可独立配置。未启用抛
ModelNotConfigured，服务连不上抛 ModelUnavailable——索引管线据此
把文件标记为 waiting_model，模型就绪后重跑即可补齐，不算失败。
"""
from __future__ import annotations

import base64
from pathlib import Path

from .config import ModelCfg
from .models import ModelNotConfigured, ModelUnavailable

_IMAGE_MIME = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".webp": "image/webp",
    ".gif": "image/gif",
    ".bmp": "image/bmp",
}

VISION_PROMPT = (
    "请详细描述这张图片的内容，用于建立文件搜索索引。"
    "如果图片中包含文字（截图、文档照片、标识等），请完整转写出所有文字。"
    "直接输出描述内容，不要任何开场白。"
)


def embedding_identity(cfg: ModelCfg) -> str:
    """Return the stable identity of the embedding vector space.

    A display model name is not enough: two compatible endpoints can expose
    different models under the same name.
    """
    return f"{cfg.base_url.rstrip('/')}\n{cfg.model}"


class ModelClient:
    def __init__(self, cfg: ModelCfg, kind: str):
        self.cfg = cfg
        self.kind = kind  # 用于选择配置节名称并生成可定位的报错信息
        self._client = None

    @property
    def enabled(self) -> bool:
        return self.cfg.enabled

    def _get_client(self):
        if not self.cfg.enabled:
            raise ModelNotConfigured(f"{self.kind} 模型未启用（配置 [models.{self.kind}] enabled = true）")
        if self._client is None:
            from openai import OpenAI
            self._client = OpenAI(
                base_url=self.cfg.base_url, api_key=self.cfg.api_key, timeout=180, max_retries=1
            )
        return self._client

    def _wrap_connection_error(self, e: Exception) -> Exception:
        from openai import APIConnectionError, APIStatusError, APITimeoutError
        if isinstance(e, (APIConnectionError, APITimeoutError, APIStatusError)):
            status = getattr(e, "status_code", None)
            suffix = f"（HTTP {status}）" if status else ""
            return ModelUnavailable(
                f"无法使用 {self.kind} 模型服务 {self.cfg.base_url}{suffix}——"
                "LM Studio/Ollama 启动了吗？模型加载了吗？"
            )
        return e

    def chat(self, messages: list[dict], **kwargs) -> str:
        client = self._get_client()
        try:
            resp = client.chat.completions.create(model=self.cfg.model, messages=messages, **kwargs)
        except Exception as e:
            raise self._wrap_connection_error(e) from e
        return resp.choices[0].message.content or ""

    def chat_with_tools(self, messages: list[dict], tools: list[dict], **kwargs) -> tuple[dict, str, list[dict]]:
        """Run one OpenAI-compatible tool-calling turn.

        The returned assistant message is already JSON-serializable and can be
        appended to the next request verbatim.  This keeps the agent independent
        of OpenAI SDK response classes and friendly to local compatible servers.
        """
        client = self._get_client()
        try:
            resp = client.chat.completions.create(
                model=self.cfg.model,
                messages=messages,
                tools=tools,
                tool_choice="auto",
                **kwargs,
            )
        except Exception as e:
            raise self._wrap_connection_error(e) from e
        message = resp.choices[0].message
        raw = message.model_dump(exclude_none=True)
        calls = []
        for call in message.tool_calls or []:
            calls.append({
                "id": call.id,
                "name": call.function.name,
                "arguments": call.function.arguments,
            })
        return raw, message.content or "", calls

    def describe_image(self, path: Path) -> str:
        mime = _IMAGE_MIME.get(path.suffix.lower(), "image/png")
        b64 = base64.b64encode(path.read_bytes()).decode("ascii")
        return self.chat([
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": VISION_PROMPT},
                    {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}},
                ],
            }
        ])

    def embed(self, texts: list[str]) -> list[list[float]]:
        client = self._get_client()
        try:
            resp = client.embeddings.create(model=self.cfg.model, input=texts)
        except Exception as e:
            raise self._wrap_connection_error(e) from e
        # 按 index 排序，防止服务端乱序返回
        data = sorted(resp.data, key=lambda d: d.index)
        return [d.embedding for d in data]
