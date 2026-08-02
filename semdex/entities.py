"""LLM-backed, schema-validated entity extraction for indexed files."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass

from .config import Config
from .db import Database
from .modelclient import ModelClient
from .models import ModelNotConfigured, ModelUnavailable

ENTITY_TYPES = {"person", "project", "organization", "date", "place", "tag"}

ENTITY_SYSTEM_PROMPT = """你是本地文件索引的实体抽取器。
只抽取文本中明确出现、对文件检索有帮助的实体：person、project、organization、date、place、tag。
不要猜测、不要补全缩写、不要把普通词堆成标签。只输出 JSON 数组，数组每项为
{"name": "实体原文", "type": "person|project|organization|date|place|tag", "context": "原文短片段"}。
没有可靠实体时输出 []。不要使用 Markdown 代码块或任何解释。"""


@dataclass
class EntityStats:
    indexed: int = 0
    failed: int = 0
    waiting_model: int = 0


def _parse_json(text: str) -> object:
    candidate = text.strip()
    if candidate.startswith("```"):
        candidate = re.sub(r"^```(?:json)?\s*|\s*```$", "", candidate, flags=re.IGNORECASE)
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        start = candidate.find("[")
        end = candidate.rfind("]")
        if start >= 0 and end > start:
            return json.loads(candidate[start:end + 1])
        raise


def extract_entities(text: str, client: ModelClient, config: Config) -> list[dict[str, str]]:
    prompt = text[:config.entities.max_chars]
    response = client.chat([
        {"role": "system", "content": ENTITY_SYSTEM_PROMPT},
        {"role": "user", "content": f"请从下面文件正文抽取实体：\n\n{prompt}"},
    ], temperature=0)
    data = _parse_json(response)
    if isinstance(data, dict):
        data = data.get("entities", [])
    if not isinstance(data, list):
        raise ValueError("实体模型返回的 JSON 不是数组")

    result: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for item in data:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name", "")).strip()
        entity_type = str(item.get("type", "")).strip().lower()
        context = str(item.get("context", "")).strip()
        key = (name.casefold(), entity_type)
        if not name or entity_type not in ENTITY_TYPES or key in seen:
            continue
        seen.add(key)
        result.append({"name": name[:200], "type": entity_type, "context": context[:500]})
        if len(result) >= config.entities.max_per_file:
            break
    return result


def index_entities(
    db: Database,
    config: Config,
    log=print,
    *,
    retry_failed: bool = False,
) -> EntityStats:
    """Fill pending entity relations without affecting the primary file index."""
    if not config.entities.enabled:
        return EntityStats()
    client = ModelClient(config.entities_model, "entities")
    statuses = ["pending", "waiting_model"]
    if retry_failed:
        statuses.append("failed")
    stats = EntityStats()
    for row in db.files_for_entities(statuses):
        file_id = int(row["id"])
        text = db.get_content(file_id)
        if not text:
            db.set_entity_status(file_id, "done")
            continue
        try:
            found = extract_entities(text, client, config)
        except (ModelNotConfigured, ModelUnavailable) as e:
            db.set_entity_status(file_id, "waiting_model", str(e))
            stats.waiting_model += 1
            log(f"⏳ 实体抽取 {row['filename']}: {e}")
            # A shared local model will fail identically for the remaining rows.
            break
        except Exception as e:
            db.set_entity_status(file_id, "failed", str(e))
            stats.failed += 1
            log(f"⚠ 实体抽取 {row['filename']} 失败: {e}")
            continue
        db.replace_entities(file_id, found)
        db.set_entity_status(file_id, "done")
        stats.indexed += 1
        log(f"✓ 实体抽取: {row['filename']}（{len(found)} 个）")
    return stats
