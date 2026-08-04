from __future__ import annotations

import json
from pathlib import Path

from semdex import agent, entities
from semdex.config import Config, EntityCfg, ModelCfg
from semdex.db import Database


def _indexed_file(db: Database, root: Path, name: str, text: str) -> int:
    path = root / name
    path.write_text(text, encoding="utf-8")
    file_id, _ = db.upsert_scan(str(path), name, path.suffix, path.stat().st_size, path.stat().st_mtime, name)
    db.save_content(file_id, text, name)
    db.set_status(file_id, "done")
    return file_id


def test_entity_indexing_persists_relations(tmp_path: Path, monkeypatch):
    root = tmp_path / "files"
    root.mkdir()
    cfg = Config(
        db_path=tmp_path / "index.db",
        llm=ModelCfg(enabled=True, model="fake"),
        entities=EntityCfg(enabled=True),
    )
    db = Database(cfg.db_path)
    file_id = _indexed_file(db, root, "meeting.txt", "张三负责广州地铁项目")

    class FakeClient:
        def __init__(self, *_args):
            pass

        def chat(self, *_args, **_kwargs):
            return '[{"name":"张三","type":"person","context":"张三负责"},{"name":"广州地铁","type":"project","context":"广州地铁项目"}]'

    monkeypatch.setattr(entities, "ModelClient", FakeClient)
    stats = entities.index_entities(db, cfg, log=lambda *_: None)
    assert stats.indexed == 1
    assert {item["name"] for item in db.entities_for_file(file_id)} == {"张三", "广州地铁"}
    assert db.files_by_entity("张三") == [file_id]
    db.close()


def test_agent_uses_only_tool_results(tmp_path: Path, monkeypatch):
    root = tmp_path / "files"
    root.mkdir()
    cfg = Config(db_path=tmp_path / "index.db", llm=ModelCfg(enabled=True, model="fake"))
    db = Database(cfg.db_path)
    file_id = _indexed_file(db, root, "meeting.txt", "张三讨论广州地铁项目")

    class FakeClient:
        def __init__(self, *_args):
            self.turn = 0

        def chat_with_tools(self, _messages, _tools, **_kwargs):
            self.turn += 1
            if self.turn == 1:
                return (
                    {"role": "assistant", "content": None, "tool_calls": []},
                    "",
                    [{"id": "call-1", "name": "search_fulltext", "arguments": '{"query":"张三"}'}],
                )
            return ({"role": "assistant", "content": "找到了会议记录。"}, "找到了会议记录。", [])

        def chat(self, *_args, **_kwargs):
            raise AssertionError("不应使用回退计划")

    monkeypatch.setattr(agent, "ModelClient", FakeClient)
    result = agent.ask(db, cfg, "张三的文件在哪")
    assert result.answer == "找到了会议记录。"
    assert [hit.file_id for hit in result.hits] == [file_id]
    assert result.steps[0]["tool"] == "search_fulltext"
    db.close()


def test_metadata_filter_can_narrow_an_existing_candidate_set(tmp_path: Path):
    root = tmp_path / "files"
    root.mkdir()
    cfg = Config(db_path=tmp_path / "index.db")
    db = Database(cfg.db_path)
    pdf_id = _indexed_file(db, root, "report.pdf", "张三项目报告")
    text_id = _indexed_file(db, root, "note.txt", "张三项目笔记")

    rows = db.filter_files(ext="pdf", file_ids=[text_id, pdf_id])
    assert [int(row["id"]) for row in rows] == [pdf_id]
    db.close()


def test_agent_structured_fallback_intersects_search_and_metadata(tmp_path: Path, monkeypatch):
    root = tmp_path / "files"
    root.mkdir()
    cfg = Config(db_path=tmp_path / "index.db", llm=ModelCfg(enabled=True, model="fake"))
    db = Database(cfg.db_path)
    pdf_id = _indexed_file(db, root, "report.pdf", "张三项目报告")
    text_id = _indexed_file(db, root, "note.txt", "张三项目笔记")

    class FakeClient:
        def __init__(self, *_args):
            pass

        def chat_with_tools(self, *_args, **_kwargs):
            return ({"role": "assistant", "content": ""}, "", [])

        def chat(self, *_args, **_kwargs):
            return '{"searches":[{"mode":"fulltext","query":"张三"}],"filters":[{"ext":"pdf"}],"entities":[]}'

    monkeypatch.setattr(agent, "ModelClient", FakeClient)
    result = agent.ask(db, cfg, "张三的 PDF")
    assert result.fallback_used
    assert [hit.file_id for hit in result.hits] == [pdf_id]
    assert result.steps[1]["arguments"]["file_ids"] == sorted([pdf_id, text_id])
    db.close()


def test_agent_native_entity_search_intersects_the_current_candidates(tmp_path: Path, monkeypatch):
    root = tmp_path / "files"
    root.mkdir()
    cfg = Config(
        db_path=tmp_path / "index.db",
        llm=ModelCfg(enabled=True, model="fake"),
        entities=EntityCfg(enabled=True),
    )
    db = Database(cfg.db_path)
    matched_id = _indexed_file(db, root, "report.txt", "quarterly launch report")
    _indexed_file(db, root, "notes.txt", "quarterly planning notes")
    outside_id = _indexed_file(db, root, "contact.txt", "customer contact record")
    db.replace_entities(matched_id, [{"name": "Alice", "type": "person"}])
    db.replace_entities(outside_id, [{"name": "Alice", "type": "person"}])

    class FakeClient:
        def __init__(self, *_args):
            self.turn = 0

        def chat_with_tools(self, _messages, _tools, **_kwargs):
            self.turn += 1
            if self.turn == 1:
                return (
                    {"role": "assistant", "content": None, "tool_calls": []},
                    "",
                    [{"id": "search", "name": "search_fulltext", "arguments": '{"query":"quarterly"}'}],
                )
            if self.turn == 2:
                return (
                    {"role": "assistant", "content": None, "tool_calls": []},
                    "",
                    [{"id": "entity", "name": "search_by_entity", "arguments": '{"name":"Alice"}'}],
                )
            return ({"role": "assistant", "content": "找到了季度报告。"}, "找到了季度报告。", [])

        def chat(self, *_args, **_kwargs):
            raise AssertionError("不应使用回退计划")

    monkeypatch.setattr(agent, "ModelClient", FakeClient)
    result = agent.ask(db, cfg, "Alice 的季度报告")

    assert [hit.file_id for hit in result.hits] == [matched_id]
    assert [step["result_count"] for step in result.steps] == [2, 1]
    db.close()


def test_agent_detail_requires_a_file_id_from_an_earlier_tool_result(tmp_path: Path, monkeypatch):
    root = tmp_path / "files"
    root.mkdir()
    cfg = Config(db_path=tmp_path / "index.db", llm=ModelCfg(enabled=True, model="fake"))
    db = Database(cfg.db_path)
    returned_id = _indexed_file(db, root, "visible.txt", "公开的项目会议记录")
    unreturned_id = _indexed_file(db, root, "private.txt", "绝密：不应通过详情工具泄露")

    class FakeClient:
        def __init__(self):
            self.turn = 0
            self.tool_payloads: list[dict] = []

        def chat_with_tools(self, messages, _tools, **_kwargs):
            self.turn += 1
            if self.turn == 1:
                return (
                    {"role": "assistant", "content": None, "tool_calls": []},
                    "",
                    [{"id": "search", "name": "search_fulltext", "arguments": '{"query":"公开"}'}],
                )
            if self.turn == 2:
                return (
                    {"role": "assistant", "content": None, "tool_calls": []},
                    "",
                    [
                        {
                            "id": "allowed-detail",
                            "name": "get_file_detail",
                            "arguments": json.dumps({"file_id": returned_id}),
                        },
                        {
                            "id": "blocked-detail",
                            "name": "get_file_detail",
                            "arguments": json.dumps({"file_id": unreturned_id}),
                        },
                    ],
                )
            self.tool_payloads = [
                json.loads(message["content"])
                for message in messages
                if message.get("role") == "tool"
            ]
            return ({"role": "assistant", "content": "已找到公开记录。"}, "已找到公开记录。", [])

        def chat(self, *_args, **_kwargs):
            raise AssertionError("不应使用回退计划")

    fake = FakeClient()
    monkeypatch.setattr(agent, "ModelClient", lambda *_args: fake)
    result = agent.ask(db, cfg, "找公开项目记录")

    assert [hit.file_id for hit in result.hits] == [returned_id]
    assert fake.tool_payloads[-2]["file"]["file_id"] == returned_id
    assert fake.tool_payloads[-2]["text_excerpt"] == "公开的项目会议记录"
    assert fake.tool_payloads[-1]["hits"] == []
    assert "此前的工具结果" in fake.tool_payloads[-1]["error"]
    assert "绝密" not in json.dumps(fake.tool_payloads[-1], ensure_ascii=False)
    db.close()


def test_agent_can_inspect_only_a_previously_returned_indexed_image(tmp_path: Path, monkeypatch):
    root = tmp_path / "files"
    root.mkdir()
    image_path = root / "receipt.png"
    image_path.write_bytes(b"\x89PNG\r\n\x1a\nplaceholder")
    cfg = Config(
        db_path=tmp_path / "index.db",
        folders=[root],
        llm=ModelCfg(enabled=True, model="fake"),
    )
    db = Database(cfg.db_path)
    image_id, _ = db.upsert_scan(
        str(image_path), image_path.name, image_path.suffix,
        image_path.stat().st_size, image_path.stat().st_mtime, image_path.name,
    )
    db.save_content(image_id, "票据图片", "ocr")
    db.set_status(image_id, "done")

    class FakeClient:
        def __init__(self):
            self.turn = 0
            self.tool_payloads: list[dict] = []
            self.inspected: list[tuple[Path, str, bytes]] = []

        def chat_with_tools(self, messages, _tools, **_kwargs):
            self.turn += 1
            if self.turn == 1:
                return (
                    {"role": "assistant", "content": None, "tool_calls": []},
                    "",
                    [{"id": "search", "name": "search_fulltext", "arguments": '{"query":"票据"}'}],
                )
            if self.turn == 2:
                return (
                    {"role": "assistant", "content": None, "tool_calls": []},
                    "",
                    [{
                        "id": "image",
                        "name": "inspect_image",
                        "arguments": json.dumps({"file_id": image_id, "question": "总金额是多少"}),
                    }],
                )
            self.tool_payloads = [
                json.loads(message["content"])
                for message in messages
                if message.get("role") == "tool"
            ]
            return ({"role": "assistant", "content": "总金额是 128 元。"}, "总金额是 128 元。", [])

        def describe_image(self, path, prompt):
            self.inspected.append((path, prompt, path.read_bytes()))
            return "票据日期 2026-08-04，总金额 128 元"

        def chat(self, *_args, **_kwargs):
            raise AssertionError("不应使用回退计划")

    fake = FakeClient()
    monkeypatch.setattr(agent, "ModelClient", lambda *_args: fake)
    result = agent.ask(db, cfg, "查看票据图片里的总金额")

    assert result.answer == "总金额是 128 元。"
    assert [step["tool"] for step in result.steps] == ["search_fulltext", "inspect_image"]
    assert fake.tool_payloads[-1]["image_description"].endswith("128 元")
    assert fake.inspected[0][0] != image_path.resolve()
    assert fake.inspected[0][0].suffix == ".png"
    assert fake.inspected[0][2] == image_path.read_bytes()
    assert not fake.inspected[0][0].exists()
    assert "总金额是多少" in fake.inspected[0][1]
    db.close()


def test_agent_rejects_an_image_reached_through_a_parent_symlink(tmp_path: Path):
    root = (tmp_path / "files").resolve()
    root.mkdir()
    original = root / "album"
    original.mkdir()
    image_path = original / "receipt.png"
    image_path.write_bytes(b"\x89PNG\r\n\x1a\noriginal")
    cfg = Config(db_path=tmp_path / "index.db", folders=[root])
    db = Database(cfg.db_path)
    image_id, _ = db.upsert_scan(
        str(image_path), image_path.name, image_path.suffix,
        image_path.stat().st_size, image_path.stat().st_mtime, image_path.name,
    )
    db.save_content(image_id, "票据图片", "ocr")
    db.set_status(image_id, "done")

    replacement = root / "replacement"
    replacement.mkdir()
    (replacement / image_path.name).write_bytes(b"\x89PNG\r\n\x1a\nreplacement")
    image_path.unlink()
    original.rmdir()
    original.symlink_to(replacement, target_is_directory=True)

    class FakeClient:
        def describe_image(self, *_args, **_kwargs):
            raise AssertionError("符号链接图片不应发送给模型")

    payload, hits = agent._execute_tool(
        db,
        cfg,
        "inspect_image",
        {"file_id": image_id},
        allowed_file_ids={image_id},
        model_client=FakeClient(),
    )

    assert hits == []
    assert "不安全" in payload["error"] or "符号链接" in payload["error"]
    db.close()


def test_agent_rejects_text_renamed_with_an_image_extension(tmp_path: Path):
    root = (tmp_path / "files").resolve()
    root.mkdir()
    image_path = root / "not-an-image.png"
    image_path.write_text("this is plain text", encoding="utf-8")
    cfg = Config(db_path=tmp_path / "index.db", folders=[root])
    db = Database(cfg.db_path)
    image_id, _ = db.upsert_scan(
        str(image_path), image_path.name, image_path.suffix,
        image_path.stat().st_size, image_path.stat().st_mtime, image_path.name,
    )
    db.save_content(image_id, "伪装图片", image_path.name)
    db.set_status(image_id, "done")

    class FakeClient:
        def describe_image(self, *_args, **_kwargs):
            raise AssertionError("伪装图片不应发送给模型")

    payload, hits = agent._execute_tool(
        db,
        cfg,
        "inspect_image",
        {"file_id": image_id},
        allowed_file_ids={image_id},
        model_client=FakeClient(),
    )

    assert hits == []
    assert "图片格式不匹配" in payload["error"]
    db.close()


def test_agent_fallback_ignores_entity_plan_when_entities_are_disabled(tmp_path: Path, monkeypatch):
    root = tmp_path / "files"
    root.mkdir()
    cfg = Config(db_path=tmp_path / "index.db", llm=ModelCfg(enabled=True, model="fake"))
    db = Database(cfg.db_path)
    file_id = _indexed_file(db, root, "meeting.txt", "公开项目会议记录")

    class FakeClient:
        def __init__(self, *_args):
            self.tool_names: list[str] = []

        def chat_with_tools(self, _messages, tools, **_kwargs):
            self.tool_names = [item["function"]["name"] for item in tools]
            raise agent.ModelUnavailable("tools unsupported")

        def chat(self, messages, **_kwargs):
            assert "entities 必须是空数组" in messages[0]["content"]
            return json.dumps({
                "searches": [{"mode": "fulltext", "query": "公开项目"}],
                "filters": [],
                "entities": ["不应执行"],
            })

    fake = FakeClient()
    monkeypatch.setattr(agent, "ModelClient", lambda *_args: fake)
    result = agent.ask(db, cfg, "找公开项目")

    assert "search_by_entity" not in fake.tool_names
    assert [hit.file_id for hit in result.hits] == [file_id]
    assert [step["tool"] for step in result.steps] == ["search_fulltext"]
    db.close()


def test_disabled_entity_tool_call_does_not_clear_existing_candidates(tmp_path: Path, monkeypatch):
    root = tmp_path / "files"
    root.mkdir()
    cfg = Config(db_path=tmp_path / "index.db", llm=ModelCfg(enabled=True, model="fake"))
    db = Database(cfg.db_path)
    file_id = _indexed_file(db, root, "meeting.txt", "公开项目会议记录")

    class FakeClient:
        def __init__(self, *_args):
            self.turn = 0
            self.entity_error = ""

        def chat_with_tools(self, messages, tools, **_kwargs):
            assert "search_by_entity" not in {
                item["function"]["name"] for item in tools
            }
            self.turn += 1
            if self.turn == 1:
                return (
                    {"role": "assistant", "content": None, "tool_calls": []},
                    "",
                    [{"id": "search", "name": "search_fulltext", "arguments": '{"query":"公开项目"}'}],
                )
            if self.turn == 2:
                return (
                    {"role": "assistant", "content": None, "tool_calls": []},
                    "",
                    [{"id": "entity", "name": "search_by_entity", "arguments": '{"name":"项目"}'}],
                )
            payloads = [
                json.loads(message["content"])
                for message in messages
                if message.get("role") == "tool"
            ]
            self.entity_error = payloads[-1]["error"]
            return ({"role": "assistant", "content": "保留关键词结果。"}, "保留关键词结果。", [])

        def chat(self, *_args, **_kwargs):
            raise AssertionError("不应使用回退计划")

    fake = FakeClient()
    monkeypatch.setattr(agent, "ModelClient", lambda *_args: fake)
    result = agent.ask(db, cfg, "找公开项目")

    assert [hit.file_id for hit in result.hits] == [file_id]
    assert "未启用" in fake.entity_error
    assert [step["result_count"] for step in result.steps] == [1, 0]
    db.close()


def test_agent_empty_metadata_filter_clears_previous_hits(tmp_path: Path, monkeypatch):
    root = tmp_path / "files"
    root.mkdir()
    cfg = Config(db_path=tmp_path / "index.db", llm=ModelCfg(enabled=True, model="fake"))
    db = Database(cfg.db_path)
    file_id = _indexed_file(db, root, "meeting.txt", "张三项目会议记录")
    unrelated_pdf = _indexed_file(db, root, "unrelated.pdf", "不相关的 PDF 文件")

    class FakeClient:
        def __init__(self, *_args):
            self.turn = 0

        def chat_with_tools(self, _messages, _tools, **_kwargs):
            self.turn += 1
            if self.turn == 1:
                return (
                    {"role": "assistant", "content": None, "tool_calls": []},
                    "",
                    [{"id": "search", "name": "search_fulltext", "arguments": '{"query":"张三"}'}],
                )
            if self.turn == 2:
                return (
                    {"role": "assistant", "content": None, "tool_calls": []},
                    "",
                    [{"id": "filter", "name": "filter_by_metadata", "arguments": '{"ext":"pdf"}'}],
                )
            return ({"role": "assistant", "content": "没有 PDF 文件。"}, "没有 PDF 文件。", [])

        def chat(self, *_args, **_kwargs):
            raise AssertionError("不应使用回退计划")

    monkeypatch.setattr(agent, "ModelClient", FakeClient)
    result = agent.ask(db, cfg, "张三的 PDF")

    assert result.answer == "没有 PDF 文件。"
    assert result.hits == []
    assert [step["result_count"] for step in result.steps] == [1, 0]
    assert result.steps[0]["tool"] == "search_fulltext"
    assert result.steps[1]["tool"] == "filter_by_metadata"
    assert file_id not in [hit.file_id for hit in result.hits]
    assert unrelated_pdf not in [hit.file_id for hit in result.hits]
    db.close()


def test_agent_cannot_use_a_guessed_id_to_expand_a_metadata_filter(tmp_path: Path, monkeypatch):
    root = tmp_path / "files"
    root.mkdir()
    cfg = Config(db_path=tmp_path / "index.db", llm=ModelCfg(enabled=True, model="fake"))
    db = Database(cfg.db_path)
    visible_id = _indexed_file(db, root, "visible.txt", "公开项目记录")
    private_id = _indexed_file(db, root, "private.txt", "绝密内容绝不能泄露")

    class FakeClient:
        def __init__(self, *_args):
            self.turn = 0
            self.tool_payloads: list[dict] = []

        def chat_with_tools(self, messages, _tools, **_kwargs):
            self.turn += 1
            if self.turn == 1:
                return (
                    {"role": "assistant", "content": None, "tool_calls": []},
                    "",
                    [{"id": "search", "name": "search_fulltext", "arguments": '{"query":"公开"}'}],
                )
            if self.turn == 2:
                return (
                    {"role": "assistant", "content": None, "tool_calls": []},
                    "",
                    [{
                        "id": "forged-filter",
                        "name": "filter_by_metadata",
                        "arguments": json.dumps({"file_ids": [private_id]}),
                    }],
                )
            if self.turn == 3:
                return (
                    {"role": "assistant", "content": None, "tool_calls": []},
                    "",
                    [{
                        "id": "forged-detail",
                        "name": "get_file_detail",
                        "arguments": json.dumps({"file_id": private_id}),
                    }],
                )
            self.tool_payloads = [
                json.loads(message["content"])
                for message in messages
                if message.get("role") == "tool"
            ]
            return ({"role": "assistant", "content": "没有匹配文件。"}, "没有匹配文件。", [])

        def chat(self, *_args, **_kwargs):
            raise AssertionError("不应使用回退计划")

    fake = FakeClient()
    monkeypatch.setattr(agent, "ModelClient", lambda *_args: fake)
    result = agent.ask(db, cfg, "找公开项目")

    assert visible_id not in [hit.file_id for hit in result.hits]
    assert private_id not in [hit.file_id for hit in result.hits]
    assert fake.tool_payloads[-2]["hits"] == []
    assert "此前工具返回的候选" in fake.tool_payloads[-2]["error"]
    assert fake.tool_payloads[-1]["hits"] == []
    assert "绝密" not in json.dumps(fake.tool_payloads, ensure_ascii=False)
    db.close()
