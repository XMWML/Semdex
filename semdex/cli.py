"""命令行入口。所有查询类命令支持 --json，输出稳定的结构化结果，
方便外部程序直接调用（退出码：0 成功，1 失败）。
"""
from __future__ import annotations

import argparse
import ipaddress
import json
import sys
import time

from . import __version__
from .config import load_config, resolve_config_path, write_default_config
from .db import Database
from .models import EmbeddingRebuildRequired, ModelNotConfigured, ModelUnavailable


def _print_json(obj) -> None:
    print(json.dumps(obj, ensure_ascii=False, indent=2))


def _fail(msg: str, as_json: bool) -> int:
    if as_json:
        _print_json({"ok": False, "error": msg})
    else:
        print(f"错误: {msg}", file=sys.stderr)
    return 1


def cmd_init(args) -> int:
    try:
        path = write_default_config(args.config)
    except FileExistsError as e:
        return _fail(str(e), False)
    print(f"已生成配置文件: {path}")
    print("下一步：编辑该文件，把要索引的文件夹填进 [watch] folders，然后运行 `semdex index`")
    return 0


def cmd_index(args) -> int:
    from .indexer import index_pending
    from .scanner import scan

    config = load_config(args.config)
    if not config.folders:
        return _fail("配置里还没有监控文件夹（[watch] folders 为空）", args.json)
    db = Database(config.db_path)
    log = (lambda *_: None) if args.json else print
    try:
        t0 = time.time()
        scan_stats = scan(db, config, log=log)
        index_stats = index_pending(db, config, log=log, retry_failed=args.retry_failed)
        elapsed = round(time.time() - t0, 2)
    finally:
        db.close()

    if args.json:
        _print_json({"ok": True, "scan": scan_stats.to_dict(),
                     "index": index_stats.to_dict(), "elapsed_sec": elapsed})
    else:
        print(f"\n扫描: 共 {scan_stats.scanned} 个文件，"
              f"新增/变化 {scan_stats.new_or_changed}，移除 {scan_stats.removed}，"
              f"超大跳过 {scan_stats.too_large}")
        print(f"索引: 完成 {index_stats.indexed}，失败 {index_stats.failed}，"
              f"等待模型 {index_stats.waiting_model}，无提取器 {index_stats.skipped}"
              f"（耗时 {elapsed}s）")
        if index_stats.waiting_capability:
            print(f"等待本地能力（OCR / ASR 等）: {index_stats.waiting_capability}")
        if index_stats.entities_indexed or index_stats.entity_failed:
            print(f"实体: 完成 {index_stats.entities_indexed}，失败 {index_stats.entity_failed}")
        if index_stats.waiting_model:
            print("提示: 启动 LM Studio 并在配置中启用对应模型后，重新运行 `semdex index` 即可补索引")
    return 0


def cmd_search(args) -> int:
    from .search import search

    config = load_config(args.config)
    db = Database(config.db_path)
    try:
        hits = search(db, config, args.query, mode=args.mode, limit=args.limit)
    except (ModelNotConfigured, ModelUnavailable, ValueError, RuntimeError) as e:
        db.close()
        return _fail(str(e), args.json)
    finally:
        if db.conn:
            try:
                db.close()
            except Exception:
                pass

    if args.json:
        _print_json({"ok": True, "query": args.query, "mode": args.mode,
                     "count": len(hits), "hits": [h.to_dict() for h in hits]})
    else:
        if not hits:
            print("没有找到匹配的文件")
            return 0
        for i, h in enumerate(hits, 1):
            print(f"{i}. {h.filename}  (score={h.score:.4f}, {h.source})")
            print(f"   {h.path}")
            if h.snippet:
                print(f"   {h.snippet}")
            print()
    return 0


def cmd_embed(args) -> int:
    from .indexer import embed_missing

    config = load_config(args.config)
    db = Database(config.db_path)
    try:
        n = embed_missing(db, config, log=print if not args.json else (lambda *_: None),
                          rebuild=args.rebuild)
    except (EmbeddingRebuildRequired, ModelNotConfigured, ModelUnavailable, ValueError) as e:
        return _fail(str(e), args.json)
    finally:
        db.close()
    if args.json:
        _print_json({"ok": True, "embedded_files": n})
    else:
        print(f"完成，共向量化 {n} 个文件")
    return 0


def cmd_status(args) -> int:
    config = load_config(args.config)
    db = Database(config.db_path)
    try:
        counts = db.counts()
        emb_model = db.meta_get("embedding_model")
        emb_rebuild_required = db.meta_get("embedding_rebuild_required") == "1"
    finally:
        db.close()
    info = {
        "ok": True,
        "config_path": str(config.config_path),
        "db_path": str(config.db_path),
        "folders": [str(p) for p in config.folders],
        "models": {
            "llm": config.llm.enabled,
            "vision": config.vision.enabled,
            "embedding": config.embedding.enabled,
        },
        "capabilities": {
            "ocr": config.ocr.enabled,
            "asr": config.asr.enabled,
            "entities": config.entities.enabled,
        },
        "embedding_model_in_db": emb_model,
        "embedding_rebuild_required": emb_rebuild_required,
        "files": counts,
    }
    if args.json:
        _print_json(info)
    else:
        print(f"配置: {info['config_path']}")
        print(f"数据库: {info['db_path']}")
        print(f"监控文件夹: {', '.join(info['folders']) or '（未配置）'}")
        m = info["models"]
        print(f"模型: llm={'开' if m['llm'] else '关'} "
              f"vision={'开' if m['vision'] else '关'} "
              f"embedding={'开' if m['embedding'] else '关'}")
        print(f"文件总数: {counts['total']}")
        for status, n in sorted(counts["by_status"].items()):
            print(f"  {status}: {n}")
        print(f"实体数: {counts['entities']}")
    return 0


def cmd_ask(args) -> int:
    from .agent import ask

    config = load_config(args.config)
    db = Database(config.db_path)
    try:
        result = ask(db, config, args.query)
    except (ModelNotConfigured, ModelUnavailable, ValueError, RuntimeError) as e:
        return _fail(str(e), args.json)
    finally:
        db.close()
    if args.json:
        _print_json({"ok": True, "query": args.query, **result.to_dict()})
    else:
        print(result.answer)
        for i, hit in enumerate(result.hits, 1):
            print(f"\n{i}. {hit.filename}\n   {hit.path}\n   {hit.snippet}")
    return 0


def cmd_entities(args) -> int:
    from .entities import index_entities

    config = load_config(args.config)
    if not config.entities.enabled:
        return _fail("实体抽取未启用（配置 [entities] enabled = true）", args.json)
    db = Database(config.db_path)
    try:
        stats = index_entities(db, config, log=print if not args.json else (lambda *_: None),
                               retry_failed=args.retry_failed)
    except (ModelNotConfigured, ModelUnavailable, ValueError, RuntimeError) as e:
        return _fail(str(e), args.json)
    finally:
        db.close()
    payload = {"indexed": stats.indexed, "failed": stats.failed, "waiting_model": stats.waiting_model}
    if args.json:
        _print_json({"ok": True, **payload})
    else:
        print(f"实体抽取: 完成 {stats.indexed}，失败 {stats.failed}，等待模型 {stats.waiting_model}")
    return 0


def cmd_watch(args) -> int:
    from .watcher import run_watcher

    config = load_config(args.config)
    try:
        run_watcher(config)
    except (RuntimeError, ValueError) as e:
        return _fail(str(e), False)
    return 0


def cmd_config(args) -> int:
    path = resolve_config_path(args.config)
    print(f"配置文件路径: {path}")
    if path.exists():
        print(path.read_text(encoding="utf-8"))
        return 0
    print("（文件不存在，运行 `semdex init` 生成）")
    return 1


def _is_loopback_host(host: str) -> bool:
    normalized = host.strip().lower()
    if normalized == "localhost":
        return True
    try:
        return ipaddress.ip_address(normalized).is_loopback
    except ValueError:
        return False


def cmd_serve(args) -> int:
    import threading
    import webbrowser

    import uvicorn

    from .web.app import create_app

    if not _is_loopback_host(args.host):
        return _fail(
            "Web 服务只允许绑定本机回环地址（127.0.0.1、::1 或 localhost），"
            "以保护文件内容和设置接口",
            False,
        )

    config_path = resolve_config_path(args.config)
    if not config_path.exists():
        try:
            write_default_config(args.config)
            print(f"已生成配置文件: {config_path}（可在网页设置中添加索引目录）")
        except FileExistsError:
            # Another local invocation created it between exists() and write.
            pass
    config = load_config(args.config)
    app = create_app(config)
    url = f"http://{args.host}:{args.port}"
    print(f"Semdex Web 界面: {url}  （Ctrl+C 退出）")
    if not args.no_open:
        threading.Timer(0.8, lambda: webbrowser.open(url)).start()
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="semdex", description="本地文件语义索引系统")
    p.add_argument("-c", "--config", help="配置文件路径（默认 ~/.semdex/config.toml，或环境变量 SEMDEX_CONFIG）")
    p.add_argument("-V", "--version", action="version", version=f"semdex {__version__}")
    sub = p.add_subparsers(dest="command", required=True)

    sp = sub.add_parser("init", help="生成默认配置文件")
    sp.set_defaults(func=cmd_init)

    sp = sub.add_parser("index", help="扫描并索引监控文件夹（增量）")
    sp.add_argument("--retry-failed", action="store_true", help="重试之前失败/跳过的文件")
    sp.add_argument("--json", action="store_true", help="输出 JSON")
    sp.set_defaults(func=cmd_index)

    sp = sub.add_parser("search", help="搜索文件内容")
    sp.add_argument("query", help="查询语句")
    sp.add_argument("--mode", default="hybrid", choices=["hybrid", "fulltext", "semantic"],
                    help="搜索模式（默认 hybrid，未启用 embedding 时自动退化为 fulltext）")
    sp.add_argument("--limit", type=int, default=20)
    sp.add_argument("--json", action="store_true", help="输出 JSON")
    sp.set_defaults(func=cmd_search)

    sp = sub.add_parser("ask", help="用本地 LLM 回答自然语言文件检索问题")
    sp.add_argument("query", help="例如：上个月和张三有关的 PDF 在哪")
    sp.add_argument("--json", action="store_true", help="输出 JSON")
    sp.set_defaults(func=cmd_ask)

    sp = sub.add_parser("embed", help="给缺向量的文件补 embedding")
    sp.add_argument("--rebuild", action="store_true", help="重建全部向量（换 embedding 模型后使用）")
    sp.add_argument("--json", action="store_true")
    sp.set_defaults(func=cmd_embed)

    sp = sub.add_parser("entities", help="为已索引文件补抽实体关系")
    sp.add_argument("--retry-failed", action="store_true", help="重试之前实体抽取失败的文件")
    sp.add_argument("--json", action="store_true", help="输出 JSON")
    sp.set_defaults(func=cmd_entities)

    sp = sub.add_parser("watch", help="实时监听文件变化并增量索引")
    sp.set_defaults(func=cmd_watch)

    sp = sub.add_parser("status", help="查看索引状态")
    sp.add_argument("--json", action="store_true")
    sp.set_defaults(func=cmd_status)

    sp = sub.add_parser("config", help="显示当前配置")
    sp.set_defaults(func=cmd_config)

    sp = sub.add_parser("serve", help="启动 Web 界面")
    sp.add_argument("--host", default="127.0.0.1", help="仅支持本机回环地址")
    sp.add_argument("--port", type=int, default=8787)
    sp.add_argument("--no-open", action="store_true", help="不自动打开浏览器")
    sp.set_defaults(func=cmd_serve)

    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except FileNotFoundError as e:
        as_json = getattr(args, "json", False)
        return _fail(str(e), as_json)
    except ValueError as e:
        as_json = getattr(args, "json", False)
        return _fail(str(e), as_json)
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    sys.exit(main())
