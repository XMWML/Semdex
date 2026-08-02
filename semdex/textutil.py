"""CJK 文本处理：FTS5 按字索引变换、查询构建、摘要片段生成。

FTS5 的 unicode61 分词器不切分中文（相邻汉字会黏成一个 token），
这里的方案是：入库前在每个 CJK 字符两侧插空格（按字切分），
查询时把含 CJK 的词转成短语查询（"地 铁" 匹配相邻的 地铁），
两字词、专名等都能命中；英文 token 不受影响。
"""
from __future__ import annotations

# CJK 统一表意 + 扩展A + 兼容表意 + 日文假名 + 谚文音节
_CJK_RANGES = (
    (0x3040, 0x30FF),   # 平假名/片假名
    (0x3400, 0x4DBF),   # 扩展A
    (0x4E00, 0x9FFF),   # 基本区
    (0xAC00, 0xD7AF),   # 谚文
    (0xF900, 0xFAFF),   # 兼容表意
)


def is_cjk(ch: str) -> bool:
    cp = ord(ch)
    return any(lo <= cp <= hi for lo, hi in _CJK_RANGES)


def contains_cjk(s: str) -> bool:
    return any(is_cjk(c) for c in s)


def cjk_spaced(text: str) -> str:
    """在每个 CJK 字符两侧插入空格，使 unicode61 按字建 token。"""
    out: list[str] = []
    for ch in text:
        if is_cjk(ch):
            out.append(" ")
            out.append(ch)
            out.append(" ")
        else:
            out.append(ch)
    return "".join(out)


def build_fts_query(query: str) -> str:
    """把用户查询转成 FTS5 MATCH 表达式。

    - 含 CJK 的 token → 按字切分后的短语查询（保持相邻关系）
    - 纯英文/数字 token → 加引号防止与 FTS5 语法字符冲突
    - 多个 token 之间为隐式 AND
    """
    parts: list[str] = []
    for tok in query.split():
        tok = tok.replace('"', '""')
        if contains_cjk(tok):
            spaced = " ".join(cjk_spaced(tok).split())
            parts.append(f'"{spaced}"')
        else:
            parts.append(f'"{tok}"')
    return " ".join(parts)


def make_snippet(text: str, query: str, width: int = 160) -> str:
    """从原始正文中截取包含查询词的片段（展示用，避免 FTS 空格化文本外漏）。"""
    if not text:
        return ""
    flat = " ".join(text.split())
    lowered = flat.casefold()
    # 优先整个查询串，其次按长度从长到短尝试各 token
    candidates = [query.strip()] + sorted(query.split(), key=len, reverse=True)
    for term in candidates:
        term = term.strip()
        if not term:
            continue
        idx = lowered.find(term.casefold())
        if idx >= 0:
            start = max(0, idx - width // 3)
            end = min(len(flat), idx + len(term) + width)
            prefix = "…" if start > 0 else ""
            suffix = "…" if end < len(flat) else ""
            return f"{prefix}{flat[start:end]}{suffix}"
    # 找不到精确位置（如语义命中、多词分散），退化为开头片段
    if len(flat) <= width:
        return flat
    return flat[:width] + "…"
