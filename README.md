# Semdex

**给你的文件系统装上语义记忆。**

Semdex（**Sem**antic in**dex**）是一个本地文件语义索引系统：监控你指定的文件夹，把各类文件的**内容**——而不只是文件名——提取成文本和向量存进本地 SQLite，然后用关键词、语义或自然语言把文件找回来。默认配置模板把 OpenAI 兼容接口指向回环地址上的本地 LM Studio / Ollama；若你显式将 `base_url` 改为远程 OpenAI 兼容服务，供模型处理的文件内容会发送给该服务。

> 详细设计讨论见同目录《设计方案.md》。
>
> 从安装、网页设置到本地 OCR/Whisper 服务接入，见《[使用说明.md](使用说明.md)》。

---

## 缘起：微软的 WinFS，一个早了二十年的梦

2003 年 PDC 大会上，比尔·盖茨亲自演示了下一代 Windows "Longhorn" 的三大支柱之一 —— **WinFS（Windows Future Storage）**。它的野心是彻底改造文件系统：

- 文件不再是躺在目录树里的字节流，而是存进关系数据库（基于 SQL Server 引擎）的**结构化对象（Item）**；
- 文档、邮件、联系人、照片共享统一的 Schema，彼此之间有**关系**——"找出和这个人相关的所有东西"是一句查询就能回答的问题；
- 检索不再靠记住路径和文件名，而是靠内容和关联。

然后它失败了。2004 年被从 Longhorn 中砍掉，2006 年 6 月项目正式取消，技术资产拆散回流到 SQL Server 和 Entity Framework 里。失败的原因很多——把整个文件系统架在数据库上的性能代价、和现有软件生态的兼容性、范围失控——但有一个根本性的缺环：**当年没有任何技术能把非结构化内容（图片、扫描件、随手写的文档）自动变成可查询的结构化语义**。Schema 设计得再精美，没人手工填元数据，数据库就是空的。

二十年后，这个缺环被本地大模型补上了：视觉模型能"看懂"图片，embedding 能把模糊的语义变成可计算的向量，LLM 能读懂任何格式的残骸。Semdex 不复刻 WinFS 激进的"接管文件系统"路线——文件还是你的文件，待在原地不动——而是做一层**影子索引**：数据库旁路记录每个文件的内容语义，检索时再指回原文件。WinFS 想要的结果，用一条务实得多的路径实现。

---

## 已实现（当前版本）

- ✅ **多文件夹增量索引**：sha256 内容判重两级加速（size+mtime 未变直接跳过），文件改动/删除/新增自动同步，反复运行只处理变化部分
- ✅ **分层内容提取**，按文件类型路由，全部可扩展：
  - 文本/代码/Markdown/CSV 等 40+ 扩展名直接读取（utf-8 / gb18030 自动识别）
  - PDF 文本层提取；扫描 PDF 可回退到已配置的 OCR（本机 Tesseract 或本地 HTTP 服务）
  - docx / xlsx / pptx 解析（含表格、按工作表/页组织）
  - 图片先走确定性 OCR，再按需补充本地视觉模型描述
  - 邮件（`.eml` / `.mbox`）、ZIP/CBZ 压缩包递归（最多 3 层，受成员数和解压总量限制）、legacy Office（通过本机 LibreOffice 转换）
  - 音频/视频通过可选的 faster-whisper 或 OpenAI 兼容 Whisper 服务转写；未知但可读为文本的格式可走受限 LLM 兜底
  - **自定义脚本提取器**：配置一条规则，任意格式 `脚本 <安全快照文件>` → stdout 即索引文本
- ✅ **三种检索模式**：
  - 关键词（FTS5 BM25，中文按字切分方案，两字词精确命中）
  - 语义（本地 embedding + 余弦相似度）
  - 混合（RRF 融合，默认模式）
- ✅ **按用途配置模型**：检索 Agent、实体抽取、未知文本兜底、视觉理解和 embedding 可分别选择 OpenAI 兼容服务与模型；模型没启动时优雅降级（图片进 `waiting_model` 队列等待补索引，语义搜索退化为关键词）
- ✅ **自然语言 Agent 搜索**：本地 LLM 只能调用受限的全文、语义、元数据、实体和文件详情工具；不支持原生工具调用的服务会退化为结构化检索计划
- ✅ **实体与关系**：可选 LLM 为已索引文件抽取人名、项目、机构、日期、地点、标签，支持按实体反查文件
- ✅ **实时文件监听**：`semdex watch` 通过 macOS FSEvents / Linux inotify / Windows 原生观察器触发防抖增量索引；默认每日全量对账并重试失败项，可配置或关闭
- ✅ **CLI**（`--json` 结构化输出 + 稳定退出码，方便外部程序/脚本接入）
- ✅ **跨平台原生桌面界面与 WebUI**：两套界面均支持即输即搜、问答、全文查看、打开/定位文件、重新索引/完整重建，以及完整的目录、模型、OCR、ASR 与功能设置；桌面界面在 macOS / Windows / Linux 使用各自文件管理器
- ✅ **REST API**：Web 界面用的接口全部开放，可被其他程序直接调用

## 当前边界

- OCR 可使用本机 Tesseract，也可对接满足 multipart/JSON 协议的本地 HTTP 服务（例如用 PaddleOCR 包一层本地接口）；扫描 PDF 仍需 Poppler 的 `pdftoppm`。
- ASR 可通过 `python3 "Start Semdex.py" --with-asr --sync-only` 安装 faster-whisper，也可对接 OpenAI 兼容的本地 Whisper 转写接口。未安装或服务不在线时文件会进入 `waiting_capability`，恢复后重跑即可。
- legacy Office 需要本机 LibreOffice；未知二进制格式不会执行任意命令，建议配置专用脚本提取器。
- 当前向量检索使用 NumPy 暴力余弦，适合万级 chunk；更大规模可再接入 sqlite-vec，不影响现有索引格式。自部署 embedding 服务需提供 OpenAI 兼容的 `/embeddings` 接口；当前没有 OCR 那样的任意 multipart HTTP 向量适配器。

---

## 快速开始

```bash
cd ~/Desktop/Semdex
python3 "Start Semdex.py" --web  # 安装依赖并打开 WebUI（数据和下载都在项目内）
                             # 在“设置”中添加目录后，点“保存并开始索引”
```

已添加索引目录后，可在另一个终端使用命令行：

```bash
uv run semdex index          # 扫描并索引（增量，可反复跑）
uv run semdex search "地铁"   # 命令行搜索
uv run semdex ask "上个月和张三有关的 PDF 在哪"  # 自然语言问答（需启用 LLM）
uv run semdex watch          # 实时监听并增量索引
```

### 原生桌面界面

macOS 上直接双击项目根目录的 `Start Semdex.command`。它会在项目内安装 GUI 依赖并打开原生桌面界面，不需要手动启动服务。

Windows / Linux 或希望从终端启动时（Windows 可将 `python3` 换成 `py`）：

```bash
python3 "Start Semdex.py"
```

桌面界面与 WebUI 共用同一套配置和索引：搜索（混合 / 关键词 / 语义 / 问答）、全文、打开/定位、重新索引、完整重建，以及 WebUI 设置页中的全部设置项均可直接使用。桌面版不启动 HTTP 服务；Windows 使用资源管理器、Linux 打开所在目录作为“定位文件”的降级行为。

也可以先执行 `uv run semdex init`，手工编辑配置文件的 `[watch] folders` 后再运行上述命令。网页右上角的“设置”支持添加一个或多个索引目录，按需开启模型、OCR、ASR 和实体/兜底功能。完整操作和本地服务协议见《[使用说明.md](使用说明.md)》。

安全提示：`semdex serve` 只允许绑定 `127.0.0.1`、`::1` 或 `localhost`，不会对局域网或公网开放。设置接口可以修改模型服务地址和索引范围，因此当前版本不提供无认证的远程监听。

### 接入 LM Studio

1. LM Studio → Developer → **Start Server**（默认 `http://localhost:1234/v1`）
2. 按需加载模型：语义搜索要一个 embedding 模型（推荐 bge-m3），图片识别要一个视觉模型（推荐 Qwen2-VL），问答/实体要一个通用 LLM（推荐 Qwen）
3. 在网页“设置”中，为检索 Agent、实体、兜底、视觉和 embedding 分别填入接口地址与模型名，并按需启用；也可以手工编辑对应 `[models.*]` 小节
4. 补跑：

```bash
uv run semdex index    # waiting_model / waiting_capability 的文件自动补索引
uv run semdex embed    # 给已索引文件补向量（首次启用 embedding 后跑一次）
# 换 embedding 模型时：uv run semdex embed --rebuild
uv run semdex entities # 给已有正文补抽实体（启用 [entities] 后）
```

模型服务不在时一切照常：文本/PDF/Office 正常进索引，混合搜索自动退化为关键词。

网页中的“默认 LLM”是兼容旧版 `[models.llm]` 配置的默认项，不是额外的功能用途。新配置请在实际要使用的“检索 Agent”“实体抽取”或“提取兜底”卡片中分别填写并启用模型；各用途可以指向不同的本地服务。

---

## 架构

```
                        ┌─────────────────────────────┐
                        │   CLI (--json)  │  Web 界面   │
                        └───────┬─────────────┬───────┘
                                │             │
                        ┌───────▼─────────────▼───────┐
                        │        search.py            │
                        │  FTS5 BM25 ⊕ 向量余弦 → RRF   │
                        └──────────────┬──────────────┘
                                       │
   scanner.py ──► indexer.py ──►  SQLite (db.py)
   增量扫描        提取→FTS→分块     files / contents /
   sha256 判重     →向量化          contents_fts / chunks / meta
                      │
              ┌───────▼────────┐         ┌──────────────────┐
              │  extractors/    │────────►│  modelclient.py   │
              │  脚本规则 > 内置  │         │  OpenAI 兼容接口   │
              │  扩展名路由      │         │  llm/vision/embed │
              └────────────────┘         │  (LM Studio 等)    │
                                         └──────────────────┘
```

| 模块 | 职责 |
|---|---|
| `config.py` | TOML 配置（项目内 `.semdex/config.toml`，`-c` / `SEMDEX_CONFIG` 可覆盖） |
| `scanner.py` | 遍历监控文件夹，排除规则/大小上限，两级变化判定，清理消失文件 |
| `extractors/` | 内容提取路由：自定义脚本规则优先，其次内置扩展名映射 |
| `modelclient.py` | OpenAI 兼容模型能力封装，按用途选择不同 LLM / 视觉 / embedding |
| `indexer.py` / `entities.py` | 提取 → 写 contents+FTS → 向量化 → 可选实体关系抽取 |
| `chunker.py` | 段落/句边界优先的滑窗分块 |
| `search.py` | 三种检索模式 + RRF 融合 + 降级逻辑 |
| `agent.py` | 受限工具调用的自然语言检索，不允许任意文件系统或 shell 工具 |
| `watcher.py` | 跨平台事件监听、防抖后复用同一增量扫描管线 |
| `db.py` | SQLite 存储层（WAL、外键级联、FTS5、向量 BLOB） |
| `cli.py` / `gui.py` / `web/` | CLI、原生桌面 GUI 与 WebUI 三个薄壳，共用同一套核心 |

### 文件状态机

```
（扫描发现新文件/内容变化）→ pending → done          正常完成
                                    → failed         提取失败（--retry-failed 重试）
                                    → skipped        无适用提取器
                                    → waiting_model  等模型（启用模型后 index 自动重试）
                                    → waiting_capability  等 OCR / ASR / LibreOffice 等本地能力
                                    → too_large      超过当前大小限制（提高限制后自动重新入队）
```

---

## 技术细节

**中文全文检索——按字切分方案。** SQLite FTS5 的 unicode61 分词器不切分中文，相邻汉字黏成一个 token，"地铁"搜不到"广州地铁三号线"；trigram 分词器又要求至少 3 字符，两字词是盲区。Semdex 的做法（`textutil.py`）：入库时在每个 CJK 字符两侧插空格，让 FTS 按**字**建 token；查询时把含中文的词转成短语查询（`"地 铁"` 要求两字相邻），两字词、人名、专名都能精确命中，英文不受影响。展示用的摘要片段从原文重新截取，空格化文本只存在于索引内部。FTS 完全未命中时还有 LIKE 子串兜底。

**混合检索——RRF 融合。** 关键词（BM25）对精确词强、对模糊表达弱；语义向量相反。默认 hybrid 模式两路各取候选，按 Reciprocal Rank Fusion（`Σ 1/(60+rank)`）融合排序，只依赖名次不依赖两路分数可比性。

**向量存储。** 分块（800 字符、100 重叠、段落/句号边界优先）后调本地 embedding 模型，float32 BLOB 存 SQLite，查询时 numpy 矩阵余弦。万级 chunk 毫秒级，规模上去换 sqlite-vec 只动 `db.py`。embedding 的模型名和服务地址都会记录在 meta 表；任一项变更会立即清除旧向量并暂停语义检索，混合检索自动退化为关键词。执行 `semdex embed --rebuild` 完整重建成功后才恢复语义检索，绝不会混用新旧向量。

**增量索引。** 两级判重：size+mtime 都没变直接跳过（不读文件）；变了才算 sha256，哈希相同只更新元数据不重跑提取——**改一下 mtime 不会烧一遍模型**。删除/移动的文件记录（含 FTS、向量）自动清理。

**优雅降级状态机。** 用于内容提取的视觉或兜底模型发生 `ModelNotConfigured`（没开）/ `ModelUnavailable`（开了连不上）时，文件会进 `waiting_model` 而非 failed，模型就绪后重跑 `index` 自动补齐；实体抽取有独立的等待状态。embedding 中途失败不影响已写入的全文索引，`semdex embed` 随时补。

**扩展一个文件类型有三条路，成本递增：**
1. 配置一条脚本规则（不用改代码）：
   ```toml
   [[extractors.rules]]
   match = "*.eml"
   script = "/Users/me/bin/extract_eml.sh"   # 收到同名同扩展名的安全快照路径，stdout 输出文本
   ```
2. 写一个 `Extractor` 子类（`name` + `exts` + `extract()`），加进 `extractors/__init__.py` 的 `_BUILTIN` 列表；
3. 对可读文本启用 `[agent_fallback]`：只把当前文件的限长文本交给本地 LLM 摘要；二进制格式仍需专用提取器或脚本。

脚本提取器不会直接拿到监控目录中的原始路径。为避免文件在提取时被替换为符号链接，Semdex 会先通过受限文件描述符复制一份同名同扩展名的临时快照，再把该快照路径作为唯一参数传给脚本。依赖相邻文件或原始目录结构的脚本需要改为把所需内容一起写入文件，或实现专用提取器。

---

## 对外接口

### CLI（`--json` 供程序调用，退出码 0/1）

```bash
uv run semdex search "广州地铁" --json --limit 5
# {"ok": true, "query": "广州地铁", "mode": "hybrid", "count": 2, "hits": [
#   {"file_id": 3, "path": "/Users/…/会议记录.txt", "filename": "会议记录.txt",
#    "ext": ".txt", "mtime": 1754…, "score": 0.0163, "snippet": "…", "source": "fulltext"}]}

uv run semdex index --json      # {"ok": true, "scan": {…}, "index": {…}, "elapsed_sec": 1.2}
uv run semdex status --json     # 计数、模型开关、路径
uv run semdex embed --json      # 补向量
uv run semdex ask "上月的 PDF" --json  # 自然语言检索
uv run semdex entities --json   # 补抽实体关系
```

### REST API（`semdex serve` 后）

| 方法 | 路径 | 说明 |
|---|---|---|
| GET | `/api/search?q=&mode=hybrid\|fulltext\|semantic&limit=` | 搜索 |
| GET | `/api/ask?q=` | 本地 LLM 工具调用问答 |
| GET | `/api/status` | 状态计数 / 模型开关 / 是否索引中 |
| GET | `/api/settings` | 当前设置（不返回 API Key） |
| PUT | `/api/settings` | 校验、原子保存并热加载设置 |
| POST | `/api/index` | 触发后台重新索引（进行中返回 409） |
| POST | `/api/rebuild` | 清空派生索引并完整重建（进行中返回 409） |
| GET | `/api/content?file_id=` | 查看某文件提取出的全文 |
| POST | `/api/open` `{path, reveal}` | 打开文件 / 在访达中显示（仅 macOS） |

---

## 项目自包含（方便整体迁移）

Semdex 自己的依赖环境、索引、临时文件和可选 Whisper 下载都默认落在项目文件夹内：

```text
Semdex/
  .uv-python/           uv 下载和管理的 Python 解释器（需要时）
  .venv/                 Python 虚拟环境
  .uv-cache/             uv / pip 下载缓存
  .semdex/
    config.toml          本地设置（权限 0600）
    index.db*            SQLite 索引及 WAL/SHM
    tmp/                 提取和 OCR 的受控临时文件
    models/whisper/      faster-whisper 下载的模型
```

`Start Semdex.command`（macOS）和 `Start Semdex.py`（Windows / Linux / 终端）都会在调用 `uv sync` 前设置 uv 托管 Python、Hugging Face、Python 包缓存和临时目录环境变量到上述项目目录。运行 `semdex gui` 或 `semdex serve` 时，未显式覆盖的默认配置也使用同一目录。项目根目录、`.semdex/`、`.venv/`、`.uv-cache/` 和 `.uv-python/` 已在默认索引排除规则中，避免索引自身的数据。

整个 `Semdex/` 文件夹可以直接搬到外置硬盘。注意 `.venv` 里记录了绝对路径，**搬家后在新位置通过启动器同步一次**：

```bash
python3 "Start Semdex.py" --sync-only
```

初始模板中的 `db_path`、`temp_dir` 和 `model_dir` 都相对 `.semdex/config.toml` 解析；复制整个项目到外置硬盘后会自动指向新位置。配置里可以改为绝对外部路径，也可以使用 `-c` / `SEMDEX_CONFIG` 指定独立配置。旧版本的 `~/.semdex/` 不会被自动移动或删除；需要继续使用旧索引时，显式通过 `-c ~/.semdex/config.toml` 指定即可。

需要 ASR 时，使用 `python3 "Start Semdex.py" --with-asr --sync-only` 安装可选依赖。若刻意绕过启动器直接执行首次 `uv sync`，请先设置 `UV_PYTHON_INSTALL_DIR` 到项目内 `.uv-python/`，否则 uv 托管解释器会下载到用户目录。

## 测试

```bash
uv run pytest -q    # 离线回归：索引、搜索、OCR/格式路由、实体、Agent、增量和降级路径
```
