# P1 RAG 提速模块报告

**完成时间**：2026-09-04
**负责人**：ShawnOfData
**状态**：✅ 完成（检索热路径提速三项已落地并通过 import 验证；基准测试待跑）

## 1. 开发内容（做了什么）

本次将 ML Tutor 后端中与「机器学习助教」主线直接相关的模块迁入 `ml-tutor`（独立项目，全新 git 历史），并落地本轮核心——**RAG 检索热路径三项提速优化**。

### 1.1 项目收敛：只保留后端核心子集
- 新建独立项目骨架于 `e:\课设2026\ml-tutor`：`docs/AI_DEV_CHARTER.md`（AI 开发宪法）、`.gitignore`、`README.md`、`reports/`。
- 从 `ML Tutor-main` 复制后端代码包（保留 `ml_tutor` 包名便于 `import` 自洽），**剔除**无关目录：`tutorbot / co_writer / events / app / config / logging`。
- 修正一处 robocopy 误删：`XD config` 按名称匹配所有层级，误删了 `services/config`，已补复制。

### 1.2 提速优化三件套（本轮核心）
1. **检索结果缓存**（[retrieval_cache.py](file:///e:/课设2026/ml-tutor/ml_tutor/services/rag/retrieval_cache.py)）
   - 新增零依赖的线程安全 LRU + TTL 缓存；`normalize_query` 归一化空白/大小写，让近似查询可命中。
   - 接入 [service.py `RAGService.search`](file:///e:/课设2026/ml-tutor/ml_tutor/services/rag/service.py)：命中缓存直接返回，miss 才走完整检索。

2. **进程内索引 / 存储上下文缓存**（[storage.py](file:///e:/课设2026/ml-tutor/ml_tutor/services/rag/pipelines/llamaindex/storage.py)）
   - 新增 `_INDEX_CACHE`，按 `storage_dir` 复用 `StorageContext + VectorStoreIndex`，消除每次检索的 `load_index_from_storage` 磁盘重解析。
   - 用「目录内文件最新 mtime」做失效指纹（`_storage_cursor`），KB 增/删/重建自动触发重载；上限 8 个索引。

3. **去除每次全量向量校验**（storage.py）
   - 原 `retrieve_nodes` 每次都 `_validate_persisted_embeddings` 全量遍历所有向量。
   - 改为 `_needs_validation` + 惰性标记：同一磁盘状态只校验一次，热路径跳过。

### 1.3 遗留说明
- Ollama 预热无新增接口：复用现有 `embedding_adapter.verify_embedding_connectivity()`（一次 embed 即触发模型载入），接入点留给启动钩子阶段。

## 2. 涉及文件

| 文件路径 | 改动类型 | 作用 |
| --- | --- | --- |
| `ml_tutor/services/rag/retrieval_cache.py` | 新增 | LRU+TTL 结果缓存 + 查询归一化 |
| `ml_tutor/services/rag/service.py` | 修改 | `search()` 接入结果缓存；新增 `_cache_key`/`_retrieve` |
| `ml_tutor/services/rag/pipelines/llamaindex/storage.py` | 修改 | 索引缓存 + 惰性向量校验；`import threading/time` |
| `ml_tutor/services/config/` | 补复制 | robocopy 误删后恢复 |
| `docs/AI_DEV_CHARTER.md`、`.gitignore`、`README.md`、`reports/` | 新增 | 项目宪法与骨架 |

## 3. 需要警惕的问题（坑 / 风险 / 未解决事项）

1. **`XD config` 误删**：robocopy 按目录名全层级匹配，后期增补目录时须核对是否误伤同名子目录。
2. **editable 安装冲突**：本机 `ML Tutor-main` 以 editable 方式安装为 `ml_tutor`，验证时须在 `ml-tutor` 目录下运行且依赖 `sys.path` 优先（已用 cwd 验证命中本项目）。正式开发建议为 ml-tutor 建独立 venv。
3. **索引缓存一致性**：mtime 指纹依赖文件系统 mtime 精度；跨文件系统/网络盘部署时有失效滞后风险，`clear_index_cache()` 提供显式兜底。
4. **未跑真实基准**：当前仅通过「import + 语法」验证，无实际检索耗时对照（需数据源/embedding 服务）。

## 4. 需要对接的内容（下游依赖谁 / 谁依赖我）

- 本模块依赖：`ml_tutor.services.embedding`（Ollama `nomic-embed-text` 768 维）、`ml_tutor.services.config`、LlamaIndex + `faiss`。
- 依赖本模块：`knowledge.py`（KB 增/删通过 mtime 变化触发索引缓存失效）、`rag_tool`（agent 调 `RAGService.search`，自动受益缓存）。
- 接口变更：
  - `ml_tutor.services.rag.pipelines.llamaindex.storage` 新增导出 `clear_index_cache()`（KB 变更兜底可调用）；
  - `service.search()` 行为不变，仅内部多缓存层。

## 5. 测试与验证（怎么证明「能用」）

- 语法校验：`ast.parse` 对三个改动文件 → `SYNTAX_OK`。
- import 验证（非沙箱，`cwd=ml-tutor`）：`ml_tutor.services.rag.{service, factory, retrieval_cache}` 与 `pipelines.llamaindex.storage` 全部导入成功；
  - `retrieve_nodes` / `clear_index_cache` 存在性断言通过；
  - `factory.DEFAULT_PROVIDER == "llamaindex"`。
- 性能基准：❌ **未跑**（需 Ollama + 已有 KB；当前机器数据未迁移），下一阶段补 before/after 数字。

## 6. 后续建议

- **补基准测试**：迁移现有 KB（`ml-textbook` / `ML2`）或新建小 KB，实测「缓存命中 vs 未命中」检索耗时差。
- **接入 Ollama 预热**：在服务启动钩子调用 `verify_embedding_connectivity()`。
- **对话主链路 import 验证**：本轮仅验证了 RAG 链，`capabilities/chat` + `agents/chat` 链路待下一轮验证。
- **精简去重（你已确认方向）**：砍多余 capability（visualize/math_animator 等）与 i18n 多语言 prompt，收窄到 RAG + 对话 + 个性化主线。

---

## 7. 补充：D / B / A / C 阶段结果

### 7.1 D — 独立 venv（消除包名冲突）
- 建 `ml-tutor/.venv`（Python 3.11.3），`pip install -e . --no-build-isolation` 成功装入 `ml_tutor-1.4.2`（editable）。
- 新 venv 内 `ml_tutor` **唯一指向 ml-tutor**，与 `ML Tutor-main` 的 editable 安装彻底隔离；RAG/server 依赖全部装齐。
- 踩坑：首次误判安装完成（匹配到 pip 升级日志）；且一个后台 install 未停导致 `[WinError 32] httpx2\_client.py` 文件占用。**已停掉并发 job 后重装成功**。

### 7.2 B — 精简（注销多余能力）
- `runtime/bootstrap/builtin_capabilities.py` 注册项收敛为 **chat / deep_solve / deep_question / deep_research**，从注册表移除 `math_animator / paper_analysis / visualize / auto`。
- 保留其源码文件（被 `agents/*`、`book/blocks` 等级联 import），避免硬删引发大面积断裂——**对外不再暴露/调度**。
- i18n 多语言 prompt 深度削减**留后置**（牵涉 `PromptManager` 全局语言路由，对运行价值低）。

### 7.3 A — 对话主链路 import 验证
- 修复三处复制误删/缺失：`ml_tutor/config`、`ml_tutor/logging`、`ml_tutor/events`（均为 core 依赖，robocopy 按名排除时误伤）。
- 验证通过（新 venv）：`capabilities/chat`、`agents/chat/{agentic_pipeline,chat_agent}`、`deep_solve/deep_question/deep_research`、`services/rag/service`、`runtime/orchestrator`、注册表全部 import OK（EXIT=0）。

### 7.4 C — 检索提速基准（新增小纯净 KB）
- 新建 `data/user/settings/model_catalog.json`（embbing: ollama / nomic-embed-text / 768 维，`data/` 已 gitignore 不入库）。
- 用一段公开教育向文本建 `bench` KB，同进程测同一查询三次（`reports/bench_rag_speed.py`）：

| 阶段 | 耗时 | 相对 cold |
| --- | --- | --- |
| cold（首次：全量加载索引+逐向量校验+融合检索） | **2679.1 ms** | 1× |
| warm（索引缓存命中 + 跳过校验） | **2.4 ms** | ≈0.0009× |
| hot（相同查询命中结果缓存） | **2.3 ms** | ≈0.0009× |

- 结论：检索热路径从**秒级降到毫秒级**（≈1100× 提速），结果缓存对重复查询零开销。cold 里含一次性的持久化加载 + 全量校验，正是缓存+惰性校验要消除的部分。
- 已知小 KB 现象：BM25 在不足 `top_k*2` 的语料上会抛 `ValueError: k ... larger than corpus`，被捕获不影响返回；真实教材语料无此问题。

### 7.5 遗留
- i18n 深度削减（后置）。
- model catalog 目前仅含 embedding；llm/service 对话真正调用 LLM 仍需在 Catalog 配置一个 LLM profile。