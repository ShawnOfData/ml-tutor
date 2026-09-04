# P2 后端闭环验收报告

## 一、开发内容

完成后端"配置解析 → RAG 检索 → LLM 非流式/流式生成"的闭环验收，确认后端主链路在本项目独立环境下可端到端运行。

主要动作：

1. 在 `model_catalog.json` 新增 LLM profile：本机 Ollama `qwen2:7b`，`base_url=http://localhost:11434/v1`（走 OpenAI 兼容端点），无需 API key。
2. 编写临时验收脚本 `reports/verify_llm_flow.py`：解析 LLM 配置 → 用一段 ML 教材类长文本建 KB → 检索得上下文 → LLM 非流式 + 流式生成。
3. 定位并处理两项环境问题（详见"警惕问题"）。

## 二、涉及文件

| 文件 | 说明 |
| --- | --- |
| `data/user/settings/model_catalog.json` | 新增 `services.llm` profile（ollama / qwen2:7b），终结空 profile 状态 |
| `reports/verify_llm_flow.py` | 临时验收脚本（数据根固定为本项目、建真实 KB、检索、双路生成） |
| `reports/P2_backend_loop_report.md` | 本报告 |
| 上游 `DeepTutor-main\.venv` | 卸载遗留的 `deeptutor-1.4.2` editable 安装 |

## 三、需要警惕的问题

1. **运行目录决定数据根（本次串包根因）**：`ml_tutor/runtime/home.py` 在未设置 `DEEPTUTOR_HOME` 时用 `Path.cwd()` 作为数据根。此前 shell 工作目录残留在上游 `DeepTutor-main`，导致 `model_catalog` 被解析到上游配置文件，串出 DeepSeek（key 已失效）。**必须在本项目目录下运行**，或显式设置 `DEEPTUTOR_HOME=ml-tutor`。
2. **上游 editable 残留**：`DeepTutor-main\.venv` 曾装有 `deeptutor-1.4.2` editable，会把 `import ml_tutor` 劫持到上游源码。已卸载解除。若日后复现配置"读错地方"，优先查：上游 site-packages 是否残留 `__editable__*.pth`/finder、以及当前 cwd。
3. **本地 Ollama 的 URL 形态**：`build_chat_url` 会给 base_url 追加 `/chat/completions`，因此下载的是 `http://localhost:11434/chat/completions` 时返回 404；必须用 `http://localhost:11434/v1` → 命中 `/v1/chat/completions`。
4. **合成小文档触发 BM25 越界**：不足 10 个 chunk 的小语料会让 fusion/BM25 报 `k larger than corpus` 的 ValueError（`service.search` 内部已捕获并正常返回）。真实教材语料节点远大于 k，无此现象；但验收结论遇到告警无需惊慌。
5. **faiss 未安装**：日志提示 `FAISS conversion failed, fallback to SimpleVectorStore`。功能可用，性能（向量检索）弱于 faiss，后续可选装。

## 四、需要对接的内容

- LLM profile（ollama/qwen2:7b）需对接 Settings > Catalog 界面（前端展示、连接测试）。
- 运行时需保证 `cwd`/`DEEPTUTOR_HOME` 指向本项目，前后端启动时统一注入，避免多目录串配。

## 五、测试验证

验收命令（本项目目录下）：

```
.venv\Scripts\python.exe reports\verify_llm_flow.py
```

结果：

| 环节 | 结果 |
| --- | --- |
| 配置解析 | `qwen2:7b / ollama / http://localhost:11434/v1` |
| 检索（真实教材 KB） | `chars=182, error=None` |
| LLM 非流式 | `chars=1878`，回答正确 |
| LLM 流式 | `chunks=219, chars=2060`，逐块输出正常 |
| 汇总 | `retrieve=OK, sync=True, stream=True` |

## 六、后续建议

1. 接入 Ollama 预热钩子（启动时 `verify_embedding_connectivity`），消除首次检索冷启动延迟。
2. 接入真实教材入库，做更大语料规模的检索/生成校验。
3. 首次 `git` 提交，把可运行骨架推上 `ml-tutor` 远程仓库。
4. 评估是否安装 `faiss` 以增强向量检索与持久化。