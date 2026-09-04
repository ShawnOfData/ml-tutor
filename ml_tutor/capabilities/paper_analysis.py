from __future__ import annotations

import json
import logging
import re
from typing import Any

from ml_tutor.capabilities._i18n import StatusI18n
from ml_tutor.capabilities._shared import emit_capability_result
from ml_tutor.capabilities.request_contracts import get_capability_request_schema
from ml_tutor.core.agentic.usage import UsageTracker
from ml_tutor.core.capability_protocol import BaseCapability, CapabilityManifest
from ml_tutor.core.context import UnifiedContext
from ml_tutor.core.stream_bus import StreamBus


_ARXIV_ID_RE = re.compile(r"(?:arxiv\.org/(?:abs|pdf)/)?(\d{4}\.\d{4,5})(?:v\d+)?")
_URL_RE = re.compile(r"https?://[^\s]+")


def _extract_arxiv_id(text: str) -> str | None:
    m = _ARXIV_ID_RE.search(text)
    return m.group(1) if m else None


def _extract_url(text: str) -> str | None:
    m = _URL_RE.search(text)
    return m.group(0) if m else None


def _is_pdf_url(url: str) -> bool:
    return url.lower().endswith(".pdf") or "/pdf/" in url


def _to_abstract_url(arxiv_id: str) -> str:
    return f"https://arxiv.org/abs/{arxiv_id}"


_SYSTEM_PROMPT = """你是一位顶级的 AI/ML 论文分析专家。请基于论文内容，用中文生成一份结构化的论文拆解报告。

请严格按以下 JSON 格式输出（不要包含 markdown 代码块标记，仅输出纯 JSON）：

{{
  "title": "论文标题（中文翻译）",
  "original_title": "原标题",
  "authors": ["作者1", "作者2"],
  "year": 2024,
  "venue": "发表会议/期刊",
  "category": "论文类别（如：LLM、多模态、强化学习等）",
  "core_problem": "核心问题",
  "core_contribution": "核心贡献（1-2句话）",
  "methodology": {{
    "approach": "总体方法描述",
    "key_techniques": ["关键技术1", "关键技术2"],
    "innovation_points": ["创新点1", "创新点2"]
  }},
  "results": {{
    "main_findings": "主要发现",
    "key_metrics": {{"指标名": "数值"}},
    "ablation_findings": "消融实验发现（如有）"
  }},
  "strengths": ["优点1", "优点2", "优点3"],
  "weaknesses": ["局限1", "局限2"],
  "related_work": "与已有工作的关系",
  "impact": "潜在影响与应用价值",
  "takeaways": ["关键 takeaways/学习要点1", "学习要点2", "学习要点3"],
  "discussion_questions": ["思考问题1", "思考问题2", "思考问题3"]
}}"""


_RECOMMENDED_ML_PAPERS = [
    {"title": "Attention Is All You Need", "url": "https://arxiv.org/abs/1706.03762", "venue": "NeurIPS 2017", "category": "NLP / 大模型基础"},
    {"title": "Deep Residual Learning for Image Recognition (ResNet)", "url": "https://arxiv.org/abs/1512.03385", "venue": "CVPR 2016", "category": "计算机视觉"},
    {"title": "BERT: Pre-training of Deep Bidirectional Transformers", "url": "https://arxiv.org/abs/1810.04805", "venue": "NAACL 2019", "category": "NLP / 预训练"},
    {"title": "Generative Adversarial Networks (GAN)", "url": "https://arxiv.org/abs/1406.2661", "venue": "NeurIPS 2014", "category": "生成模型"},
    {"title": "Mastering the Game of Go with Deep Neural Networks (AlphaGo)", "url": "https://www.nature.com/articles/nature16961", "venue": "Nature 2016", "category": "强化学习"},
    {"title": "A Guide to Convolution Arithmetic for Deep Learning", "url": "https://arxiv.org/abs/1603.07285", "venue": "arXiv 2016", "category": "深度学习基础"},
    {"title": "LoRA: Low-Rank Adaptation of Large Language Models", "url": "https://arxiv.org/abs/2106.09685", "venue": "ICLR 2022", "category": "LLM / 高效微调"},
    {"title": "Denoising Diffusion Probabilistic Models (DDPM)", "url": "https://arxiv.org/abs/2006.11239", "venue": "NeurIPS 2020", "category": "生成模型"},
    {"title": "CLIP: Learning Transferable Visual Models From Natural Language", "url": "https://arxiv.org/abs/2103.00020", "venue": "ICML 2021", "category": "多模态"},
    {"title": "DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model", "url": "https://arxiv.org/abs/2405.04434", "venue": "arXiv 2024", "category": "LLM / MoE"},
    {"title": "GPT-4 Technical Report", "url": "https://arxiv.org/abs/2303.08774", "venue": "arXiv 2023", "category": "LLM"},
    {"title": "Llama: Open and Efficient Foundation Language Models", "url": "https://arxiv.org/abs/2302.13971", "venue": "arXiv 2023", "category": "LLM / 开源模型"},
    {"title": "Playing Atari with Deep Reinforcement Learning (DQN)", "url": "https://arxiv.org/abs/1312.5602", "venue": "NeurIPS 2013", "category": "强化学习"},
    {"title": "An Image is Worth 16x16 Words: Transformers for Image Recognition (ViT)", "url": "https://arxiv.org/abs/2010.11929", "venue": "ICLR 2021", "category": "计算机视觉"},
    {"title": "Graph Attention Networks (GAT)", "url": "https://arxiv.org/abs/1710.10903", "venue": "ICLR 2018", "category": "图神经网络"},
]


class PaperAnalysisCapability(BaseCapability):
    manifest = CapabilityManifest(
        name="paper_analysis",
        description="Deep paper analysis with structured report generation.",
        stages=["extracting", "analyzing", "summarizing", "reporting"],
        tools_used=["web_fetch", "paper_search", "web_search"],
        cli_aliases=["paper"],
        trigger_keywords=["论文拆解", "论文分析", "paper analysis", "拆解论文", "分析论文", "读论文", "论文解读"],
        request_schema=get_capability_request_schema("paper_analysis"),
    )

    async def run(self, context: UnifiedContext, stream: StreamBus) -> None:
        from ml_tutor.services.llm.config import get_llm_config

        llm_config = get_llm_config()
        kb_name = context.knowledge_bases[0] if context.knowledge_bases else None
        turn_id = str(context.metadata.get("turn_id", "") or context.session_id or "paper-analysis")
        i18n = StatusI18n(self.name, context.language)

        usage = UsageTracker(model=getattr(llm_config, "model", None))
        text = str(context.user_message or "").strip()
        url = _extract_url(text)
        arxiv_id = _extract_arxiv_id(text)
        file_attachment = self._find_file_attachment(context)

        paper_content = ""

        async with stream.stage("extracting", source=self.name):
            if file_attachment is not None:
                await stream.progress(i18n.t("extracting_pdf", "Parsing uploaded document..."), source=self.name)
                paper_content = await self._extract_document(file_attachment, stream, i18n)
            elif arxiv_id is not None:
                abs_url = _to_abstract_url(arxiv_id)
                await stream.progress(i18n.t("extracting_url", "Fetching paper via URL..."), source=self.name)
                paper_content = await self._fetch_paper(abs_url, stream, i18n)
            elif url is not None:
                await stream.progress(i18n.t("extracting_url", "Fetching paper via URL..."), source=self.name)
                if _is_pdf_url(url):
                    paper_content = await self._fetch_pdf_url(url, stream, i18n)
                else:
                    html = await self._web_fetch(url, stream, i18n)
                    if html:
                        paper_content = html[:8000]
            else:
                await stream.progress(i18n.t("extracting", "Searching paper..."), source=self.name)
                paper_content = await self._search_paper(text, stream, i18n)

            if not paper_content or len(paper_content.strip()) < 50:
                await stream.error(i18n.t("paper_not_found", "Paper not found."), source=self.name)
                return

            await stream.progress(i18n.t("paper_found", "Paper found.", title="Paper"), source=self.name)

        async with stream.stage("analyzing", source=self.name):
            await stream.progress(i18n.t("analyzing", "Analyzing paper..."), source=self.name)
            await stream.thinking(i18n.t("analyzing_core", "Analyzing core contribution..."), source=self.name, stage="analyzing")

        async with stream.stage("summarizing", source=self.name):
            await stream.progress(i18n.t("summarizing", "Generating summary..."), source=self.name)

        async with stream.stage("reporting", source=self.name):
            await stream.progress(i18n.t("reporting", "Generating report..."), source=self.name)
            report = await self._llm_analyze(paper_content, llm_config, usage, stream, i18n)
            if report:
                await stream.content(report, source=self.name, stage="reporting")
                parsed = self._try_parse_json(report)
                payload: dict[str, Any] = {
                    "response": report,
                    "report_json": parsed,
                    "recommended_papers": _RECOMMENDED_ML_PAPERS,
                }
                await emit_capability_result(stream, payload, source=self.name, usage=usage)
            else:
                await stream.error(i18n.t("llm_call_failed", "LLM call failed."), source=self.name)

    def _find_file_attachment(self, context: UnifiedContext) -> dict[str, Any] | None:
        for a in context.attachments:
            name = (a.filename or "").lower()
            if name.endswith(".pdf") or name.endswith(".doc") or name.endswith(".docx"):
                return {"filename": a.filename, "base64": a.base64, "ext": name.rsplit(".", 1)[-1]}
            mime = (a.mime_type or "").lower()
            if mime in ("application/pdf", "application/msword", "application/vnd.openxmlformats-officedocument.wordprocessingml.document"):
                ext = "pdf" if "pdf" in mime else "docx"
                return {"filename": a.filename, "base64": a.base64, "ext": ext}
            if (a.type or "").lower() in ("pdf", "doc", "docx"):
                return {"filename": a.filename, "base64": a.base64, "ext": a.type.lower()}
        return None

    async def _extract_document(self, attachment: dict[str, Any], stream: StreamBus, i18n: StatusI18n) -> str:
        import base64
        import io

        ext = (attachment.get("ext") or "").lower()
        data = base64.b64decode(attachment["base64"])
        fname = attachment.get("filename", "document")

        if not data:
            return f"文件「{fname}」内容为空，请确认文件非空后重新上传。"

        if ext == "pdf":
            return self._extract_pdf_in_memory(data, fname)

        if ext == "docx":
            return self._extract_docx_in_memory(data, fname)

        if ext == "doc":
            return f"不支持旧版 .doc 格式「{fname}」，请在 Word 中另存为 .docx 或 PDF 后重新上传。"

        return f"不支持的文件格式「{fname}」，请上传 PDF 或 DOCX 文件。"

    def _extract_pdf_in_memory(self, data: bytes, fname: str) -> str:
        try:
            import fitz
            with fitz.open(stream=data, filetype="pdf") as doc:
                if doc.is_encrypted and not doc.authenticate(""):
                    return f"PDF 文件「{fname}」已加密，无法读取，请解除密码后重新上传。"
                text = "\n".join(page.get_text() for page in doc)
            result = text.strip()
            if result:
                return result[:12000]
            return f"PDF 文件「{fname}」已解析但未提取到文字内容，可能是扫描件或图片型 PDF，建议使用可复制文字的 PDF。"
        except ImportError:
            pass
        except Exception as exc:
            logging.getLogger(__name__).warning("fitz failed on %s: %s, falling back to pypdf", fname, exc)

        try:
            from pypdf import PdfReader
            import io
            reader = PdfReader(io.BytesIO(data))
            if getattr(reader, "is_encrypted", False):
                return f"PDF 文件「{fname}」已加密，无法读取。"
            pages = [f"--- Page {i} ---\n{p.extract_text() or ''}" for i, p in enumerate(reader.pages, 1)]
            text = "\n\n".join(pages)
            result = text.strip()
            if result:
                return result[:12000]
            return f"PDF 文件「{fname}」已解析但未提取到文字内容。"
        except ImportError:
            return f"无法读取 PDF 文件「{fname}」，服务器缺少 PDF 解析库（pymupdf/pypdf），请联系管理员安装。"
        except Exception as exc:
            logging.getLogger(__name__).warning("pypdf also failed on %s: %s", fname, exc)
            return f"无法解析 PDF 文件「{fname}」，文件可能已损坏或格式不兼容。"

    def _extract_docx_in_memory(self, data: bytes, fname: str) -> str:
        import io
        try:
            from docx import Document as DocxDocument
            from docx.opc.exceptions import PackageNotFoundError
        except ImportError:
            return f"无法读取 DOCX 文件「{fname}」，服务器缺少 python-docx 库。"

        try:
            doc = DocxDocument(io.BytesIO(data))
            text = "\n".join(p.text for p in doc.paragraphs)
            result = text.strip()
            if result:
                return result[:12000]
            return f"DOCX 文件「{fname}」已解析但未找到可提取的文字内容，请确认文件非空。"
        except PackageNotFoundError:
            pass
        except Exception as exc:
            logging.getLogger(__name__).warning("python-docx failed on %s: %s, falling back to XML", fname, exc)

        # Fallback: parse DOCX as ZIP and extract raw XML text
        try:
            import zipfile
            from xml.etree import ElementTree
            with zipfile.ZipFile(io.BytesIO(data)) as z:
                xml_files = [n for n in z.namelist() if n.startswith("word/") and n.endswith(".xml")]
                texts = []
                for xf in xml_files:
                    try:
                        tree = ElementTree.parse(z.open(xf))
                        parts = list(tree.itertext())
                        texts.extend(t for t in parts if t.strip())
                    except Exception:
                        pass
                text = "\n".join(texts)
                result = text.strip()
                if result:
                    return result[:12000]
        except zipfile.BadZipFile:
            pass
        except Exception as exc:
            logging.getLogger(__name__).warning("XML fallback also failed on %s: %s", fname, exc)

        return f"无法解析 DOCX 文件「{fname}」，文件可能已损坏或格式不兼容，请在 Word 中另存为 PDF 后重新上传。"

    async def _fetch_paper(self, url: str, stream: StreamBus, i18n: StatusI18n) -> str:
        html = await self._web_fetch(url, stream, i18n)
        if not html:
            return ""
        text = self._html_to_text(html)
        return text[:8000]

    async def _fetch_pdf_url(self, url: str, stream: StreamBus, i18n: StatusI18n) -> str:
        abs_url = url.replace("/pdf/", "/abs/").replace(".pdf", "")
        html = await self._web_fetch(abs_url, stream, i18n)
        if not html:
            return f"无法获取 PDF 链接中的内容：{url}，请尝试直接粘贴论文标题和摘要。"
        text = self._html_to_text(html)
        return text[:8000]

    async def _search_paper(self, query: str, stream: StreamBus, i18n: StatusI18n) -> str:
        return f"未找到相关论文，请尝试直接输入论文的 arXiv 链接或粘贴标题摘要。"

    async def _web_fetch(self, url: str, stream: StreamBus, i18n: StatusI18n) -> str:
        from ml_tutor.tools.builtin import WebFetchTool

        tool = WebFetchTool()
        result = await tool.execute(url=url)
        if result.success:
            return result.content
        return ""

    async def _llm_analyze(
        self,
        paper_content: str,
        llm_config: Any,
        usage: UsageTracker,
        stream: StreamBus,
        i18n: StatusI18n,
    ) -> str:
        from ml_tutor.core.agentic import (
            LLMClientConfig,
            build_completion_kwargs,
            build_openai_client,
        )

        client_config = LLMClientConfig(
            binding=llm_config.binding or "openai",
            model=llm_config.model,
            api_key=llm_config.api_key,
            base_url=llm_config.base_url,
            api_version=getattr(llm_config, "api_version", None),
            extra_headers=getattr(llm_config, "extra_headers", None),
            reasoning_effort=getattr(llm_config, "reasoning_effort", None),
        )
        client = build_openai_client(client_config)
        kwargs = build_completion_kwargs(
            temperature=0.3,
            model=llm_config.model,
            max_tokens=4096,
            binding=llm_config.binding or "openai",
            reasoning_effort=client_config.reasoning_effort,
        )
        messages = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f"请分析以下论文内容，生成结构化的拆解报告：\n\n{paper_content}",
            },
        ]
        try:
            response = await client.chat.completions.create(
                model=llm_config.model,
                messages=messages,
                **kwargs,
            )
            result = response.choices[0].message.content or ""
            if response.usage:
                usage.add_from_response(response.usage)
            return result
        except Exception as exc:
            logging.getLogger(__name__).warning("_llm_analyze failed: %s", exc)
            return ""

    def _html_to_text(self, html: str) -> str:
        text = re.sub(r"<[^>]+>", " ", html)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def _try_parse_json(self, text: str) -> dict[str, Any] | None:
        text = text.strip()
        if text.startswith("```"):
            lines = text.splitlines()
            text = "\n".join(lines[1:-1])
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            m = re.search(r"\{.*\}", text, re.DOTALL)
            if m:
                try:
                    return json.loads(m.group(0))
                except json.JSONDecodeError:
                    pass
        return None
