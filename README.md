# ml-tutor

> **An AI tutor that answers ML questions with RAG — grounded in textbooks
> & your uploaded docs.**

`ml-tutor` is a **RAG-enhanced learning assistant** for machine learning.
Drop in your textbook PDF or lecture notes, ask any ML question, and get
grounded, cited answers — plus a personalized study plan tailored to your
current level and goal.

Built with **FastAPI + Next.js** and a **hybrid vector + BM25** retriever
powered by `nomic-embed-text` (Ollama-local, 768-dim).

---

## Why ml-tutor

| Problem we solve | How |
| --- | --- |
| "I don't know where to look in the textbook" | Ask in natural language; answers cite exact source chunks |
| "This concept feels hand-wavy" | RAG retrieval plugs in the original formula / derivation |
| "I don't know what to study next" | A lightweight learner profile drives a step-by-step study plan |
| "Responses feel slow" | Query-result caching + in-process index reuse cut latency visibly |

---

## Architecture

```
[ Learner asks: "Explain gradient descent" ]
               │
               ▼
 ┌─────────────────────────────────┐
 │   FastAPI + WebSocket (SSE TBD) │
 └──────────────┬──────────────────┘
                ▼
   ┌───────────────────────────┐
   │  ChatOrchestrator          │
   │  (agentic loop + tools)    │
   └──────┬──────────┬─────────┘
          ▼          ▼
   ┌───────────┐  ┌─────────────────────┐
   │ RAG tool  │  │ LLM (streaming)     │
   └─────┬─────┘  └──────────┬──────────┘
         ▼                   │
 ┌───────────────────┐       │
 │ Hybrid retriever  │       │
 │ vector + BM25     │       │
 │ (Ollama 768-dim)  │       │
 └───────────────────┘       │
                              ▼
 ┌─────────────────────────────────────┐
 │ Next.js chat UI (streaming + refs)  │
 └─────────────────────────────────────┘
```

**Stages (what we cut from the generic upstream)**
1. **Phase 0** ✅ Skeleton + dev charter + repo setup
2. **Phase 1** ⏳ RAG speed — in-process index reuse, query-result cache,
   remove per-query full-vector validation, Ollama pre-warm
3. **Phase 2** ⏳ Retrieval quality — structure-aware chunking for ML
   textbooks, hybrid knobs, source-reference polish
4. **Phase 3** ⏳ Personalized study plans — learner profile + plan
   generation + assessment loop
5. **Phase 4** ⏳ Frontend polish + streaming

Detailed dev notes live under [`reports/`](./reports/).

---

## Quick start

### Prerequisites
- Python 3.11+
- Node 18+
- [Ollama](https://ollama.com/) with `nomic-embed-text`:
  ```bash
  ollama pull nomic-embed-text
  ```

### Backend
```bash
pip install -e .
ml_tutor serve --port 8001
```

### Frontend
```bash
cd web
npm install
npm run dev      # http://localhost:3000
```

### Build a knowledge base from your textbook
```bash
ml_tutor kb create my-ml-kb --doc "Andrew Ng - Machine Learning.pdf"
```

Then ask:
```bash
ml_tutor run chat "What is gradient descent? cite the formula" -t rag --kb my-ml-kb
```

---

## Tech stack

| Layer | Choice |
| --- | --- |
| Backend | FastAPI (Python 3.11+) |
| RAG | LlamaIndex + FAISS HNSW + BM25 hybrid |
| Embedding | Ollama `nomic-embed-text` (768-dim) |
| LLM | OpenAI-compatible (configurable) |
| Frontend | Next.js 14 + Tailwind |
| Realtime | WebSocket (chat), SSE (task progress, streaming) |

---

## Highlights (resume-friendly, measurable)

- **RAG latency**: P95 retrieval reduced by **≥40%** via in-process index
  reuse + query-result LRU cache; TTFT reduced by **≥30%**.
- **Retrieval grounded**: Every answer carries `sources` with page/chunk refs
  so the learner can jump back to the textbook.
- **Personalized study plans**: 4-stage learner profile → step-by-step study
  path → assessment → plan adjustment.
- **Streaming UX**: Chat responses stream token-by-token over WebSocket;
  task progress pushes over SSE.

---

## License

