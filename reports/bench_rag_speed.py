"""RAG retrieval speed benchmark (bench KB = small, clean, self-contained).

Measures the three caching layers added for the P1 speed iteration by timing
the same retrieval three times within one process:

  1. cold   : first call (load index from disk + full vector validation + fetch)
  2. warm   : second call (in-process index reuse + validation skipped)
  3. hot    : third call (idempotent query served from the query-result cache)

Run from the project root with the project venv:
    .venv\\Scripts\\python.exe reports\\bench_rag_speed.py

Output (milliseconds) is intentionally best-effort, not a formal benchmark.
"""

from __future__ import annotations

import asyncio
import pathlib
import time

# ---------- 1. Model-catalog config (local Ollama) ---------------------------
CATALOG = {
    "version": 1,
    "services": {
        "llm": {"active_profile_id": None, "active_model_id": None, "profiles": []},
        "search": {"active_profile_id": None, "profiles": []},
        "embedding": {
            "active_profile_id": "emb-ollama",
            "active_model_id": "emb-nomic",
            "profiles": [
                {
                    "id": "emb-ollama",
                    "name": "Local Ollama",
                    "binding": "ollama",
                    "base_url": "http://localhost:11434",
                    "api_key": "",
                    "api_version": "",
                    "extra_headers": {},
                    "models": [
                        {
                            "id": "emb-nomic",
                            "name": "nomic-embed-text",
                            "model": "nomic-embed-text",
                            "dimension": 768,
                            "supported_dimensions": "512,768,1024",
                        }
                    ],
                }
            ],
        },
    },
}

# ---------- 2. Small, clean retrieval corpus (public/educational wording) -----
SAMPLE = """
Introduction to Gradient Descent

Machine-learning models often learn by minimizing a loss function. The loss
measures how far the model's predictions are from the true labels. Gradient
descent is the workhorse optimization algorithm used to find the parameter
values that minimize this loss.

Concept

Imagine a landscape where the height at each point equals the loss for a given
set of parameters. We want to walk downhill toward the lowest valley. The
gradient is a vector that points in the direction of steepest ascent. If we
want to reduce the loss, we should move opposite to the gradient, that is,
downhill.

The update rule

At each step we update every parameter theta according to:

    theta := theta - learning_rate * gradient_of_loss(theta)

The learning rate, often written as alpha or eta, controls how large a step we
take. A too-small learning rate makes learning slow because each step barely
changes the parameters. A too-large learning rate can cause the loss to
oscillate or even diverge, because we overshoot the minimum.

Choosing the learning rate

In practice the learning rate is one of the most important hyperparameters.
Common strategies include using a fixed rate, decaying the rate over time,
and adaptive methods such as Adam that keep a per-parameter step size. It is
often worth testing a small grid of learning rates and observing the loss
curve on a validation split.

Stochastic and batch variants

Full-batch gradient descent computes the gradient using the entire training
set each step, which is accurate but expensive. Stochastic gradient descent
(SGD) uses a single randomly chosen example per step, which is cheap but
noisy. Mini-batch gradient descent sits in between: it samples a small random
subset of examples per step, balancing speed and stability. Most modern
libraries use mini-batches.

Local minima and plateaus

Gradient descent only guarantees progress toward a local, not necessarily
global, minimum. In high-dimensional parameter spaces, many local minima can
be near-optimal in practice. Momentum helps by adding a fraction of the
previous update to the current step, smoothing the trajectory and escaping
small plateaus.

Convergence criteria

We typically stop training when the loss change between epochs falls below a
small threshold, when a fixed number of epochs is reached, or when the
validation metric stops improving. Monitoring both training and validation
loss helps detect overfitting, where the model memorizes the data rather than
generalizing.

Relation to other algorithms

Gradient descent is a first-order method because it uses only the gradient.
Newton's method uses second derivatives for faster local convergence but is
more expensive and harder to apply at scale. Many libraries implement
first-order methods with momentum and adaptive learning rates as the default
choice for neural networks.

Summary

Gradient descent iteratively moves parameters opposite to the gradient to
minimize a differentiable loss. The learning rate trades speed against
stability, mini-batches trade noise against cost, and momentum helps escape
plateaus. It is the foundation of modern deep-learning training.
"""


def _setup_catalog() -> pathlib.Path:
    from ml_tutor.services.config.model_catalog import get_model_catalog_service

    service = get_model_catalog_service()
    service.save(CATALOG)
    return service.path


def _write_source(root: pathlib.Path) -> pathlib.Path:
    source = root / "data" / "bench_source.txt"
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_text(SAMPLE, encoding="utf-8")
    return source


async def _main() -> None:
    root = pathlib.Path(r"E:\课设2026\ml-tutor")
    catalog_path = _setup_catalog()
    print("catalog:", catalog_path)

    source = _write_source(root)
    kb_dir = root / "data" / "knowledge_bases"

    from ml_tutor.services.rag.service import RAGService

    service = RAGService(kb_base_dir=str(kb_dir))

    print("building KB 'bench' ...")
    ok = await service.initialize("bench", [str(source)])
    print("build ok:", ok)

    query = "What is gradient descent and how does the learning rate affect it?"
    labels = [
        "cold (load+validate+fetch)",
        "warm (index cache+skip validate)",
        "hot  (query result cache)",
    ]
    times: list[float] = []
    for label in labels:
        t0 = time.perf_counter()
        result = await service.search(query, "bench")
        dt = (time.perf_counter() - t0) * 1e3
        times.append(dt)
        print(f"{label:<30}: {dt:8.1f} ms  (answer_chars={len(result.get('answer') or '')})")

    if len(times) == 3 and times[0] > 0:
        print(f"warm vs cold : {times[1] / times[0]:.2f}x")
        print(f"hot  vs cold : {times[2] / times[0]:.2f}x")


if __name__ == "__main__":
    asyncio.run(_main())