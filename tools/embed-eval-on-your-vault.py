#!/usr/bin/env python3
"""
embed-eval-on-your-vault.py  --  Deliverable for "Pick the Model From Your Own Data" (KE-1)

Pick your embedding model from YOUR corpus, not someone else's leaderboard.

The whole idea, in four lines (this is the conceptual entry point from the piece):

    for model in candidate_models:
        embed(model, your_notes)
        score = recall_at_k(model, your_questions, ground_truth, k=5)
        print(f"{model}: recall@5 = {score:.3f}")

Why it matters: a small test set is a saturated benchmark. With ~50 notes the right
answer is near the top no matter which model you use -- there aren't enough documents
for a wrong answer to hide. A saturated benchmark doesn't error; it returns a
confident, specific, USELESS ranking. Run this on your REAL pile, not a toy slice.
On a real pile the ranking can invert (in the source experiment, nomic-embed-text
topped 54 notes but slipped mid-pack at 995 notes; embeddinggemma went the other way).

--------------------------------------------------------------------------------
DEFAULT BACKEND: Ollama (local, free). Start it with `ollama serve` and pull the
models you want to test (e.g. `ollama pull embeddinggemma`). Swap the `embed()`
function to hit any provider (OpenAI, Cohere, a Modal endpoint, etc.).

QUERIES FILE FORMAT (JSON list; one object per test question):
    [
      {"question": "how do I prime the pump tubing?", "answer_note": "projects/garden/priming.md"},
      {"question": "what model won the eval?",        "answer_note": "eval/findings.md"}
    ]
  - "answer_note" is the note that SHOULD be retrieved, given as a path RELATIVE to
    --corpus (must match how the corpus walker names notes below). A CSV with
    columns question,answer_note works too.

USAGE:
    python3 embed-eval-on-your-vault.py \
        --corpus ~/vault \
        --queries queries.json \
        --models embeddinggemma,nomic-embed-text,bge-m3 \
        --k 10 \
        --out results.csv

Output CSV columns match the source eval format exactly:
    name,provider,model,dim,recall@1,recall@5,recall@10,mrr@10,q_latency_ms,approx_cost_usd
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

# ------------------------------------------------------------------ backend ---
OLLAMA_URL = "http://localhost:11434/api/embeddings"


def embed(model: str, text: str, provider: str = "ollama") -> list[float]:
    """Return one embedding vector for `text`. Swap this to change providers.

    Kept deliberately tiny and dependency-free (urllib) so the script is a single
    file you can drop anywhere. For batching/perf, replace with the provider SDK.
    """
    if provider == "ollama":
        payload = json.dumps({"model": model, "prompt": text}).encode()
        req = urllib.request.Request(
            OLLAMA_URL, data=payload, headers={"Content-Type": "application/json"}
        )
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                return json.load(r)["embedding"]
        except urllib.error.URLError as e:
            sys.exit(
                f"embed() failed for model={model!r}: {e}\n"
                f"Is `ollama serve` running and `ollama pull {model}` done?"
            )
    raise ValueError(f"unknown provider: {provider!r} -- edit embed() to add it")


# ------------------------------------------------------------------ math ------
def cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


def rank_notes(qvec, note_vecs: dict[str, list[float]]) -> list[str]:
    """Return note ids sorted by descending cosine similarity to the query."""
    return [
        nid
        for nid, _ in sorted(
            ((nid, cosine(qvec, v)) for nid, v in note_vecs.items()),
            key=lambda t: t[1],
            reverse=True,
        )
    ]


def recall_at_k(ranked: list[str], truth: str, k: int) -> int:
    return 1 if truth in ranked[:k] else 0


def reciprocal_rank(ranked: list[str], truth: str, k: int) -> float:
    for i, nid in enumerate(ranked[:k], start=1):
        if nid == truth:
            return 1.0 / i
    return 0.0


# ------------------------------------------------------------------ corpus ----
def load_corpus(corpus_dir: Path, exts=(".md", ".txt")) -> dict[str, str]:
    """note_id (path relative to corpus_dir) -> full text. note_id is what the
    queries file's `answer_note` must match."""
    notes: dict[str, str] = {}
    for p in sorted(corpus_dir.rglob("*")):
        if p.is_file() and p.suffix.lower() in exts:
            rel = str(p.relative_to(corpus_dir))
            try:
                notes[rel] = p.read_text(encoding="utf-8", errors="ignore")
            except OSError as e:
                print(f"  ! skipping {rel}: {e}", file=sys.stderr)
    return notes


def load_queries(path: Path) -> list[dict]:
    if path.suffix.lower() == ".json":
        data = json.loads(path.read_text())
    else:  # CSV with columns question,answer_note
        data = list(csv.DictReader(path.open()))
    out = []
    for row in data:
        q = (row.get("question") or "").strip()
        a = (row.get("answer_note") or "").strip()
        if q and a:
            out.append({"question": q, "answer_note": a})
    return out


# ------------------------------------------------------------------ eval ------
def evaluate(
    model: str, provider: str, notes: dict[str, str], queries: list[dict], k: int
) -> dict:
    # 1) embed every note (this is the boring, load-bearing step: real notes)
    note_vecs: dict[str, list[float]] = {}
    for nid, text in notes.items():
        note_vecs[nid] = embed(model, text, provider)
    dim = len(next(iter(note_vecs.values()))) if note_vecs else 0

    # 2) embed each question, rank notes, score
    r1 = r5 = r10 = 0
    mrr = 0.0
    latencies: list[float] = []
    missing = 0
    for q in queries:
        truth = q["answer_note"]
        if truth not in note_vecs:
            missing += 1  # ground-truth note not found in corpus -> counts as a miss
        t0 = time.perf_counter()
        qvec = embed(model, q["question"], provider)
        ranked = rank_notes(qvec, note_vecs)
        latencies.append((time.perf_counter() - t0) * 1000.0)
        r1 += recall_at_k(ranked, truth, 1)
        r5 += recall_at_k(ranked, truth, 5)
        r10 += recall_at_k(ranked, truth, min(10, k))
        mrr += reciprocal_rank(ranked, truth, min(10, k))
    n = len(queries) or 1
    if missing:
        print(
            f"  ! {missing} query answer_note(s) not present in --corpus "
            f"(counted as misses -- check your paths)",
            file=sys.stderr,
        )
    return {
        "name": model,
        "provider": provider,
        "model": model,
        "dim": dim,
        "recall@1": r1 / n,
        "recall@5": r5 / n,
        "recall@10": r10 / n,
        "mrr@10": mrr / n,
        "q_latency_ms": round(sum(latencies) / len(latencies), 1) if latencies else 0.0,
        "approx_cost_usd": 0.0,  # local models are free; set from provider pricing otherwise
    }


# ------------------------------------------------------------------ cli -------
COLUMNS = [
    "name",
    "provider",
    "model",
    "dim",
    "recall@1",
    "recall@5",
    "recall@10",
    "mrr@10",
    "q_latency_ms",
    "approx_cost_usd",
]


def main() -> None:
    ap = argparse.ArgumentParser(description="Embedding eval on your own vault.")
    ap.add_argument("--corpus", type=Path, help="directory of .md/.txt notes")
    ap.add_argument(
        "--queries", type=Path, help="JSON or CSV of {question, answer_note}"
    )
    ap.add_argument("--models", type=str, help="comma-separated model names")
    ap.add_argument(
        "--provider", default="ollama", help="embed backend (default: ollama)"
    )
    ap.add_argument("--k", type=int, default=10, help="top-k cutoff (default 10)")
    ap.add_argument("--out", type=Path, default=Path("results.csv"))
    ap.add_argument(
        "--demo", action="store_true", help="explain the queries format and exit"
    )
    args = ap.parse_args()

    if args.demo or not (args.corpus and args.queries and args.models):
        print(__doc__)
        print(
            "Nothing to run: pass --corpus, --queries and --models. "
            "See the QUERIES FILE FORMAT block above."
        )
        return

    notes = load_corpus(args.corpus)
    if not notes:
        sys.exit(f"No .md/.txt notes under {args.corpus}")
    queries = load_queries(args.queries)
    if not queries:
        sys.exit(f"No usable questions in {args.queries}")
    print(
        f"Corpus: {len(notes)} notes | Queries: {len(queries)} | Models: {args.models}"
    )
    if len(notes) < 200:
        print(
            "  ! WARNING: fewer than 200 notes. Small sets saturate -- the "
            "ranking you get here may not survive on your real pile.",
            file=sys.stderr,
        )

    rows = []
    for model in [m.strip() for m in args.models.split(",") if m.strip()]:
        print(f"\n== {model} ==")
        row = evaluate(model, args.provider, notes, queries, args.k)
        rows.append(row)
        print(
            f"  recall@1={row['recall@1']:.3f}  recall@5={row['recall@5']:.3f}  "
            f"recall@10={row['recall@10']:.3f}  mrr@10={row['mrr@10']:.3f}  "
            f"{row['q_latency_ms']}ms/q"
        )

    # sort best-first by recall@1 then mrr@10, write CSV, print table
    rows.sort(key=lambda r: (r["recall@1"], r["mrr@10"]), reverse=True)
    with args.out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"\nWrote {args.out}")
    print("\nRANKING (best first):")
    for r in rows:
        print(
            f"  {r['name']:<28} r@1={r['recall@1']:.2f}  r@5={r['recall@5']:.2f}  "
            f"mrr@10={r['mrr@10']:.2f}"
        )
    print(
        "\nRun it again after your vault grows. The boring input -- more of your "
        "own real data -- is what makes the decision trustworthy."
    )


if __name__ == "__main__":
    main()
