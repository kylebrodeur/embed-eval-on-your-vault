# embed-eval-on-your-vault

**Pick the embedding model from YOUR corpus, not someone else's leaderboard.**

A single-file Python eval harness for embedding models. Run it on your own vault
of notes, score multiple candidates against a hand-built ground-truth set, and
let the ranking pick the model.

## Why this exists

A small benchmark is a saturated benchmark. With ~50 notes, the right answer
is near the top no matter which model you use. There are not enough documents
for a wrong answer to hide. A saturated benchmark does not return an error; it
returns a confident, specific, useless ranking.

This script lets you run the same A/B on your real pile:

- It loads every `.md` / `.txt` file in your vault.
- It embeds them with each candidate model.
- It scores each model against a hand-built ground-truth set of
  `{question, answer_note}` pairs.
- It writes a CSV you can diff across runs as your vault grows.

The source experiment that motivated this is in the [blog post](https://kylebrodeur.substack.com):
seven models on a 54-note toy slice (all looked great), then the same harness
on 995 notes (the ranking inverted). The "winner" sank. The mid-pack model
took first. The default the system had been quietly using dropped near the
bottom.

## Default backend: Ollama

Local, free. Start it with `ollama serve` and pull the models you want to test:

```bash
ollama serve &
ollama pull embeddinggemma
ollama pull nomic-embed-text
ollama pull bge-large
ollama pull all-minilm
```

Swap the `embed()` function in the script to hit any other provider
(OpenAI, Cohere, a Modal endpoint, etc.). The signature is:

```python
def embed(model: str, text: str, provider: str = "ollama") -> list[float]:
    ...
```

## Usage

```bash
# 1. Make a queries file (JSON list of {question, answer_note})
cat > queries.json << 'QUERIES'
[
  {"question": "how do I prime the pump tubing?", "answer_note": "iot-rig/priming.md"},
  {"question": "what model won the eval?",        "answer_note": "eval/findings.md"}
]
QUERIES

# 2. Run the eval
python3 embed-eval-on-your-vault.py \
    --corpus ~/path/to/your/vault \
    --queries queries.json \
    --models embeddinggemma,nomic-embed-text,bge-large,all-minilm \
    --k 10 \
    --out results.csv
```

`answer_note` is a path relative to `--corpus`. The CSV output is:

```
name,provider,model,dim,recall@1,recall@5,recall@10,mrr@10,q_latency_ms,approx_cost_usd
```

## Why the harness is a single file

It has no dependencies beyond `python3`. Drop it next to your notes and run
it. Copy it, version it, fork it. The point is to make the eval so cheap you
will actually run it again next time your vault grows.

## The eval harness itself

`embed-eval-on-your-vault.py` is the whole thing. Read the docstring at the
top for the queries file format and the embed-backend swap pattern.

## Companion dataset

The CSVs that produced the
[blog post results](https://huggingface.co/datasets/kylebrodeur/embedding-eval-results)
live on HuggingFace. Use them to spot-check the harness against a known
ranking before you trust it on your own corpus.

## License

MIT. Use it, fork it, ship it.
