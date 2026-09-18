# Embeduino

**Niche-docs RAG** for Arduino / embedded language-reference documentation.



> Local-only demo. Do not push secrets; `.chroma/` and `.env` are gitignored.

---

## Problem

Generic fixed-size chunking cuts through Arduino reference pages mid-section (e.g. splitting `## Syntax` from its code fence), which hurts retrieval precision for API-style docs. Portfolio RAG demos also often fail open: they invent answers when nothing relevant was retrieved.

## Approach

1. **Offline-first corpus** under `data/arduino_docs/` (markdown + one HTML page) so ingest works without scraping.
2. **Structure-aware chunking**: split on headings, keep fenced code as dedicated chunks, small word overlap between oversized prose splits. Optional **fixed-size** baseline for comparison (`--strategy fixed`).
3. **Local embeddings** via `sentence-transformers` (`all-MiniLM-L6-v2`); OpenAI embeddings/chat via `.env` if desired.
4. **Chroma** persistence under `.chroma/`.
5. **Weak-retrieval gate**: if top similarity &lt; `EMBEDUINO_MIN_SCORE`, refuse with “I don’t know”.
6. **Logging** of retrieved chunk ids, scores, and previews on every `ask`.

## Demo

```bash
cd /workspace/embeduino
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"

# Ingest sample corpus (structure-aware)
python -m embeduino ingest --reset

# Ask with citations
python -m embeduino ask "What does digitalWrite do?"

# Golden Q&A smoke eval
python -m embeduino eval
```

Optional live fetch (network-dependent; offline corpus is enough):

```bash
python -m embeduino ingest --reset --fetch
```

Compare chunkers:

```bash
python -m embeduino ingest --reset --strategy structure
python -m embeduino ingest --reset --strategy fixed
```

## Install

Requirements: Python 3.10+, ~1GB disk for the embedding model on first run.

```bash
cd /workspace/embeduino
python -m venv .venv && source .venv/bin/activate
pip install -U pip
pip install -e ".[dev]"
# or: pip install -r requirements.txt && pip install -e .
cp .env.example .env   # edit only if using OpenAI
```

### Configuration (`.env`)

| Variable | Default | Meaning |
|----------|---------|---------|
| `EMBEDUINO_EMBEDDING_BACKEND` | `local` | `local` or `openai` |
| `EMBEDUINO_LOCAL_MODEL` | `all-MiniLM-L6-v2` | sentence-transformers model |
| `OPENAI_API_KEY` | _(empty)_ | Required for OpenAI embed/gen |
| `EMBEDUINO_GENERATION_BACKEND` | `local` | `local` (extractive) or `openai` |
| `EMBEDUINO_TOP_K` | `4` | Retrieved neighbors |
| `EMBEDUINO_MIN_SCORE` | `0.30` | Cosine-similarity floor |
| `EMBEDUINO_CHROMA_PATH` | `.chroma` | Persist directory |

Never commit `.env` or API keys.

## Project layout

```
embeduino/
├── data/arduino_docs/     # offline sample corpus
├── embeduino/
│   ├── chunker.py         # structure-aware + fixed-size
│   ├── loader.py          # local + optional docs.arduino.cc fetch
│   ├── embeddings.py      # local / OpenAI
│   ├── store.py           # Chroma wrapper
│   ├── rag.py             # ask + weak-retrieval gate
│   └── cli.py             # ingest / ask / eval
├── scripts/golden_qa.jsonl
├── tests/test_chunker.py
├── .env.example
├── LICENSE                # MIT
└── pyproject.toml
```

## Eval

Minimal golden set in `scripts/golden_qa.jsonl` (known API facts + one out-of-domain “don’t know” case):

```bash
python -m embeduino ingest --reset
python -m embeduino eval
pytest -q
```

Expected: most factual questions pass via retrieved text; the flux-capacitor question should trigger weak retrieval / “I don’t know”.

## License

MIT — see [LICENSE](LICENSE).
