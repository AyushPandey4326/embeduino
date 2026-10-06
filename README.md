# Embeduino

**Niche-docs RAG** for Arduino / embedded language-reference documentation with live web search.

Ingest a small offline docs corpus into a local vector store, retrieve with structure-aware chunks using hybrid search (vector + BM25), and answer with **citations** — or an honest **"I don't know"** when retrieval is weak. When local documentation doesn't cover a topic, Embeduino can search trusted web sources in real-time to stay current.

> Local-only demo. Do not push secrets; `.chroma/`, `.serp_cache/` and `.env` are gitignored.

---

## What's New: SerpApi Hackathon 2026 Enhancements

**This project existed before September 1, 2026.** The following features were added specifically for the [SerpApi India Hackathon 2026](https://serpapi.github.io/serpapi-india-hackathon-2026/):

### 🔥 Latest: Fixed RRF Structural Bias (Critical)

**Third Windows 11 real-key test** showed async mode working (9.8s, "Searching the web..." displayed, 1 credit used), but **answer was still "I don't know" with zero web chunks in citations**. Top 4 retrieved were all local (analog_read, digital_write, pin_mode, pwm_analog_write) each with RRF score ~0.032.

**Root Cause**: Local chunks appeared in **BOTH** `vector_ranked` and `bm25_ranked` lists → combined RRF ≈ 2/(k+rank) ≈ 0.032. Web chunks appeared in **ONLY** `web_ranked` → max RRF = 1/(k+1) ≈ 0.016. With `top_k=4`, web chunks could **never** compete.

**Fix**: Score web chunks through the same rankers:
1. Combine local+web **before** reranking → web chunks appear in `vector_ranked`
2. Add web chunks to BM25 corpus → web chunks appear in `bm25_ranked`
3. RRF now sees web in both lists → fair 2× multiplier for all

**Result**: Web chunks now earn competitive RRF scores (>0.025 typical), reach citations, extractive answers use web as primary with URLs.

**Validation**: 5 new integration tests (31 total pass) with realistic local+web fusion paths.

---

### Live Web Search Integration

The offline corpus goes stale as new Arduino boards, libraries, and core versions are released. Questions about topics not in the local documentation now trigger a **live SerpApi Google Search** restricted to trusted embedded sources (docs.arduino.cc, github.com, forum.arduino.cc).

- **BM25 keyword retrieval** over local chunks alongside vector search
- **Reciprocal Rank Fusion (RRF)** merges vector, BM25, and web results into a single ranked list
- **SerpApi Google Search** provides live results as retrievable chunks with URL citations
- **Smart triggering**: web search fires automatically when local retrieval is weak OR when the question mentions boards/libraries/versions not in the local corpus
- **Credit safety**: on-disk cache (gitignored), per-run call cap, clear output showing whether a SerpApi credit was used
- **Citation grounding**: OpenAI-generated answers validate that every cited chunk ID exists in the context

### Why Live Search?

The local corpus is 8 reference pages covering core Arduino functions (digitalWrite, pinMode, analogRead, etc.). Newer boards like the **Arduino UNO R4 WiFi**, **Nano ESP32**, or **Nano 33 BLE Sense** aren't documented locally. Without SerpApi, those questions fail with "I don't know". With SerpApi, Embeduino retrieves current information from trusted sources, fuses it with local results, and provides grounded answers with web citations.

**SerpApi is a genuine, central part of the pipeline** — not a cosmetic call. The system depends on live search data to answer questions the local corpus cannot cover.

---

## How SerpApi Is Used

### Engine and Query

- **Engine**: Google Search API (`https://serpapi.com/search?engine=google`)
- **Async mode**: Uses `async=true` to initiate search, then polls `https://serpapi.com/searches/{id}.json` every 2.5s until `status=Success` (polling does not cost credits)
- **Query shape**: Keywords extracted from question + `site:docs.arduino.cc` (prefer docs first, fallback to no filter if 0 results)
- **Example**: `"pin specifications arduino uno R4 wifi site:docs.arduino.cc"`
- **Parameters**: `hl=en`, `gl=us` (no `num` parameter; results sliced locally)
- **Fields used**: `organic_results[].position`, `.title`, `.link`, `.snippet`
- **Timeout**: Configurable via `SERPAPI_TIMEOUT_S` (default 90s, includes polling time)

### Trigger Logic

Web search is triggered when:

1. **Mode is "always"**: every question uses web search
2. **Mode is "auto"** (default): search fires if local retrieval is weak OR the question mentions a board, library, or version not in the local corpus (e.g., "UNO R4", "ESP32", "Nano 33", "library")
3. **Mode is "off"**: web search is disabled

### Result Processing

1. SerpApi returns up to `EMBEDUINO_WEB_NUM` organic results
2. Each result becomes a retrievable chunk: `snippet` as text, `link` as URL, `title` as heading
3. Web chunks are labeled `chunk_type="web"` and carry `url` metadata

### Fusion

**Reciprocal Rank Fusion (RRF)** merges ranked lists with fair scoring:

1. **Vector search**: Raw vector results + web chunks combined, then reranked with lexical overlap and heading boosts
2. **BM25**: Keyword retrieval over local chunks + web chunks combined

**Key**: Web chunks are scored through **both** rankers (not isolated in a third list), ensuring fair competition. Both local and web chunks appear in both lists → equal 2× RRF multiplier.

RRF score: `sum over lists of 1/(k + rank_in_list)`, where `k=60` (configurable). Top-k fused results used for generation.

### Citation and Honesty

- After fusion, the weak-retrieval gate still applies: if confidence is too low, return "I don't know"
- OpenAI-generated answers validate that every cited chunk ID exists in the provided context; if validation fails, fall back to extractive answers
- Citations include `origin` (local/web) and `url` for web chunks

---

## Architecture

```
┌───────────────────────────────────────────────────────────────┐
│                        User Question                          │
└───────────────┬───────────────────────────────────────────────┘
                │
        ┌───────▼────────┐
        │   Embeddings   │
        └────────┬───────┘
                 │
      ┌──────────▼───────────┐
      │  Parallel Retrieval  │
      ├──────────────────────┤
      │ • Vector (Chroma)    │
      │ • BM25 (local)       │
      │ • Web (SerpApi*)     │ * if triggered
      └──────────┬───────────┘
                 │
        ┌────────▼──────────┐
        │  RRF Fusion       │
        │  (k=60, top-k=4)  │
        └────────┬──────────┘
                 │
       ┌─────────▼──────────┐
       │  Weak Gate         │
       │  (honesty check)   │
       └─────────┬──────────┘
                 │
        ┌────────▼──────────┐
        │  Answer Generation│
        │  + Citation Check │
        └────────┬──────────┘
                 │
         ┌───────▼────────┐
         │  Answer + URLs │
         └────────────────┘
```

---

## Problem

Generic fixed-size chunking cuts through Arduino reference pages mid-section (e.g. splitting `## Syntax` from its code fence), which hurts retrieval precision for API-style docs. Many RAG demos also fail open: they invent answers when nothing relevant was retrieved. Additionally, offline-only systems go stale as new hardware and libraries are released.

## Approach

1. **Offline-first corpus** under `data/arduino_docs/` (markdown + one HTML page) so ingest works without scraping
2. **Structure-aware chunking**: split on headings, keep fenced code as dedicated chunks, small word overlap between oversized prose splits. Optional **fixed-size** baseline for comparison (`--strategy fixed`)
3. **Hybrid retrieval**:
   - **Local embeddings** via `sentence-transformers` (`all-MiniLM-L6-v2`); OpenAI embeddings/chat via `.env` if desired
   - **BM25** keyword search over local chunks
   - **Live web search** via SerpApi when local results are insufficient
4. **Chroma** persistence under `.chroma/`
5. **Reciprocal Rank Fusion** merges vector, BM25, and web results
6. **Weak-retrieval gate**: if top similarity < `EMBEDUINO_MIN_SCORE` after fusion, refuse with "I don't know"
7. **Logging** of retrieved chunk ids, scores, and previews on every `ask`

---

## Demo

```bash
cd /workspace
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"

# Ingest sample corpus (structure-aware)
python3 -m embeduino ingest --reset

# Ask with citations (local only)
python3 -m embeduino ask "What does digitalWrite do?" --web off

# Ask with web search (requires SERPAPI_API_KEY in .env)
python3 -m embeduino ask "What are the pin specifications for Arduino UNO R4 WiFi?" --web auto

# Verbose mode (show INFO logs for debugging)
python3 -m embeduino ask "..." --verbose

# Golden Q&A smoke eval
python3 -m embeduino eval --web off
```

Optional live fetch (network-dependent; offline corpus is enough):

```bash
python3 -m embeduino ingest --reset --fetch
```

Compare chunkers:

```bash
python3 -m embeduino ingest --reset --strategy structure
python3 -m embeduino ingest --reset --strategy fixed
```

---

## Install

Requirements: Python 3.10+, ~1GB disk for the embedding model on first run.

**Windows users**: Set `PYTHONUTF8=1` in your environment or use `chcp 65001` before running to avoid encoding errors with web search results containing non-ASCII characters.

```bash
cd /workspace
python3 -m venv .venv && source .venv/bin/activate
pip install -U pip
pip install -e ".[dev]"
# or: pip install -r requirements.txt && pip install -e .
cp .env.example .env
# Edit .env: add your SERPAPI_API_KEY (free tier: 250 searches/month)
```

### Get a Free SerpApi Key

1. Sign up at [https://serpapi.com/](https://serpapi.com/)
2. Get 250 free searches per month (no credit card required)
3. Copy your API key to `.env`: `SERPAPI_API_KEY=your_key_here`

---

## Configuration

Edit `.env` or set environment variables:

| Variable | Default | Meaning |
|----------|---------|---------|
| `EMBEDUINO_EMBEDDING_BACKEND` | `local` | `local` (sentence-transformers) or `openai` |
| `EMBEDUINO_LOCAL_MODEL` | `all-MiniLM-L6-v2` | sentence-transformers model |
| `OPENAI_API_KEY` | _(empty)_ | Required for OpenAI embed/gen |
| `EMBEDUINO_GENERATION_BACKEND` | `local` | `local` (extractive) or `openai` |
| `EMBEDUINO_TOP_K` | `4` | Retrieved neighbors after fusion |
| `EMBEDUINO_MIN_SCORE` | `0.30` | Cosine-similarity floor |
| `EMBEDUINO_CHROMA_PATH` | `.chroma` | Persist directory |
| **`SERPAPI_API_KEY`** | _(empty)_ | **SerpApi key (required for web search)** |
| **`EMBEDUINO_WEB_MODE`** | **`auto`** | **Web search mode: `auto`, `always`, `off`** |
| **`EMBEDUINO_WEB_NUM`** | **`5`** | **Number of SerpApi organic results** |
| **`EMBEDUINO_WEB_SITES`** | **`docs.arduino.cc github.com forum.arduino.cc`** | **Trusted sites (space-separated)** |
| **`EMBEDUINO_SERP_CACHE`** | **`.serp_cache`** | **On-disk cache directory (gitignored)** |
| **`EMBEDUINO_SERP_MAX_CALLS`** | **`10`** | **Max SerpApi calls per run (credit cap)** |
| **`SERPAPI_TIMEOUT_S`** | **`90`** | **SerpApi search timeout (seconds, includes async polling)** |
| **`EMBEDUINO_RRF_K`** | **`60`** | **Reciprocal Rank Fusion k parameter** |

Never commit `.env`, API keys, or cache directories.

---

## Project Layout

```
embeduino/
├── data/arduino_docs/     # offline sample corpus
├── embeduino/
│   ├── chunker.py         # structure-aware + fixed-size
│   ├── loader.py          # local + optional docs.arduino.cc fetch
│   ├── embeddings.py      # local / OpenAI
│   ├── store.py           # Chroma wrapper + BM25 chunk retrieval
│   ├── fusion.py          # BM25 + RRF
│   ├── web_search.py      # SerpApi client + caching + trigger logic
│   ├── rag.py             # ask + weak-retrieval gate + citation validation
│   └── cli.py             # ingest / ask / eval
├── scripts/golden_qa.jsonl
├── tests/
│   ├── test_chunker.py
│   ├── test_fusion.py     # BM25 + RRF tests
│   ├── test_web_search.py # SerpApi parsing, caching, trigger tests
│   ├── test_rag_web_integration.py # 5 integration tests: realistic local+web fusion
│   └── fixtures/
│       └── serpapi_uno_r4_response.json
├── docs/demo.md           # 3-minute demo script
├── .env.example
├── LICENSE                # MIT
└── pyproject.toml
```

---

## SerpApi Credit Usage

### Free Tier

- **250 searches/month** on the free plan (no credit card)
- Cached searches (same query + params within 1 hour) are free via SerpApi's built-in cache
- Embeduino's on-disk cache (`.serp_cache/`) makes repeats free forever

### Credit Safety

1. **On-disk cache**: every SerpApi response is saved locally; repeat queries never hit the API
2. **Per-run max-calls cap** (`EMBEDUINO_SERP_MAX_CALLS=10`): prevents runaway loops
3. **Clear CLI output**: shows "Web search used (1 SerpApi credit)" or "served from cache (0 credits)"

### Typical Usage

- **Development**: ~60 calls (mostly cached after first run)
- **Eval runs**: ~25 calls (cached after first)
- **Demo rehearsals**: ~15 calls (cached)
- **Buffer**: ~150 calls

Well within the 250/month free tier.

---

## Eval and Tests

### Golden Q&A

Minimal golden set in `scripts/golden_qa.jsonl` (known API facts + one out-of-domain "don't know" case + web-requiring questions):

```bash
python3 -m embeduino ingest --reset
python3 -m embeduino eval --web off   # local questions only
python3 -m embeduino eval --web auto  # includes web questions (uses credits)
```

Expected: most factual questions pass via retrieved text; the flux-capacitor question should trigger weak retrieval / "I don't know"; web questions require SerpApi.

### Unit Tests

All tests use fixtures and mocks; **pytest never needs a SerpApi key or network access**:

```bash
export PATH="/home/ubuntu/.local/bin:$PATH"  # if pytest is in ~/.local/bin
python3 -m pytest tests/ -v
```

Tests cover:
- Structure-aware chunking
- BM25 ranking
- Reciprocal Rank Fusion
- SerpApi result parsing (using saved JSON fixture)
- Cache read/write
- Web search trigger logic

---

## Limitations

1. **Web snippets can be incomplete or misleading**: the system only has snippet text unless full page fetch is added
2. **Trusted-site filter**: restricts results to configured domains; may miss relevant content elsewhere
3. **Honest refusal**: the weak-retrieval gate still fires after fusion, so some questions return "I don't know" even with web results
4. **No real-time board/library detection**: trigger logic uses a static keyword list; new topics not in the list won't auto-trigger web search
5. **Free tier limits**: 250 searches/month; exceed that and API calls fail
6. **Windows encoding**: Run with `PYTHONUTF8=1` or `chcp 65001` to handle non-ASCII characters in web results


---

## License

MIT — see [LICENSE](LICENSE).
