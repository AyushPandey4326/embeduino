"""CLI: python -m embeduino ingest | ask \"...\""""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional

import click
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from embeduino.chunker import chunk_many
from embeduino.config import PROJECT_ROOT, get_settings
from embeduino.embeddings import get_embedder
from embeduino.loader import fetch_arduino_pages, load_local_corpus
from embeduino.rag import ask as rag_ask
from embeduino.store import VectorStore

console = Console()


def _setup_logging(verbose: bool) -> None:
    level = logging.INFO if verbose else logging.WARNING
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
    logging.getLogger("huggingface_hub").setLevel(logging.WARNING)
    logging.getLogger("chromadb").setLevel(logging.WARNING)


@click.group()
@click.option("-v", "--verbose", is_flag=True, help="Debug logging")
@click.pass_context
def main(ctx: click.Context, verbose: bool) -> None:
    """Embeduino — niche-docs RAG for Arduino / embedded documentation."""
    ctx.ensure_object(dict)
    ctx.obj["verbose"] = verbose
    _setup_logging(verbose)


@main.command("ingest")
@click.option(
    "--data-dir",
    type=click.Path(path_type=Path, exists=True, file_okay=False),
    default=None,
    help="Corpus directory (default: data/arduino_docs)",
)
@click.option(
    "--strategy",
    type=click.Choice(["structure", "fixed"], case_sensitive=False),
    default="structure",
    show_default=True,
    help="Chunking strategy",
)
@click.option("--fetch", is_flag=True, help="Also try fetching docs.arduino.cc")
@click.option("--reset", is_flag=True, help="Clear the collection before ingest")
@click.pass_context
def ingest_cmd(
    ctx: click.Context,
    data_dir: Optional[Path],
    strategy: str,
    fetch: bool,
    reset: bool,
) -> None:
    """Ingest local (and optional live) docs into Chroma."""
    settings = get_settings()
    data_dir = data_dir or settings.data_dir
    console.print(f"[bold]Embeduino ingest[/] strategy={strategy} data={data_dir}")

    docs = load_local_corpus(data_dir)
    if fetch:
        live = fetch_arduino_pages()
        console.print(f"Fetched {len(live)} live page(s)")
        docs.extend(live)

    if not docs:
        console.print("[red]No documents found.[/]")
        sys.exit(1)

    if strategy == "structure":
        chunks = chunk_many(docs, strategy="structure", max_chars=900, overlap_words=40)
    else:
        chunks = chunk_many(docs, strategy="fixed", size=500, overlap=50)

    console.print(f"Documents: {len(docs)}  Chunks: {len(chunks)}")
    by_type: dict[str, int] = {}
    for c in chunks:
        by_type[c.chunk_type] = by_type.get(c.chunk_type, 0) + 1
    console.print(f"Chunk types: {by_type}")

    embedder = get_embedder(settings)
    store = VectorStore(settings.chroma_path, settings.collection, embedder)
    if reset:
        store.reset()
    n = store.upsert_chunks(chunks)
    console.print(
        Panel.fit(
            f"Upserted [green]{n}[/] chunks → {settings.chroma_path} "
            f"(collection={settings.collection}, total={store.count()})",
            title="Ingest complete",
        )
    )


@main.command("ask")
@click.argument("question")
@click.option("--top-k", type=int, default=None)
@click.option("--min-score", type=float, default=None)
@click.option(
    "--web",
    type=click.Choice(["auto", "always", "off"], case_sensitive=False),
    default=None,
    help="Web search mode (default: from config or 'auto')",
)
@click.pass_context
def ask_cmd(
    ctx: click.Context,
    question: str,
    top_k: Optional[int],
    min_score: Optional[float],
    web: Optional[str],
) -> None:
    """Ask a question; print answer, citations, and retrieved chunk previews."""
    settings = get_settings()
    # Allow CLI overrides without mutating frozen dataclass fields permanently
    if top_k is not None or min_score is not None:
        from dataclasses import replace

        settings = replace(
            settings,
            top_k=top_k if top_k is not None else settings.top_k,
            min_score=min_score if min_score is not None else settings.min_score,
        )

    embedder = get_embedder(settings)
    store = VectorStore(settings.chroma_path, settings.collection, embedder)
    if store.count() == 0:
        console.print(
            "[red]Vector store is empty. Run:[/] python -m embeduino ingest --reset"
        )
        sys.exit(1)

    web_mode_override = web.lower() if web else None
    result = rag_ask(question, store, settings, web_mode_override=web_mode_override)

    style = "yellow" if result.weak_retrieval else "green"
    console.print(Panel(result.answer, title="Answer", border_style=style))
    
    if result.web_search_used:
        if result.web_status == "cache_hit":
            console.print("[green]Web search served from cache (0 SerpApi credits)[/]")
        elif result.web_status == "api_success" and result.web_credit_used:
            console.print("[yellow]Web search used (1 SerpApi credit)[/]")
        elif result.web_status == "timeout":
            console.print("[red]Web search timed out (took >90s, 1 credit used)[/]")
        elif result.web_status == "api_failed":
            console.print("[red]Web search failed (error)[/]")
        elif result.web_status == "no_key":
            console.print("[dim]Web search skipped (no API key)[/]")
        elif result.web_status == "max_calls":
            console.print("[dim]Web search skipped (max calls reached)[/]")
        else:
            console.print("[dim]Web search: unknown status[/]")
    else:
        console.print("[dim]Web search: not triggered[/]")

    console.print("\n[bold]Citations:[/]")
    for i, c in enumerate(result.citations, 1):
        origin_badge = "[cyan][web][/]" if c.get("origin") == "web" else "[dim][local][/]"
        rrf_str = f" rrf={c['rrf_score']:.3f}" if c.get("rrf_score") else ""
        console.print(f"{i}. {origin_badge} [cyan]{c['id']}[/] score={c['score']:.3f}{rrf_str}")
        if c.get("url"):
            console.print(f"   URL: {c['url']}")
        console.print(f"   Source: {c.get('source', '')}")
        console.print(f"   Heading: {c.get('heading_path', '')}")
        console.print(f"   Preview: {(c.get('preview') or '')[:80]}...")
        console.print("")

    console.print("[dim]Retrieved chunk ids:[/]")
    for h in result.retrieved:
        meta = h.get("metadata") or {}
        console.print(
            f"  • {h['id']}  score={h['score']:.3f}  "
            f"type={meta.get('chunk_type')}  "
            f"preview={(h.get('text') or '')[:80]!r}..."
        )


@main.command("eval")
@click.option(
    "--questions",
    type=click.Path(path_type=Path, exists=True),
    default=None,
    help="JSONL of {question, expect_contains?}; default scripts/golden_qa.jsonl",
)
@click.option(
    "--web",
    type=click.Choice(["auto", "always", "off"], case_sensitive=False),
    default="off",
    help="Web search mode for eval (default: off to avoid credits)",
)
@click.pass_context
def eval_cmd(ctx: click.Context, questions: Optional[Path], web: str) -> None:
    """Run minimal golden Q&A checks against the local store."""
    import json

    settings = get_settings()
    path = questions or (PROJECT_ROOT / "scripts" / "golden_qa.jsonl")
    if not path.exists():
        console.print(f"[red]Missing golden file:[/] {path}")
        sys.exit(1)

    embedder = get_embedder(settings)
    store = VectorStore(settings.chroma_path, settings.collection, embedder)
    if store.count() == 0:
        console.print("[red]Empty store — ingest first.[/]")
        sys.exit(1)

    passed = failed = 0
    web_mode = web.lower()
    
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        item = json.loads(line)
        q = item["question"]
        expect = item.get("expect_contains", [])
        expect_dont_know = item.get("expect_dont_know", False)
        expect_web = item.get("expect_web", False)
        
        # Override web mode if question expects web
        question_web_mode = "always" if expect_web else web_mode
        
        result = rag_ask(q, store, settings, web_mode_override=question_web_mode)
        ok = True
        if expect_dont_know:
            ok = result.weak_retrieval or "don't know" in result.answer.lower()
        else:
            ans_l = result.answer.lower()
            ok = (not result.weak_retrieval) and all(
                e.lower() in ans_l or any(e.lower() in (h.get("text") or "").lower() for h in result.retrieved)
                for e in expect
            )
        status = "[green]PASS[/]" if ok else "[red]FAIL[/]"
        web_badge = "[cyan][web][/]" if result.web_search_used else ""
        console.print(f"{status} {web_badge} {q}")
        if ok:
            passed += 1
        else:
            failed += 1
            console.print(f"         answer preview: {result.answer[:160]!r}")
    console.print(f"\nResults: {passed} passed, {failed} failed")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
