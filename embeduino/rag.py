"""Retrieve-and-generate with citations and weak-retrieval honesty."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Set

from embeduino.config import Settings
from embeduino.store import VectorStore

logger = logging.getLogger(__name__)

_STOP = {
    "a", "an", "the", "is", "are", "was", "were", "be", "to", "of", "in", "on",
    "for", "and", "or", "what", "how", "does", "do", "did", "i", "me", "my",
    "with", "from", "it", "this", "that", "at", "by", "as", "if", "when",
    "why", "which", "who", "can", "could", "should", "would", "about",
}


@dataclass
class AskResult:
    answer: str
    citations: List[Dict[str, Any]] = field(default_factory=list)
    retrieved: List[Dict[str, Any]] = field(default_factory=list)
    weak_retrieval: bool = False


def _tokens(text: str) -> Set[str]:
    # Keep camelCase / dotted API identifiers as whole tokens before lowercasing
    raw = re.findall(r"[A-Za-z_][A-Za-z0-9_]*(?:\(\))?", text)
    out = set()
    for t in raw:
        tl = t.lower().rstrip("()")
        if len(tl) > 1 and tl not in _STOP:
            out.add(tl)
    out |= {
        t
        for t in re.findall(r"[a-z0-9_]+", text.lower())
        if len(t) > 1 and t not in _STOP
    }
    return out


def _api_mentions(question: str) -> Set[str]:
    """Likely API / symbol names in the question (camelCase, Xxx.write, etc.)."""
    names = set()
    for m in re.findall(r"\b([A-Za-z_][A-Za-z0-9_]{2,})\b", question):
        if m.lower() in _STOP:
            continue
        if m[0].islower() and any(c.isupper() for c in m[1:]):
            names.add(m.lower())
        elif m.lower() in {
            "digitalwrite", "digitalread", "pinmode", "analogread", "analogwrite",
            "millis", "micros", "delay", "serial", "setup", "loop",
        }:
            names.add(m.lower())
        elif "write" in m.lower() or "read" in m.lower() or "mode" in m.lower():
            names.add(m.lower())
    # also catch digitalWrite-style already lowercased in question text
    q = question.lower()
    for name in (
        "digitalwrite", "digitalread", "pinmode", "analogread", "analogwrite",
        "millis", "micros", "delay", "serial.begin", "serial",
    ):
        if name.replace(".", "") in q.replace(".", "") or name in q:
            names.add(name.split(".")[0])
    return names


def _lexical_overlap(question: str, text: str) -> float:
    q = _tokens(question)
    if not q:
        return 0.0
    d = _tokens(text)
    return len(q & d) / len(q)


def _heading_boost(question: str, heading: str) -> float:
    h = (heading or "").lower()
    q = question.lower()
    boost = 0.0
    if "description" in h and any(w in q for w in ("what", "do", "does", "mean")):
        boost += 0.12
    if "parameter" in h and any(w in q for w in ("parameter", "argument", "mode", "range", "value")):
        boost += 0.12
    if "return" in h and "return" in q:
        boost += 0.10
    if "note" in h and any(w in q for w in ("why", "problem", "warn", "drawback", "issue")):
        boost += 0.10
    return boost


def _rerank(question: str, hits: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    apis = _api_mentions(question)
    scored = []
    for h in hits:
        meta = h.get("metadata") or {}
        body = h.get("text") or ""
        heading = str(meta.get("heading_path", ""))
        blob = f"{heading} {body}"
        lex = _lexical_overlap(question, blob)
        boost = _heading_boost(question, heading)
        # Strong preference when the named API appears in the heading path
        heading_l = heading.lower().replace("()", "")
        api_boost = 0.0
        for api in apis:
            if api in heading_l.replace(".", ""):
                api_boost += 0.35
            elif api in body.lower()[:120]:
                api_boost += 0.08
        combined = float(h.get("score", 0.0)) + 0.25 * lex + boost + api_boost
        item = dict(h)
        item["lex_overlap"] = lex
        item["rerank_score"] = combined
        scored.append(item)
    scored.sort(key=lambda x: x["rerank_score"], reverse=True)
    return scored


def _log_retrieved(hits: List[Dict[str, Any]]) -> None:
    for i, h in enumerate(hits, 1):
        meta = h.get("metadata") or {}
        preview = (h.get("text") or "")[:160].replace("\n", " ")
        logger.info(
            "Retrieved[%d] id=%s score=%.3f rerank=%.3f lex=%.2f source=%s heading=%s preview=%r",
            i,
            h.get("id"),
            h.get("score", 0.0),
            h.get("rerank_score", h.get("score", 0.0)),
            h.get("lex_overlap", 0.0),
            meta.get("source"),
            meta.get("heading_path"),
            preview,
        )


def _build_citations(hits: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    cites = []
    for h in hits:
        meta = h.get("metadata") or {}
        cites.append(
            {
                "id": h["id"],
                "source": meta.get("source", ""),
                "heading_path": meta.get("heading_path", ""),
                "score": round(float(h.get("score", 0.0)), 4),
                "rerank_score": round(float(h.get("rerank_score", h.get("score", 0.0))), 4),
                "preview": (h.get("text") or "")[:200],
            }
        )
    return cites


def _is_weak(question: str, hits: List[Dict[str, Any]], min_score: float) -> bool:
    if not hits:
        return True
    best = hits[0]
    if float(best.get("score", 0.0)) < min_score:
        return True
    # Out-of-domain: embedding may still fire weakly; require some lexical support
    if float(best.get("lex_overlap", 0.0)) < 0.15 and float(best.get("score", 0.0)) < 0.45:
        return True
    # Domain-ish query terms must appear somewhere in top hits
    q_terms = _tokens(question)
    content_terms = set()
    for h in hits[:3]:
        content_terms |= _tokens(h.get("text") or "")
    # Drop API-ish tokens that appear in many docs (arduino, pin, etc. alone not enough)
    distinctive = q_terms - {"arduino", "board", "sketch", "code", "function", "use", "using"}
    if distinctive and len(distinctive & content_terms) / len(distinctive) < 0.2:
        return True
    return False


def _extractive_answer(question: str, hits: List[Dict[str, Any]]) -> str:
    if not hits:
        return (
            "I don't know — no relevant documentation chunks were retrieved "
            "for this question."
        )

    apis = _api_mentions(question)
    def _api_match(h: Dict[str, Any]) -> bool:
        heading = ((h.get("metadata") or {}).get("heading_path") or "").lower().replace("()", "")
        return any(api in heading for api in apis) if apis else True

    ranked = [h for h in hits if _api_match(h)] or list(hits)
    preferred = []
    for h in ranked:
        heading = ((h.get("metadata") or {}).get("heading_path") or "").lower()
        if any(k in heading for k in ("description", "parameter", "notes", "syntax", "return")):
            preferred.append(h)
    primary = preferred[0] if preferred else ranked[0]

    # Optionally append a complementary Parameters/Returns chunk from same source
    extras_chunks = []
    src = (primary.get("metadata") or {}).get("source")
    for h in hits:
        if h is primary:
            continue
        if (h.get("metadata") or {}).get("source") != src:
            continue
        heading = ((h.get("metadata") or {}).get("heading_path") or "").lower()
        if any(k in heading for k in ("parameter", "return", "syntax", "example", "notes")):
            extras_chunks.append(h)
        if len(extras_chunks) >= 2:
            break

    def _format_chunk(h: Dict[str, Any]) -> str:
        text = h.get("text") or ""
        paras = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
        paras = [p for p in paras if not (re.match(r"^#{1,6}\s+", p) and len(p) < 60)]
        return "\n\n".join(paras[:4]) if paras else text[:700]

    meta = primary.get("metadata") or {}
    source = meta.get("source", primary.get("id", "unknown"))
    heading = meta.get("heading_path") or ""
    header = f"Based on `{source}`"
    if heading:
        header += f" ({heading})"
    header += ":"

    body = _format_chunk(primary)
    for h in extras_chunks:
        body += "\n\n" + _format_chunk(h)

    related = []
    for h in hits:
        if h is primary or h in extras_chunks:
            continue
        if len(related) >= 2:
            break
        m = h.get("metadata") or {}
        snippet = " ".join((h.get("text") or "").split())[:180]
        related.append(
            f"- [{h['id']}] {m.get('heading_path') or m.get('source')}: {snippet}"
        )

    parts = [header, "", body]
    if related:
        parts.extend(["", "Related context:", *related])
    parts.append("")
    parts.append(
        f"(Retrieved {len(hits)} chunk(s); top score={hits[0].get('score', 0):.3f}, "
        f"rerank={hits[0].get('rerank_score', hits[0].get('score', 0)):.3f})"
    )
    return "\n".join(parts)


def _openai_answer(question: str, hits: List[Dict[str, Any]], settings: Settings) -> str:
    from openai import OpenAI

    if not settings.openai_api_key:
        return _extractive_answer(question, hits)

    context_blocks = []
    for h in hits:
        meta = h.get("metadata") or {}
        context_blocks.append(
            f"[{h['id']}] source={meta.get('source')} heading={meta.get('heading_path')}\n"
            f"{h.get('text', '')}"
        )
    context = "\n\n---\n\n".join(context_blocks)
    system = (
        "You are Embeduino, a careful assistant for Arduino / embedded docs. "
        "Answer ONLY using the provided context. Cite chunk ids like [id]. "
        "If the context is insufficient, say you don't know."
    )
    user = f"Question: {question}\n\nContext:\n{context}"
    client = OpenAI(api_key=settings.openai_api_key)
    resp = client.chat.completions.create(
        model=settings.openai_chat_model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=0.1,
    )
    return (resp.choices[0].message.content or "").strip()


def ask(
    question: str,
    store: VectorStore,
    settings: Settings,
) -> AskResult:
    # Fetch a few extra neighbors for reranking
    raw = store.query(question, top_k=max(settings.top_k * 3, 12))
    hits = _rerank(question, raw)[: settings.top_k]
    _log_retrieved(hits)

    weak = _is_weak(question, hits, settings.min_score)

    if weak:
        logger.warning(
            "Weak retrieval for %r (best score=%.3f lex=%.2f min=%.3f)",
            question,
            hits[0]["score"] if hits else -1.0,
            hits[0].get("lex_overlap", 0.0) if hits else 0.0,
            settings.min_score,
        )
        return AskResult(
            answer=(
                "I don't know — retrieval confidence is too low for a grounded answer. "
                "Try rephrasing, or ingest more documentation covering this topic."
            ),
            citations=_build_citations(hits[:2]),
            retrieved=hits,
            weak_retrieval=True,
        )

    if settings.generation_backend == "openai":
        answer = _openai_answer(question, hits, settings)
    else:
        answer = _extractive_answer(question, hits)

    return AskResult(
        answer=answer,
        citations=_build_citations(hits),
        retrieved=hits,
        weak_retrieval=False,
    )
