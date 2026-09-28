"""Retrieve-and-generate with citations and weak-retrieval honesty."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Set

from embeduino.config import Settings
from embeduino.fusion import bm25_rank, reciprocal_rank_fusion
from embeduino.store import VectorStore
from embeduino.web_search import SerpApiSearcher, should_trigger_web_search

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
    web_search_used: bool = False
    web_credits_used: int = 0  # Changed from bool to int
    web_status: str = ""


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
        url = meta.get("url", "")
        origin = "web" if meta.get("chunk_type") == "web" else "local"
        
        cites.append(
            {
                "id": h["id"],
                "source": meta.get("source", ""),
                "heading_path": meta.get("heading_path", ""),
                "score": round(float(h.get("score", 0.0)), 4),
                "rerank_score": round(float(h.get("rerank_score", h.get("score", 0.0))), 4),
                "rrf_score": round(float(h.get("rrf_score", 0.0)), 4) if "rrf_score" in h else None,
                "url": url,
                "origin": origin,
                "preview": (h.get("text") or "")[:200],
            }
        )
    return cites


def _is_weak(question: str, hits: List[Dict[str, Any]], min_score: float) -> bool:
    if not hits:
        return True
    best = hits[0]
    
    # Check if we have web chunks in top results
    has_web = any(h.get("metadata", {}).get("chunk_type") == "web" for h in hits[:3])
    
    best_score = float(best.get("score", 0.0))
    best_rrf = float(best.get("rrf_score", 0.0))
    best_lex = float(best.get("lex_overlap", 0.0))
    
    # Web chunks with decent lexical overlap should pass
    if has_web and best_lex >= 0.20 and best_rrf > 0.01:
        logger.info("Accepting web chunks: lex=%.2f rrf=%.3f", best_lex, best_rrf)
        return False
    
    # Web chunks with good RRF score should pass even with lower lex
    if has_web and best_rrf > 0.025:
        logger.info("Accepting web chunks: rrf=%.3f", best_rrf)
        return False
    
    if best_score < min_score:
        return True
    
    if best_lex < 0.15 and best_score < 0.45:
        return True
    
    q_terms = _tokens(question)
    content_terms = set()
    for h in hits[:3]:
        content_terms |= _tokens(h.get("text") or "")
    
    distinctive = q_terms - {"arduino", "board", "sketch", "code", "function", "use", "using", "pin", "pins"}
    if distinctive and len(distinctive & content_terms) / len(distinctive) < 0.2:
        return True
    return False


def _extractive_answer(question: str, hits: List[Dict[str, Any]]) -> str:
    if not hits:
        return (
            "I don't know — no relevant documentation chunks were retrieved "
            "for this question."
        )
    
    # Detect product names in question (UNO R4, ESP32, Nano 33, etc.)
    product_patterns = [
        r'\b(UNO\s+R\d+(?:\s+\w+)?)\b',
        r'\b(Nano\s+(?:33|ESP32)(?:\s+\w+)?)\b',
        r'\b(ESP32(?:-\w+)?)\b',
        r'\b(Mega\s+\d+)\b',
        r'\b(Due|Leonardo|Micro|Yun)\b',
    ]
    
    mentioned_products = set()
    for pattern in product_patterns:
        for match in re.finditer(pattern, question, re.IGNORECASE):
            mentioned_products.add(match.group(1).lower())
    
    # Check if local chunks mention the product
    local_has_product = False
    if mentioned_products:
        for h in hits:
            if h.get("metadata", {}).get("chunk_type") != "web":
                text_lower = h.get("text", "").lower()
                if any(prod in text_lower for prod in mentioned_products):
                    local_has_product = True
                    break
    
    # Choose primary chunk by best rerank or RRF score across ALL sources
    # If question mentions a product not in local chunks, prefer best web chunk
    best_chunk = None
    best_score = -1.0
    
    for h in hits:
        # Use RRF score if available, else rerank_score
        score = h.get("rrf_score", h.get("rerank_score", h.get("score", 0.0)))
        
        is_web = h.get("metadata", {}).get("chunk_type") == "web"
        
        # Boost web chunks if question mentions product not in local
        if is_web and mentioned_products and not local_has_product:
            score += 0.1
        
        if score > best_score:
            best_score = score
            best_chunk = h
    
    # For ties within local chunks, prefer certain heading types
    if not best_chunk.get("metadata", {}).get("chunk_type") == "web":
        same_score_local = [
            h for h in hits
            if h.get("metadata", {}).get("chunk_type") != "web"
            and abs((h.get("rrf_score", h.get("rerank_score", h.get("score", 0.0)))) - best_score) < 0.01
        ]
        
        if len(same_score_local) > 1:
            preferred = []
            for h in same_score_local:
                heading = ((h.get("metadata") or {}).get("heading_path") or "").lower()
                if any(k in heading for k in ("description", "parameter", "notes", "syntax", "return")):
                    preferred.append(h)
            if preferred:
                best_chunk = preferred[0]
    
    primary = best_chunk
    
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
        """Extract most relevant sentences from chunk based on question overlap."""
        text = h.get("text") or ""
        
        # Split into sentences/lines
        sentences = []
        for para in re.split(r"\n\s*\n", text):
            para = para.strip()
            if not para or (re.match(r"^#{1,6}\s+", para) and len(para) < 60):
                continue
            
            # Check if this is a "Section - Key: Value" spec block (Gatsby page-data)
            # Lines like "Pins - Digital I/O Pins: 14" should each be a sentence
            if re.match(r'^[A-Z][a-zA-Z\s]+ - [^:]+:\s*.+$', para):
                # Spec line, treat as single sentence
                sentences.append(para)
            else:
                # Split on sentence boundaries
                for sent in re.split(r'[.!?]\s+', para):
                    sent = sent.strip()
                    if len(sent) > 30:  # Min sentence length
                        sentences.append(sent)
        
        if not sentences:
            return text[:700]
        
        # Score sentences by overlap with question
        q_terms = _tokens(question)
        scored = []
        for sent in sentences:
            sent_terms = _tokens(sent)
            if sent_terms:
                overlap = len(q_terms & sent_terms) / len(q_terms) if q_terms else 0
                
                # Boost spec lines for pin/power/communication questions
                boost = 0.0
                if re.match(r'^(Pins|Power|Communication|Memory|Clock) - ', sent):
                    # Check if question is about specs
                    spec_keywords = {'pin', 'voltage', 'current', 'power', 'specification', 'spec', 'communication', 'uart', 'i2c', 'spi'}
                    if any(kw in question.lower() for kw in spec_keywords):
                        boost = 0.3
                
                scored.append((overlap + boost, sent))
        
        # Sort by overlap, take top 3-6
        scored.sort(key=lambda x: -x[0])
        top_sentences = [s for _, s in scored[:6] if _>0.05]  # Lower threshold for spec lines
        
        if not top_sentences:
            # Fallback to first few sentences
            top_sentences = sentences[:4]
        
        return ". ".join(top_sentences[:6]) + "."

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
        label = f"[{h['id']}]"
        if meta.get("chunk_type") == "web":
            url = meta.get("url", "")
            label = f"[{h['id']}] (web: {url})"
        
        context_blocks.append(
            f"{label} source={meta.get('source')} heading={meta.get('heading_path')}\n"
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


def _validate_citations(answer: str, hits: List[Dict[str, Any]]) -> bool:
    """Check that all cited chunk ids in answer exist in hits."""
    cited_ids = set(re.findall(r"\[([^\]]+)\]", answer))
    available_ids = {h["id"] for h in hits}
    
    invalid = cited_ids - available_ids
    if invalid:
        logger.warning("Invalid citations found: %s", invalid)
        return False
    return True


def ask(
    question: str,
    store: VectorStore,
    settings: Settings,
    web_mode_override: str | None = None,
) -> AskResult:
    effective_web_mode = web_mode_override or settings.web_mode
    
    # 1. Vector retrieval
    raw_vector = store.query(question, top_k=max(settings.top_k * 3, 12))
    
    # 2. Check if local retrieval is weak (before reranking)
    temp_reranked = _rerank(question, raw_vector)
    local_weak = _is_weak(question, temp_reranked, settings.min_score)
    
    # 3. Decide whether to use web search
    web_search_used = False
    web_credits_used = 0
    web_status = ""
    web_hits_raw = []
    
    trigger_web = should_trigger_web_search(
        question, temp_reranked, local_weak, effective_web_mode
    )
    
    if trigger_web and settings.serpapi_api_key:
        searcher = SerpApiSearcher(
            api_key=settings.serpapi_api_key,
            cache_dir=settings.serp_cache,
            max_calls_per_run=settings.serp_max_calls,
            timeout_s=settings.serp_timeout_s,
            trusted_domains=settings.web_sites,
        )
        web_results, credits_used, status = searcher.search(
            question,
            trusted_sites=settings.web_sites,
            num_results=settings.web_num,
        )
        # Convert WebHit objects to dicts with computed scores
        web_hits_raw = [w.to_dict(score=searcher._compute_score(question, w.text, w.url)) for w in web_results]
        web_search_used = True
        web_credits_used = credits_used
        web_status = status
        logger.info(
            "Web search: %d web chunks parsed, status=%s, credits_used=%d",
            len(web_hits_raw), status, credits_used
        )
    
    # 4. Combine and rerank: vector + web through same reranker
    # This ensures web chunks get rerank scores and can compete fairly
    combined_for_rerank = list(raw_vector)
    if web_hits_raw:
        combined_for_rerank.extend(web_hits_raw)
    
    vector_ranked = _rerank(question, combined_for_rerank)
    
    # 5. BM25 retrieval over local + web combined pool
    # This ensures web chunks appear in BM25 list too
    all_chunks = store.get_all_chunks()
    if web_hits_raw:
        all_chunks = list(all_chunks) + web_hits_raw
    bm25_ranked = bm25_rank(question, all_chunks) if all_chunks else []
    
    # 6. Reciprocal Rank Fusion
    # Now web chunks appear in both vector_ranked and bm25_ranked, fixing the structural bias
    ranked_lists = [vector_ranked, bm25_ranked]
    
    fused = reciprocal_rank_fusion(ranked_lists, k=settings.rrf_k)
    hits = fused[: settings.top_k]
    _log_retrieved(hits)
    
    # 6. Post-fusion weak check
    weak = _is_weak(question, hits, settings.min_score)
    
    if weak:
        logger.warning(
            "Weak retrieval after fusion for %r (best score=%.3f lex=%.2f min=%.3f)",
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
            web_search_used=web_search_used,
            web_credits_used=web_credits_used,
            web_status=web_status,
        )
    
    # 7. Generate answer
    if settings.generation_backend == "openai":
        answer = _openai_answer(question, hits, settings)
        
        # 8. Validate citations for OpenAI answers
        if not _validate_citations(answer, hits):
            logger.warning("Citation validation failed, falling back to extractive")
            answer = _extractive_answer(question, hits)
    else:
        answer = _extractive_answer(question, hits)
    
    return AskResult(
        answer=answer,
        citations=_build_citations(hits),
        retrieved=hits,
        weak_retrieval=False,
        web_search_used=web_search_used,
        web_credits_used=web_credits_used,
        web_status=web_status,
    )
