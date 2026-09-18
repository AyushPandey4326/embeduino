"""Structure-aware and fixed-size document chunkers."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from typing import Iterable, List, Optional, Sequence


@dataclass
class Chunk:
    """A single retrievable unit of documentation."""

    id: str
    text: str
    source: str
    heading_path: str = ""
    chunk_type: str = "section"  # section | code | fixed
    start_line: Optional[int] = None
    metadata: dict = field(default_factory=dict)

    def preview(self, n: int = 120) -> str:
        t = " ".join(self.text.split())
        return t if len(t) <= n else t[: n - 3] + "..."


_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$", re.MULTILINE)
_FENCE_RE = re.compile(r"(```[\w+-]*\n.*?```)", re.DOTALL)


def _stable_id(source: str, heading_path: str, text: str, idx: int) -> str:
    digest = hashlib.sha1(
        f"{source}|{heading_path}|{idx}|{text[:80]}".encode("utf-8")
    ).hexdigest()[:12]
    safe = re.sub(r"[^\w.-]+", "_", source.replace("\\", "/").split("/")[-1])
    return f"{safe}::{idx}::{digest}"


def _strip_md_inline(s: str) -> str:
    s = re.sub(r"`([^`]+)`", r"\1", s)
    s = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", s)
    s = re.sub(r"[*_]+", "", s)
    return s.strip()


def _html_to_text_blocks(html: str) -> str:
    """Lightweight HTML → markdown-ish text for structure-aware chunking."""
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(html, "lxml")
    for tag in soup(["script", "style", "nav", "footer"]):
        tag.decompose()
    lines: List[str] = []
    body = soup.find("article") or soup.body or soup
    for el in body.descendants:
        if getattr(el, "name", None) is None:
            continue
        name = el.name.lower()
        if name in ("h1", "h2", "h3", "h4", "h5", "h6"):
            level = int(name[1])
            text = el.get_text(" ", strip=True)
            if text:
                lines.append("#" * level + " " + text)
        elif name == "pre":
            code = el.get_text()
            lines.append("```\n" + code.strip("\n") + "\n```")
        elif name == "p":
            text = el.get_text(" ", strip=True)
            if text:
                lines.append(text)
        elif name == "li":
            text = el.get_text(" ", strip=True)
            if text:
                lines.append("- " + text)
    if lines:
        return "\n\n".join(lines)
    return soup.get_text("\n", strip=True)


def _split_by_headings(md: str) -> List[tuple[str, str]]:
    """Split markdown into (heading_path, body) segments."""
    matches = list(_HEADING_RE.finditer(md))
    if not matches:
        return [("", md.strip())] if md.strip() else []

    segments: List[tuple[list[tuple[int, str]], str]] = []
    preamble = md[: matches[0].start()].strip()
    if preamble:
        segments.append(([], preamble))

    stack: List[tuple[int, str]] = []
    for i, m in enumerate(matches):
        level = len(m.group(1))
        title = _strip_md_inline(m.group(2))
        while stack and stack[-1][0] >= level:
            stack.pop()
        stack.append((level, title))
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(md)
        body = md[start:end].strip()
        segments.append((list(stack), body))

    result: List[tuple[str, str]] = []
    for stack_items, body in segments:
        path = " > ".join(t for _, t in stack_items) if stack_items else ""
        if body:
            result.append((path, body))
    return result


def _extract_code_blocks(body: str) -> tuple[List[str], str]:
    codes = _FENCE_RE.findall(body)
    prose = _FENCE_RE.sub("\n", body)
    prose = re.sub(r"\n{3,}", "\n\n", prose).strip()
    return codes, prose


def _overlap_suffix(text: str, overlap: int) -> str:
    if overlap <= 0 or not text:
        return ""
    words = text.split()
    if len(words) <= overlap:
        return text
    return " ".join(words[-overlap:])


def _body_len(text: str) -> int:
    """Content length excluding the leading ## heading line."""
    lines = text.splitlines()
    if lines and lines[0].startswith("## "):
        return len("\n".join(lines[1:]).strip())
    return len(text.strip())


_MERGEABLE_HEADINGS = re.compile(
    r"(^|[>\s])(Returns|Return value|See also|See Also)\s*$", re.I
)


def _merge_tiny_chunks(chunks: List[Chunk], min_body: int = 60) -> List[Chunk]:
    """Fold stub sections (e.g. Returns: Nothing) into the previous section."""
    if not chunks:
        return chunks
    merged: List[Chunk] = []
    for c in chunks:
        leaf = (c.heading_path or "").split(">")[-1].strip()
        stub = bool(_MERGEABLE_HEADINGS.search(leaf))
        if (
            merged
            and c.chunk_type == "section"
            and merged[-1].chunk_type == "section"
            and merged[-1].source == c.source
            and stub
            and _body_len(c.text) < min_body
        ):
            prev = merged[-1]
            combined_text = prev.text.rstrip() + "\n\n" + c.text.lstrip()
            merged[-1] = Chunk(
                id=prev.id,
                text=combined_text,
                source=prev.source,
                heading_path=prev.heading_path or c.heading_path,
                chunk_type="section",
                metadata={**prev.metadata, "merged_tiny": True},
            )
        else:
            merged.append(c)
    out: List[Chunk] = []
    for i, c in enumerate(merged):
        out.append(
            Chunk(
                id=_stable_id(c.source, c.heading_path, c.text, i),
                text=c.text,
                source=c.source,
                heading_path=c.heading_path,
                chunk_type=c.chunk_type,
                metadata=c.metadata,
            )
        )
    return out


def structure_aware_chunk(
    text: str,
    source: str,
    *,
    max_chars: int = 900,
    overlap_words: int = 40,
    keep_code_separate: bool = True,
    merge_tiny: bool = True,
) -> List[Chunk]:
    """
    Chunk by headings / sections; keep fenced code as dedicated chunks when
    large enough; apply small word overlap between prose splits of the same section.
    """
    if source.lower().endswith((".html", ".htm")) or (
        "<html" in text[:200].lower() or "<h1" in text[:500].lower()
    ):
        if "<" in text and ("<h1" in text.lower() or "<html" in text.lower()):
            text = _html_to_text_blocks(text)

    sections = _split_by_headings(text)
    chunks: List[Chunk] = []
    idx = 0

    for heading_path, body in sections:
        codes, prose = _extract_code_blocks(body) if keep_code_separate else ([], body)

        if prose:
            pieces = _split_prose(prose, max_chars=max_chars, overlap_words=overlap_words)
            for piece in pieces:
                if not piece.strip():
                    continue
                content = f"## {heading_path}\n\n{piece}" if heading_path else piece
                chunks.append(
                    Chunk(
                        id=_stable_id(source, heading_path, content, idx),
                        text=content.strip(),
                        source=source,
                        heading_path=heading_path,
                        chunk_type="section",
                        metadata={"max_chars": max_chars},
                    )
                )
                idx += 1

        for code in codes:
            code = code.strip()
            if len(code) < 20 and prose:
                continue
            content = (
                f"## {heading_path} (example)\n\n{code}" if heading_path else code
            )
            chunks.append(
                Chunk(
                    id=_stable_id(source, heading_path, content, idx),
                    text=content.strip(),
                    source=source,
                    heading_path=heading_path,
                    chunk_type="code",
                )
            )
            idx += 1

    if merge_tiny:
        chunks = _merge_tiny_chunks(chunks)
    return chunks


def _split_prose(prose: str, max_chars: int, overlap_words: int) -> List[str]:
    if len(prose) <= max_chars:
        return [prose]
    paras = re.split(r"\n\s*\n", prose)
    pieces: List[str] = []
    buf = ""
    for p in paras:
        p = p.strip()
        if not p:
            continue
        candidate = f"{buf}\n\n{p}".strip() if buf else p
        if len(candidate) <= max_chars:
            buf = candidate
            continue
        if buf:
            pieces.append(buf)
            overlap = _overlap_suffix(buf, overlap_words)
            buf = f"{overlap}\n\n{p}".strip() if overlap else p
            if len(buf) > max_chars:
                pieces.extend(_hard_split(buf, max_chars, overlap_words))
                buf = ""
        else:
            pieces.extend(_hard_split(p, max_chars, overlap_words))
            buf = ""
    if buf:
        pieces.append(buf)
    return pieces


def _hard_split(text: str, max_chars: int, overlap_words: int) -> List[str]:
    words = text.split()
    if not words:
        return []
    out: List[str] = []
    i = 0
    while i < len(words):
        chunk_words: List[str] = []
        size = 0
        while i < len(words):
            w = words[i]
            add = len(w) + (1 if chunk_words else 0)
            if chunk_words and size + add > max_chars:
                break
            chunk_words.append(w)
            size += add
            i += 1
        out.append(" ".join(chunk_words))
        if i >= len(words):
            break
        if overlap_words > 0:
            i = max(i - overlap_words, i - len(chunk_words) + 1)
    return out


def fixed_size_chunk(
    text: str,
    source: str,
    *,
    size: int = 500,
    overlap: int = 50,
) -> List[Chunk]:
    """Baseline character-window chunker for comparison."""
    if source.lower().endswith((".html", ".htm")) or (
        "<html" in text[:200].lower() or "<h1" in text[:500].lower()
    ):
        if "<" in text and ("<h1" in text.lower() or "<html" in text.lower()):
            text = _html_to_text_blocks(text)
    text = text.strip()
    if not text:
        return []
    chunks: List[Chunk] = []
    start = 0
    idx = 0
    n = len(text)
    while start < n:
        end = min(start + size, n)
        piece = text[start:end].strip()
        if piece:
            chunks.append(
                Chunk(
                    id=_stable_id(source, "fixed", piece, idx),
                    text=piece,
                    source=source,
                    heading_path="",
                    chunk_type="fixed",
                    metadata={"size": size, "overlap": overlap},
                )
            )
            idx += 1
        if end >= n:
            break
        start = max(end - overlap, start + 1)
    return chunks


def chunk_document(
    text: str,
    source: str,
    strategy: str = "structure",
    **kwargs,
) -> List[Chunk]:
    strategy = strategy.lower()
    if strategy in ("structure", "structure_aware", "smart"):
        return structure_aware_chunk(text, source, **kwargs)
    if strategy in ("fixed", "fixed_size", "baseline"):
        return fixed_size_chunk(text, source, **kwargs)
    raise ValueError(f"Unknown chunk strategy: {strategy}")


def chunk_many(
    docs: Iterable[tuple[str, str]],
    strategy: str = "structure",
    **kwargs,
) -> List[Chunk]:
    all_chunks: List[Chunk] = []
    for source, text in docs:
        all_chunks.extend(chunk_document(text, source, strategy=strategy, **kwargs))
    return all_chunks
