"""Unit tests for structure-aware and fixed-size chunkers."""

from embeduino.chunker import (
    chunk_document,
    fixed_size_chunk,
    structure_aware_chunk,
)


SAMPLE_MD = """# digitalWrite()

## Description

Write a HIGH or a LOW value to a digital pin.

If the pin has been configured as an OUTPUT with pinMode(), its voltage will be set.

## Syntax

```cpp
digitalWrite(pin, value)
```

## Parameters

- pin: the Arduino pin number.
- value: HIGH or LOW.

## Returns

Nothing

## Example Code

```cpp
void setup() {
  pinMode(13, OUTPUT);
}
void loop() {
  digitalWrite(13, HIGH);
  delay(1000);
  digitalWrite(13, LOW);
}
```
"""

SAMPLE_HTML = """<!DOCTYPE html><html><body><article>
<h1>delay()</h1>
<h2>Description</h2>
<p>Pauses the program for the amount of time in milliseconds.</p>
<h2>Syntax</h2>
<pre><code>delay(ms)</code></pre>
</article></body></html>
"""


def test_structure_splits_on_headings():
    chunks = structure_aware_chunk(SAMPLE_MD, "digital_write.md")
    assert len(chunks) >= 3
    headings = {c.heading_path for c in chunks}
    assert any("Description" in h for h in headings)
    assert any("Parameters" in h or "Syntax" in h for h in headings)


def test_structure_keeps_code_chunks():
    chunks = structure_aware_chunk(SAMPLE_MD, "digital_write.md")
    code_chunks = [c for c in chunks if c.chunk_type == "code"]
    assert len(code_chunks) >= 1
    assert any("digitalWrite" in c.text for c in code_chunks)


def test_structure_ids_stable_and_unique():
    a = structure_aware_chunk(SAMPLE_MD, "digital_write.md")
    b = structure_aware_chunk(SAMPLE_MD, "digital_write.md")
    assert [c.id for c in a] == [c.id for c in b]
    assert len({c.id for c in a}) == len(a)


def test_structure_handles_html():
    chunks = structure_aware_chunk(SAMPLE_HTML, "delay.html")
    assert len(chunks) >= 1
    blob = " ".join(c.text for c in chunks)
    assert "delay" in blob.lower() or "Pauses" in blob


def test_fixed_size_overlap():
    long_text = "word " * 400
    chunks = fixed_size_chunk(long_text, "long.txt", size=200, overlap=40)
    assert len(chunks) >= 2
    assert all(c.chunk_type == "fixed" for c in chunks)


def test_chunk_document_strategy_dispatch():
    s = chunk_document(SAMPLE_MD, "x.md", strategy="structure")
    f = chunk_document(SAMPLE_MD, "x.md", strategy="fixed", size=300, overlap=30)
    assert all(c.chunk_type in ("section", "code") for c in s)
    assert all(c.chunk_type == "fixed" for c in f)


def test_heading_path_nested():
    md = "# A\n\n## B\n\ntext about B\n\n### C\n\ntext about C\n"
    chunks = structure_aware_chunk(md, "nested.md")
    paths = [c.heading_path for c in chunks]
    assert any("A > B" in p for p in paths)
    assert any("A > B > C" in p for p in paths)
