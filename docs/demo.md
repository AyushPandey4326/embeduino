# Embeduino Live: 3-Minute Demo Script

This script demonstrates embeduino's live web search integration for the SerpApi India Hackathon 2026. All commands should be run from the repository root.

**Prerequisites:**
- Python 3.10+
- Free SerpApi API key (sign up at https://serpapi.com/ — 250 searches/month, no credit card)
- Add `SERPAPI_API_KEY=your_key_here` to `.env`

**Total runtime:** ~3 minutes (can be sped up in recording)

---

## Setup (do this before recording)

```bash
# Clone and enter repo
cd /path/to/embeduino

# Set up environment
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"

# Configure API key
cp .env.example .env
# Edit .env: add your SERPAPI_API_KEY

# Ingest local corpus
python3 -m embeduino ingest --reset
```

---

## Demo Commands (record this part)

### 0:00–0:20 — Show the problem

**Narration:** "The offline Arduino docs corpus has only 8 reference pages. Let's see what happens when we ask about a newer board."

```bash
# Show local corpus
ls -1 data/arduino_docs/
# Output: 8 markdown files (analog_read, digital_write, etc.)

# Count ingested chunks
python3 -m embeduino ask "What does the vector store contain?" --web off | grep -i "chunk"
```

---

### 0:20–0:50 — Local question works (web off)

**Narration:** "First, a question covered by the local docs — digitalWrite."

```bash
python3 -m embeduino ask "What does digitalWrite do?" --web off
```

**Expected output:**
- Answer from local docs (mentions HIGH, LOW, pin states)
- Citations from `digital_write.md`
- "Web search: not triggered"
- "0 SerpApi credits"

---

### 0:50–1:40 — Freshness question fails offline, succeeds with web

**Narration:** "Now a question about the Arduino UNO R4 WiFi — a board released in 2023, not in our local corpus."

```bash
# Try offline first
python3 -m embeduino ask "What are the pin specifications for Arduino UNO R4 WiFi?" --web off
```

**Expected output:**
- Yellow panel: "I don't know — retrieval confidence is too low"

**Narration:** "Enable web search (auto mode)."

```bash
# Same question with web auto
python3 -m embeduino ask "What are the pin specifications for Arduino UNO R4 WiFi?" --web auto
```

**Expected output:**
- Green panel with answer mentioning UNO R4 WiFi specs
- Citations table shows:
  - **Origin column**: "web" for SerpApi results, "local" for any local hits
  - **URL column**: docs.arduino.cc or github.com links
  - **RRF column**: fusion scores
- "Web search used (1 SerpApi credit)"

---

### 1:40–2:10 — Mixed fusion (local + web)

**Narration:** "Ask a hybrid question that benefits from both local docs and web results."

```bash
python3 -m embeduino ask "How do I use digitalWrite with ESP32 boards?" --web auto
```

**Expected output:**
- Answer fuses local digitalWrite knowledge with ESP32-specific web results
- Citations show both local and web origins
- RRF scores visible in table

---

### 2:10–2:30 — Honesty check (nonsense question)

**Narration:** "The honesty gate still works — nonsense questions fail even with web search."

```bash
python3 -m embeduino ask "How do I configure a quantum flux capacitor on Arduino?" --web auto
```

**Expected output:**
- Yellow panel: "I don't know — retrieval confidence is too low"
- May show web search triggered but still refuses to answer

---

### 2:30–2:50 — Cache and tests

**Narration:** "Repeat a web question — served from cache, no credit used."

```bash
python3 -m embeduino ask "What are the pin specifications for Arduino UNO R4 WiFi?" --web auto
```

**Expected output:**
- Same answer
- "Web search served from cache (0 credits)"

**Narration:** "All tests pass without network or API key."

```bash
python3 -m pytest tests/ -q
```

**Expected output:**
```
....................                                                     [100%]
20 passed in 0.18s
```

---

### 2:50–3:00 — Close

**Narration:** "Check the cache directory and repo link."

```bash
ls .serp_cache/
# Shows cached JSON files

echo "GitHub: https://github.com/YourUsername/embeduino"
echo "Setup: see README.md"
```

**Screen:** Show README.md section "What's New: SerpApi Hackathon 2026"

---

## Recording Tips

1. **Run commands in advance** to warm the embedding model and cache
2. **Clear `.serp_cache/` before recording** so the first web call shows "1 credit used"
3. **Speed up** the video if needed (tests and long output can be 2–4x)
4. **DO NOT show** `.env` or the API key on screen
5. **Open in incognito** if uploading to YouTube / Drive for submission
6. **Keep under 3:00** (or trim to 3:00 in post)
7. **Windows users**: Set `PYTHONUTF8=1` or run `chcp 65001` before recording to avoid encoding errors

---

## After Recording

1. **Trim** to exactly 3:00 or less
2. **Upload** as unlisted YouTube video or Google Drive link
3. **Ensure** the link opens in incognito (test it)
4. **Submit** via https://serpapi.github.io/serpapi-india-hackathon-2026/

---

## Commands Summary (copy-paste friendly)

```bash
# Setup
cd /path/to/embeduino
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env
# Edit .env: add SERPAPI_API_KEY
python3 -m embeduino ingest --reset

# Demo
ls -1 data/arduino_docs/
python3 -m embeduino ask "What does digitalWrite do?" --web off
python3 -m embeduino ask "What are the pin specifications for Arduino UNO R4 WiFi?" --web off
python3 -m embeduino ask "What are the pin specifications for Arduino UNO R4 WiFi?" --web auto
python3 -m embeduino ask "How do I use digitalWrite with ESP32 boards?" --web auto
python3 -m embeduino ask "How do I configure a quantum flux capacitor on Arduino?" --web auto
python3 -m embeduino ask "What are the pin specifications for Arduino UNO R4 WiFi?" --web auto
python3 -m pytest tests/ -q
ls .serp_cache/
```
