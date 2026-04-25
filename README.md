# ◎ Research Paper Q&A Assistant

> Ask anything about your research papers. Upload research papers. Ask questions. Get precise, citation-backed answers grounded only in your documents.

**Stack:** LangChain · Gemini 2.5 Flash · FAISS · Sarvam AI · Streamlit

---

## Table of Contents

1. [Overview](#overview)
2. [Features](#features)
3. [Architecture](#architecture)
4. [Project Structure](#project-structure)
5. [Module Reference](#module-reference)
6. [Voice — STT & TTS](#voice--stt--tts)
7. [Setup & Installation](#setup--installation)
8. [Configuration Reference](#configuration-reference)
9. [Running the App](#running-the-app)
10. [How to Use](#how-to-use)
11. [RAG Pipeline Deep Dive](#rag-pipeline-deep-dive)
12. [Evaluation Metrics](#evaluation-metrics)
13. [Troubleshooting](#troubleshooting)
14. [Potential Extensions](#potential-extensions)

---

## Overview

This is a production-grade **Retrieval-Augmented Generation (RAG)** system built for research workflows. Upload one or more PDF research papers, then ask natural language questions — by typing or by voice. It retrieves the most relevant passages, reranks them with a neural cross-encoder, and generates a grounded answer via Gemini 2.5 Flash. Every factual claim is accompanied by a citation showing the paper name, page number, and the verbatim supporting excerpt.

The system is designed to run fully locally (embeddings, reranking, vector store) with only the LLM and voice features requiring external API calls.

---

## Features

| Feature | Detail |
|---|---|
| **Multi-PDF ingestion** | Upload and index any number of PDFs simultaneously |
| **6-layer text cleaning** | HTML stripping, ligature repair, DOI/URL removal, watermark removal, reference section cutoff, whitespace normalisation — applied at ingestion time |
| **Section-aware chunking** | Detects section headings; falls back to recursive fixed-size splitting with overlap |
| **Local embeddings** | `all-MiniLM-L6-v2` — 384-dim, runs on CPU, zero API cost |
| **Persistent FAISS index** | Survives app restarts; incremental add without full rebuild |
| **Hybrid retrieval** | 70% FAISS semantic + 30% BM25 keyword — catches what pure embeddings miss |
| **Cross-encoder reranking** | `ms-marco-MiniLM-L-6-v2` — reranks top-10 → top-5 for higher precision |
| **Gemini 2.5 Flash generation** | Low-temperature (0.2), context-only, citation-enforcing prompt |
| **Auto evaluation** | Context utilisation, retrieval relevance, source diversity — shown per answer |
| **Voice input (STT)** | Sarvam AI `saarika:v2.5` — record your question in the browser |
| **Voice output (TTS)** | Sarvam AI `bulbul:v2` — 16 speakers, 10 Indian languages, adjustable pace |
| **Corrupt index detection** | On startup, scans saved chunks for HTML contamination and auto-discards stale indexes |
| **Clean citation panel** | Rendered via `st.components.v1.html()` — immune to Streamlit's markdown-in-expander HTML-escaping bug |

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                     INGESTION PIPELINE                   │
│                                                          │
│  PDF Upload → PyPDF Loader → 6-Layer Cleaner            │
│      → Section-Aware Chunker → MiniLM Embedder          │
│      → FAISS Index (persisted to disk)                  │
│      → BM25 Index (in-memory)                           │
└─────────────────────────────────────────────────────────┘
                          │
                   [At query time]
                          │
┌─────────────────────────────────────────────────────────┐
│                      QUERY PIPELINE                      │
│                                                          │
│  User question (text or 🎙 voice via Sarvam STT)        │
│      ↓                                                   │
│  Hybrid Retrieval: 70% FAISS + 30% BM25 → top 10        │
│      ↓                                                   │
│  Cross-Encoder Reranking (ms-marco) → top 5             │
│      ↓                                                   │
│  Context builder → structured SOURCE blocks              │
│      ↓                                                   │
│  Gemini 2.5 Flash (temp 0.2, context-only prompt)       │
│      ↓                                                   │
│  Answer + Citations + Evaluation metrics                 │
│      ↓                                                   │
│  (Optional) 🔊 Sarvam TTS → MP3 playback               │
└─────────────────────────────────────────────────────────┘
```

---

## Project Structure

```
Research Paper Q&A/
│
├── app.py                          # Streamlit UI — single entry point
├── config.py                       # All hyperparameters & paths
├── requirements.txt
├── .env.example                    # Copy → .env and add your API keys
├── README.md
│
├── data/
│   ├── raw/                        # Uploaded PDFs saved here
│   └── processed/
│       └── vectorstore/            # Persisted FAISS index files
│           ├── index.faiss
│           └── index.pkl
│
└── src/
    ├── pipeline.py                 # Orchestrator — wires all modules
    │
    ├── ingestion/
    │   └── pdf_loader.py           # PyPDFLoader; page-level extraction + metadata
    │
    ├── preprocessing/
    │   └── cleaner.py              # 6-layer HTML-first text cleaner (PRIMARY DEFENCE)
    │
    ├── chunking/
    │   └── chunker.py              # Section-aware + RecursiveCharacterTextSplitter
    │
    ├── embeddings/
    │   └── embedder.py             # HuggingFace MiniLM singleton; normalised vectors
    │
    ├── vectorstore/
    │   └── faiss_store.py          # Build / load / extend / clear FAISS index
    │
    ├── retrieval/
    │   └── retriever.py            # Semantic search + BM25Retriever + hybrid fusion
    │
    ├── reranking/
    │   └── reranker.py             # CrossEncoder; 
    │
    ├── generation/
    │   └── generator.py            # Prompt engineering + Gemini 2.5 Flash call
    │                               # Cleans excerpts before storing in source dicts
    │
    ├── evaluation/
    │   └── evaluator.py            # 3-metric quality scorer (no ground truth needed)
    │
    ├── utils/
    │   └── helpers.py              # File I/O, formatting, paper stats
    │
    └── voice/
        ├── sarvam_voice.py         # Sarvam AI STT (saarika:v2.5) + TTS (bulbul:v2)
```

---

## Module Reference

### `src/preprocessing/cleaner.py` — The Primary Defence

This module runs **before chunking and embedding**, making it the only reliable place to stop HTML contamination from reaching the FAISS index. The 6 layers run in strict order:

| Layer | What it does |
|---|---|
| 1. HTML stripping | `stdlib html.parser` extracts only visible text nodes; regex mop-up catches fragments; entity decode handles `&nbsp;`, `&amp;` etc. |
| 2. Ligature repair | Unicode ligatures (`ﬁ→fi`, `ﬂ→fl`, `ﬀ→ff`, etc.) |
| 3. Hyphen-break merge | `meth-\nod` → `method` |
| 4. Noise removal | URLs, DOIs (`10.xxxx/...`), arXiv IDs, ISSNs, watermarks, "Downloaded from" blocks |
| 5. Reference cutoff | Splits at the References / Bibliography heading — citation noise never enters the index |
| 6. Whitespace collapse | Multiple blank lines → single blank line |

It also runs a post-cleaning **HTML contamination guard**: any page still containing 2+ HTML tags after all 6 layers is discarded entirely with a warning log.

### `src/vectorstore/faiss_store.py`

Added `clear_vectorstore()` — deletes the persisted index files. Called automatically by `pipeline.py` when a corrupt index is detected at startup, and by the "Clear Knowledge Base" UI button.

### `src/pipeline.py`

`try_load_existing_index()` now includes `_index_is_corrupted()`: samples the first 50 chunks of any saved index looking for HTML tags. If found, calls `clear_vectorstore()`, sets a `_index_was_corrupt = True` flag on the pipeline object, and returns an empty pipeline. The Streamlit UI reads this flag and shows a yellow banner prompting the user to re-index.

### `src/generation/generator.py`

`generate_answer()` now applies `_strip_html_tags()` and `_clean_pdf_text()` to `doc.page_content` before storing it as the `excerpt` field in the source dict. This is a second defence layer in case any HTML slips past the cleaner (e.g. from an incremental add of an already-indexed paper with a pre-existing stale index).

### `app.py` — `_render_citations()`

The citation panel was previously rendered with `st.markdown(unsafe_allow_html=True)` inside `st.expander`. This is a known Streamlit behaviour where HTML gets silently escaped in certain versions, causing raw tags to appear as visible text. The fix replaces the entire block with a dedicated `_render_citations()` function that:

- Builds a complete `<!DOCTYPE html>` document with self-contained CSS
- Renders it via `st.components.v1.html()` into a sandboxed iframe
- Always renders correctly regardless of Streamlit version or nesting
- Auto-sizes iframe height: `max(120, n_citations × 130 + 30)` px

---

## Voice — STT & TTS

It also integrates **Sarvam AI** for both speech-to-text and text-to-speech — a production-grade Indian AI platform with excellent multilingual support.

### Speech-to-Text (STT)

**Provider:** Sarvam AI — `saarika:v2.5` model  
**How it works:**

1. User clicks the 🎙 microphone button in the query panel
2. Browser's native audio recorder captures the voice input (`st.audio_input`)
3. Audio bytes are sent to Sarvam's transcription API
4. Transcript is auto-populated into the question text area
5. If `stt_auto_submit` is set, the query fires automatically

**Supported:** Any language (auto-detect mode) or a specific BCP-47 language code

```python
# src/voice/sarvam_voice.py
from sarvamai import SarvamAI
client = SarvamAI(api_subscription_key=api_key)
response = client.speech_to_text.transcribe(
    file=audio_file,
    model="saarika:v2.5",
    language_code="unknown",   # auto-detect
)
```

### Text-to-Speech (TTS)

**Provider:** Sarvam AI — `bulbul:v2` model  
**How it works:**

1. User clicks **🔊 Listen** below any answer
2. Answer text is cleaned (markdown symbols stripped, truncated to 2000 chars)
3. Long text is split into ≤490-char sentence-aware chunks
4. Each chunk is sent to Sarvam TTS; MP3 audio parts are concatenated
5. Audio is cached in `st.session_state.tts_audio_cache` keyed by a stable content hash
6. `st.audio()` plays the result inline

**Configurable in the sidebar under 🔊 TTS Settings:**

| Setting | Options |
|---|---|
| Speaker voice | 16 voices: anushka, abhilash, manisha, vidya, arya, karun, hitesh, aditya, ritu, priya, neha, rahul, pooja, rohan, simran, kavya |
| Language | English (India), Hindi, Bengali, Tamil, Telugu, Kannada, Malayalam, Marathi, Gujarati, Punjabi |
| Pace | 0.5× – 2.0× (slider, step 0.1) |

```python
# src/voice/sarvam_voice.py
response = client.text_to_speech.convert(
    text=chunk,
    target_language_code="en-IN",
    speaker="anushka",
    pace=1.0,
    model="bulbul:v2",
    output_audio_codec="mp3",
)
```

## Setup & Installation

### Prerequisites

- Python 3.10 or higher
- A Google AI Studio API key (for Gemini)
- A Sarvam AI API key (for voice features — optional)

### 1. Clone the repository

```bash
git clone https://github.com/AnkitaSarkar25/Research-Paper-Q-A-Assistant.git
cd Research-Paper-Q-A-Assistant
```

### 2. Create a virtual environment

```bash
python -m venv venv

# macOS / Linux
source venv/bin/activate

# Windows
venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

First-run model downloads (automatic, cached in `~/.cache/huggingface/`):

| Model | Size | Purpose |
|---|---|---|
| `all-MiniLM-L6-v2` | ~22 MB | Sentence embeddings |
| `cross-encoder/ms-marco-MiniLM-L-6-v2` | ~85 MB | Reranking |

### 4. Configure API keys

```bash
cp .env.example .env
```

Edit `.env`:

```env
# Required — Gemini LLM
GOOGLE_API_KEY=AIza...

# Optional — Sarvam AI voice features
SARVAM_API_KEY=your_sarvam_key_here
```

Get your keys:
- **Google API key:** https://aistudio.google.com/app/apikey (free tier available)
- **Sarvam API key:** https://app.sarvam.ai (free tier available)

---

## Configuration Reference

All tuneable parameters live in `config.py`. Change a value here and it propagates to every module automatically.

| Parameter | Default | Description |
|---|---|---|
| `LLM_MODEL_NAME` | `gemini-2.5-flash` | Gemini model identifier |
| `EMBEDDING_MODEL_NAME` | `all-MiniLM-L6-v2` | Local embedding model |
| `CHUNK_SIZE` | `500` | Characters per text chunk |
| `CHUNK_OVERLAP` | `100` | Overlap between consecutive chunks |
| `TOP_K_RETRIEVE` | `10` | Chunks fetched from FAISS + BM25 |
| `TOP_K_RERANK` | `5` | Chunks passed to LLM after reranking |
| `SEMANTIC_WEIGHT` | `0.7` | FAISS weight in hybrid search (BM25 = 0.3) |
| `MAX_OUTPUT_TOKENS` | `5000` | Gemini max output length |
| `TEMPERATURE` | `0.2` | LLM temperature — lower = less hallucination |

---

## Running the App

```bash
streamlit run app.py
```

Open **http://localhost:8501** in Chrome or Edge (required for the microphone button).

---

## How to Use

### Basic workflow

1. Enter your **Google API key** in the sidebar (or set it in `.env`)
2. Optionally enter your **Sarvam API key** for voice features
3. Upload one or more **PDF research papers** using the file uploader
4. Click **⚡ Process & Index PDFs** — this runs the full ingestion pipeline
5. Type a question in the query box, or click **🎙 Record** to ask by voice
6. Click **Search & Answer →**
7. View the answer, expand **📎 cited passages** to see sources, click **🔊 Listen** to hear the answer

### Effective questions

```
What is the main contribution of this paper?
How does the attention mechanism work in the Transformer?
What datasets were used for evaluation, and what were the results?
Compare the methodologies across the papers you have indexed.
What are the limitations the authors acknowledge?
In the NIDS paper, what ML models were found to perform best?
```

### Voice tips

- The microphone button requires **Chrome or Edge** (Web Speech API)
- Speak clearly and pause at the end — Sarvam auto-detects when you stop
- If STT fails, check that `SARVAM_API_KEY` is set in your `.env`
- TTS reads the first ~2000 characters of the answer
- Adjust **pace** in the sidebar TTS Settings if the voice is too fast or slow

---

## RAG Pipeline Deep Dive

### Why hybrid search?

Pure semantic search (embeddings) captures meaning but misses exact keyword matches. If you ask about a specific model name like "ResNet-50" or a dataset "NSL-KDD", the embedding may not strongly match chunks that contain those exact terms. BM25 is a classical TF-IDF ranker that excels at exact matches. The weighted fusion (`0.7 × semantic + 0.3 × BM25`) consistently outperforms either approach alone.

### Why cross-encoder reranking?

The bi-encoder (embedding) retrieves based on independent query and document vectors — fast but imprecise. The cross-encoder sees both query and passage together as a single input, producing a much more accurate relevance score. The tradeoff: it's slower, so we only run it on the top-10 candidates from retrieval (not the full corpus). Reranking top-10 → top-5 dramatically improves the precision of what the LLM actually sees.

### Why temperature 0.2?

Lower temperature makes the model stick more closely to the retrieved context and less likely to generate plausible-sounding but unsupported claims. The system prompt also explicitly instructs the model: *"If the context does not contain enough information, say so — do not guess."*

### Context block format

Each retrieved chunk is formatted for the LLM as:

```
[SOURCE 1: paper_name | Page: 4 | Score: 0.87]
<chunk text here>

---

[SOURCE 2: paper_name | Page: 7 | Score: 0.72]
<chunk text here>
```

This structured format makes it easy for Gemini to write precise inline citations.

---


## Evaluation Metrics

Three metrics are calculated automatically for every answer, with no ground-truth labels required:

| Metric | How it works | Ideal value |
|---|---|---|
| **Context Utilisation** | Fraction of answer sentences that contain a 5-gram match from retrieved context. Measures how grounded the answer is. | > 65% |
| **Retrieval Relevance** | Average keyword overlap between the query and retrieved excerpts (stop words excluded). Measures retrieval quality. | > 50% |
| **Source Diversity** | Unique papers ÷ total sources. A score of 1.0 means each source comes from a different paper. | Depends on query |
| **Overall Quality** | Simple average of the three scores above. | > 55% |

The UI also detects uncertainty hedges in the answer text (`"I think"`, `"possibly"`, `"I don't know"`) and flags them with a warning banner — a low-cost signal that the retrieval may have missed relevant content.

---

## Troubleshooting

**HTML still showing in citations**
The FAISS index was built before the cleaner fix. Click **⊘ Clear Knowledge Base** in the sidebar and re-index your PDFs. The auto-detection at startup should catch this and prompt you automatically.

**`GOOGLE_API_KEY not set`**
Add the key to `.env` or paste it in the sidebar. The app reads `.env` at startup via `python-dotenv`.

**`ModuleNotFoundError`**
```bash
pip install -r requirements.txt
```

**Voice button not working**
The microphone requires **Chrome or Edge**. Firefox does not support the Web Speech API. Also check that `SARVAM_API_KEY` is set.

**TTS plays no sound**
Check that your browser tab is not muted. Also verify the Sarvam key is correct — the app will show an error message if the API call fails.

**Slow first run**
The embedding model (`all-MiniLM-L6-v2`, ~22 MB) and cross-encoder (`ms-marco-MiniLM-L-6-v2`, ~85 MB) download once on first use, then load from `~/.cache/huggingface/` instantly on subsequent runs.

**PDF extracts garbled text**
Some PDFs are scanned images rather than text-based. For OCR support, install `pytesseract` + `pdf2image` and switch `PyPDFLoader` to `UnstructuredPDFLoader` in `src/ingestion/pdf_loader.py`.

**FAISS index marked corrupt and auto-cleared**
The yellow banner means the saved index contained HTML-contaminated chunks. Re-upload and re-index — the new cleaner guarantees this won't happen again.

---

## Interview Talking Points

**Q: Why use hybrid search instead of pure semantic search?**
Semantic embeddings capture meaning but can miss exact technical terms — model names, dataset identifiers, equation numbers. BM25 catches those exact matches. The 70/30 weighted fusion consistently beats either approach alone. This is also how production search engines like Elasticsearch work.

**Q: Why cross-encoder reranking after retrieval?**
First-stage retrieval (FAISS + BM25) is optimised for recall — it casts a wide net. Cross-encoders see (query, document) jointly and produce far more accurate relevance scores than bi-encoders. We only run it on the 10 candidates (not the full corpus) to keep it fast. This two-stage retrieve-then-rerank pattern is standard in production RAG systems.

**Q: How do you prevent hallucination?**
Three mechanisms: (1) Temperature 0.2 — less creative. (2) Strict system prompt — "use ONLY the provided context; if the answer isn't there, say so." (3) Structured SOURCE blocks in the context — the LLM must write citations, which anchors it to the retrieved text.

**Q: Why sentence-transformers instead of Gemini embeddings?**
Running embeddings locally avoids API quota limits, adds zero latency, and works offline. The model is swappable by changing one line in `config.py`. The abstraction is the point — good engineering means the LLM and embedding model are interchangeable.

**Q: Why FAISS instead of a vector database like Pinecone?**
FAISS runs 100% locally with no signup, no API, no network dependency. It persists to two files and loads in milliseconds. For a project of this scale (hundreds to thousands of chunks), FAISS is the right tool. Production systems would use a managed vector DB for scale, multi-tenancy, and metadata filtering.

**Q: Why does the FAISS index need an integrity check?**
RAG systems can silently degrade when the index is built from corrupted data. If HTML markup enters `page_content` before embedding, the index stores noisy vectors and every retrieval returns garbage-contaminated excerpts. Detecting and auto-discarding corrupt indexes at startup prevents silent, hard-to-diagnose quality regressions.

---

## Potential Extensions

- **OCR support** — Tesseract + pdf2image for scanned PDFs
- **Conversation memory** — Multi-turn Q&A with `ConversationBufferWindowMemory`
- **Streaming output** — Stream Gemini responses token-by-token with `st.write_stream()`
- **RAGAS evaluation** — Automated faithfulness + answer relevancy scoring with ground truth
- **ColBERT reranking** — Late-interaction neural reranker (state-of-the-art precision)
- **Figure and table extraction** — `pdfplumber` for structured data, vision model for figures
- **Cloud deployment** — Dockerise + deploy to GCP Cloud Run or AWS Lambda
- **Multi-user support** — Per-user FAISS indexes with a managed vector DB (Qdrant, Weaviate)
- **Citation graph** — Visualise which paper sections are cited most frequently

---

## License

MIT — free to use, modify, and share.