# 🔬 Research Paper Q&A Assistant

A production-grade **Retrieval-Augmented Generation (RAG)** system that lets you upload research papers (PDFs) and ask questions about them. Answers are grounded in the papers, include precise citations, and are evaluated automatically for quality.

Built with **LangChain · Gemini Flash 2.5 · FAISS · Streamlit**.

---

## 🏗️ Architecture Overview

```
User uploads PDFs
        │
        ▼
┌─────────────────┐
│   PDF Ingestion  │  PyPDFLoader — extracts text page by page
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Preprocessing  │  Remove headers, fix ligatures, collapse whitespace
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Smart Chunking  │  Section-aware split → fixed-size fallback (800 chars, 150 overlap)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   Embeddings    │  sentence-transformers/all-MiniLM-L6-v2 (local, cached)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  FAISS Index    │  Persisted to disk — survives app restarts
└────────┬────────┘
         │
    [At query time]
         │
         ▼
┌─────────────────────────────────────┐
│   Hybrid Retrieval (top 10)         │
│   70% semantic (FAISS) +            │
│   30% keyword (BM25)                │
└────────────────┬────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────┐
│   Cross-Encoder Reranking (top 5)   │
│   ms-marco-MiniLM-L-6-v2            │
└────────────────┬────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────┐
│   Gemini Flash 2.5 Generation       │
│   Strict context-only prompt        │
│   + citation format enforcement     │
└────────────────┬────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────┐
│   Automatic Evaluation              │
│   • Context utilisation score       │
│   • Retrieval relevance score       │
│   • Source diversity score          │
└─────────────────────────────────────┘
```

---

## 📁 Project Structure

```
research_qa/
│
├── app.py                        # Streamlit UI entry point
├── config.py                     # All hyperparameters & paths in one place
├── requirements.txt
├── .env.example                  # Copy → .env and add your API key
├── README.md
│
├── data/
│   ├── raw/                      # Uploaded PDFs saved here
│   └── processed/
│       └── vectorstore/          # Persisted FAISS index
│
└── src/
    ├── pipeline.py               # ← Orchestrator: wires all modules together
    │
    ├── ingestion/
    │   └── pdf_loader.py         # PyPDFLoader wrapper; extracts page-level docs
    │
    ├── preprocessing/
    │   └── cleaner.py            # Noise removal: headers, ligatures, whitespace
    │
    ├── chunking/
    │   └── chunker.py            # Section-aware + RecursiveCharacterTextSplitter
    │
    ├── embeddings/
    │   └── embedder.py           # HuggingFace MiniLM; singleton + caching
    │
    ├── vectorstore/
    │   └── faiss_store.py        # Build / persist / load / extend FAISS index
    │
    ├── retrieval/
    │   └── retriever.py          # Semantic search + BM25 + hybrid fusion
    │
    ├── reranking/
    │   └── reranker.py           # Cross-encoder reranker; lexical fallback
    │
    ├── generation/
    │   └── generator.py          # Prompt engineering + Gemini Flash 2.5 call
    │
    ├── evaluation/
    │   └── evaluator.py          # Context utilisation, relevance, diversity
    │
    └── utils/
        └── helpers.py            # Shared utilities (file I/O, formatting, stats)
```

---

## 🚀 Quick Start

### 1. Clone the repository

```bash
git clone https://github.com/yourusername/research-qa-assistant.git
cd research-qa-assistant
```

### 2. Create and activate a virtual environment

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

> **Note:** The first run will download two models automatically:
> - `all-MiniLM-L6-v2` (~22 MB) for embeddings
> - `cross-encoder/ms-marco-MiniLM-L-6-v2` (~85 MB) for reranking
>
> Both are cached in `~/.cache/huggingface/` after the first download.

### 4. Set your Google API key

Option A — via `.env` file (recommended for development):
```bash
cp .env.example .env
# Edit .env and paste your key
GOOGLE_API_KEY=AIza...
```

Option B — via the sidebar input in the Streamlit UI.

Get a free API key at: https://aistudio.google.com/app/apikey

### 5. Run the app

```bash
streamlit run app.py
```

Open http://localhost:8501 in your browser.

---

## 🎮 How to Use

1. **Enter your Google API key** in the sidebar.
2. **Upload research paper PDFs** using the file uploader.
3. Click **"Process & Index PDFs"** — this runs the full ingestion pipeline.
4. **Type a question** in the main panel and click **"Ask"**.
5. View the **answer, citations, and evaluation metrics**.

### Example questions

```
What is the main contribution of this paper?
How does the attention mechanism work in the Transformer?
What datasets were used for evaluation?
Compare the results across the papers you have.
What are the limitations mentioned by the authors?
```

---

## 🧠 Key Design Decisions (Interview Talking Points)

### Why Hybrid Search?
Pure semantic search (embeddings) excels at capturing meaning but misses exact keyword matches — crucial for model names, dataset identifiers, or specific numerical results. BM25 is the classical TF-IDF ranker that catches those exact matches. Combining both (70/30 split) gives the best of both worlds.

### Why Cross-Encoder Reranking?
First-stage retrieval (FAISS/BM25) is optimised for **recall** — it casts a wide net. But passing 10 noisy chunks to the LLM degrades answer quality. The cross-encoder sees (query, passage) jointly and produces a much more accurate relevance score. We rerank 10 → 5 chunks, dramatically improving **precision**.

### Why is Temperature set to 0.2?
Lower temperature → less creative, more faithful. In a RAG system you want the LLM to paraphrase the retrieved context, not invent new content. 0.2 is the sweet spot between readability and faithfulness.

### Why sentence-transformers over Gemini embeddings?
Running embeddings locally avoids API quota limits during demos and adds zero latency overhead. The pipeline is fully abstracted — swapping to Gemini embeddings requires changing one line in `config.py`.

### Why FAISS over a cloud vector DB?
For a fresher portfolio project running locally, FAISS is ideal: no signup, no API key, no network dependency, and it's the industry-standard library used by Meta AI. The same FAISS index format is used in production systems.

---

## 📊 Evaluation Metrics Explained

| Metric | What it measures | Good value |
|--------|-----------------|------------|
| **Context Utilisation** | Fraction of answer sentences that contain 5-gram matches from retrieved context | > 0.65 |
| **Retrieval Relevance** | Average keyword overlap between query and retrieved excerpts | > 0.50 |
| **Source Diversity** | Unique papers / total sources (1.0 = each source from a different paper) | depends on query |
| **Overall Quality** | Average of the three scores above | > 0.55 |

---

## ⚙️ Configuration Reference (`config.py`)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `LLM_MODEL_NAME` | `gemini-2.5-flash-preview-05-20` | Gemini model to use |
| `EMBEDDING_MODEL_NAME` | `all-MiniLM-L6-v2` | Local embedding model |
| `CHUNK_SIZE` | `800` | Characters per chunk |
| `CHUNK_OVERLAP` | `150` | Overlap between chunks |
| `TOP_K_RETRIEVE` | `10` | Chunks fetched from FAISS |
| `TOP_K_RERANK` | `5` | Chunks passed to LLM after reranking |
| `SEMANTIC_WEIGHT` | `0.7` | Weight of semantic vs BM25 in hybrid search |
| `TEMPERATURE` | `0.2` | LLM temperature (lower = less hallucination) |

---

## 🔧 Troubleshooting

**`ModuleNotFoundError`**
```bash
pip install -r requirements.txt
```

**`GOOGLE_API_KEY not set`**
Add the key to `.env` or paste it in the sidebar.

**Slow first run**
The embedding and reranking models download on first use. Subsequent runs are instant (models cached).

**PDF text extraction is poor**
Some PDFs are scanned images. For OCR support, install `pytesseract` and `pdf2image` and use `UnstructuredPDFLoader` instead of `PyPDFLoader` in `src/ingestion/pdf_loader.py`.

**FAISS index seems stale after adding new papers**
Click "Clear Knowledge Base" in the sidebar and re-index all papers. The index is rebuilt correctly from scratch.

---

## 🛣️ Potential Extensions (for interviews)

- **OCR support** — handle scanned PDFs with Tesseract
- **Conversation memory** — multi-turn Q&A using LangChain `ConversationBufferMemory`
- **Streaming output** — stream Gemini responses token-by-token
- **RAGAS evaluation** — automated faithfulness / answer relevancy scoring
- **ColBERT reranking** — late-interaction neural reranker (state-of-the-art)
- **Multi-modal** — extract and query figures/tables from PDFs
- **Cloud deployment** — Dockerise + deploy to GCP Cloud Run

---

## 📄 License

MIT — free to use, modify, and share.
