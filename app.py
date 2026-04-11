"""
app.py — Streamlit UI for the Research Paper Q&A Assistant.

Design philosophy:
  - Clean, dark-themed academic interface
  - Sidebar for PDF management, main panel for Q&A
  - Expandable sections for citations and evaluation metrics
  - Session state caches the pipeline so the index persists across re-runs

Run with:  streamlit run app.py
"""

import os
import sys
import logging
from pathlib import Path

import streamlit as st

# ── Path setup (allow imports from project root) ──────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))

from src.utils.helpers import setup_logging, save_uploaded_file
from config import RAW_DIR, VECTORSTORE_DIR

setup_logging(logging.INFO)

# ── Page config (must be first Streamlit call) ────────────────────────────────
st.set_page_config(
    page_title="Research Paper Q&A",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  /* Import fonts */
  @import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Mono:wght@400;500&family=Inter:wght@300;400;500;600&display=swap');

  /* Root palette */
  :root {
    --bg:        #0e1117;
    --surface:   #161b27;
    --surface2:  #1e2537;
    --accent:    #4f8ef7;
    --accent2:   #7ee8a2;
    --warn:      #f7c94f;
    --text:      #e2e8f0;
    --muted:     #8896b3;
    --border:    #2a3450;
  }

  /* Global */
  html, body, [data-testid="stAppViewContainer"] {
    background-color: var(--bg) !important;
    color: var(--text);
    font-family: 'Inter', sans-serif;
  }

  /* Header */
  .app-header {
    background: linear-gradient(135deg, #1a2240 0%, #0e1117 100%);
    border-bottom: 1px solid var(--border);
    padding: 2rem 2rem 1.5rem;
    margin: -1rem -1rem 2rem;
  }
  .app-title {
    font-family: 'DM Serif Display', serif;
    font-size: 2.4rem;
    color: #ffffff;
    letter-spacing: -0.02em;
    margin: 0;
  }
  .app-subtitle {
    color: var(--muted);
    font-size: 0.95rem;
    margin-top: 0.3rem;
    font-weight: 300;
  }

  /* Cards */
  .card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1.5rem;
    margin-bottom: 1rem;
  }
  .card-accent {
    border-left: 3px solid var(--accent);
  }

  /* Answer box */
  .answer-box {
    background: var(--surface);
    border: 1px solid var(--border);
    border-left: 4px solid var(--accent2);
    border-radius: 0 12px 12px 0;
    padding: 1.5rem 2rem;
    margin: 1rem 0;
    font-size: 1rem;
    line-height: 1.75;
  }

  /* Source chips */
  .source-chip {
    display: inline-block;
    background: var(--surface2);
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 0.2rem 0.6rem;
    font-family: 'DM Mono', monospace;
    font-size: 0.78rem;
    color: var(--accent);
    margin: 0.2rem 0.2rem 0 0;
  }

  /* Metric pills */
  .metric-good  { color: #7ee8a2; font-weight: 600; }
  .metric-med   { color: #f7c94f; font-weight: 600; }
  .metric-bad   { color: #f87171; font-weight: 600; }

  /* Section labels */
  .section-label {
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: var(--muted);
    margin-bottom: 0.6rem;
  }

  /* Sidebar */
  [data-testid="stSidebar"] {
    background: var(--surface) !important;
    border-right: 1px solid var(--border);
  }

  /* Input */
  .stTextArea textarea, .stTextInput input {
    background: var(--surface2) !important;
    color: var(--text) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    font-family: 'Inter', sans-serif !important;
  }

  /* Buttons */
  .stButton > button {
    background: var(--accent) !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 500 !important;
    padding: 0.5rem 1.5rem !important;
    transition: opacity 0.15s;
  }
  .stButton > button:hover { opacity: 0.85; }

  /* Expander */
  .streamlit-expanderHeader {
    background: var(--surface2) !important;
    border-radius: 8px !important;
    color: var(--text) !important;
    font-size: 0.9rem !important;
  }

  /* Progress bar */
  .stProgress > div > div { background: var(--accent) !important; }

  /* Divider */
  hr { border-color: var(--border) !important; }

  /* Hide Streamlit branding */
  #MainMenu, footer, header { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# ── Session state initialisation ──────────────────────────────────────────────

@st.cache_resource(show_spinner="Loading AI pipeline …")
def get_pipeline():
    """
    Load or restore the RAG pipeline (cached across all sessions).

    st.cache_resource keeps this alive for the entire server process lifetime.
    On first call it tries to load a saved FAISS index; falls back to fresh.
    """
    from src.pipeline import try_load_existing_index
    return try_load_existing_index()


pipeline = get_pipeline()

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []   # list of {question, result} dicts

if "processing" not in st.session_state:
    st.session_state.processing = False


# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="app-header">
  <div class="app-title">🔬 Research Paper Q&A</div>
  <div class="app-subtitle">
    RAG · Gemini Flash 2.5 · Hybrid Retrieval · Cross-Encoder Reranking
  </div>
</div>
""", unsafe_allow_html=True)


# ── Sidebar — Paper Management ────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📄 Knowledge Base")

    # API key input
    api_key = st.text_input(
        "Google API Key",
        type="password",
        placeholder="AIza…",
        help="Get yours at https://aistudio.google.com/app/apikey",
    )
    if api_key:
        os.environ["GOOGLE_API_KEY"] = api_key

    st.divider()

    # File uploader
    uploaded_files = st.file_uploader(
        "Upload Research Papers",
        type=["pdf"],
        accept_multiple_files=True,
        help="Upload one or more PDF research papers.",
    )

    ingest_btn = st.button("⚡ Process & Index PDFs", use_container_width=True)

    if ingest_btn and uploaded_files:
        if not api_key and not os.getenv("GOOGLE_API_KEY"):
            st.warning("⚠️ Please enter your Google API key first.")
        else:
            saved_paths = []
            for uf in uploaded_files:
                path = save_uploaded_file(uf, RAW_DIR)
                saved_paths.append(path)

            with st.spinner("Ingesting PDFs → cleaning → chunking → embedding …"):
                try:
                    result = pipeline.ingest_pdfs(saved_paths)
                    if result.get("new_chunks", 0) > 0:
                        st.success(
                            f"✅ Indexed **{result['new_chunks']}** new chunks "
                            f"from **{len(uploaded_files)}** paper(s)."
                        )
                    else:
                        st.info(result.get("message", "Nothing new to index."))
                except Exception as e:
                    st.error(f"Ingestion failed: {e}")

    st.divider()

    # Knowledge base status
    st.markdown("#### 📊 Knowledge Base Status")
    if pipeline.is_ready():
        stats = pipeline.get_stats()
        st.metric("Total Chunks", stats.get("total_chunks", 0))
        st.metric("Papers Indexed", stats.get("unique_papers", 0))

        with st.expander("Per-paper chunk counts"):
            for paper, count in stats.get("per_paper", {}).items():
                st.markdown(f"- **{paper}**: {count} chunks")
    else:
        st.info("No papers indexed yet. Upload PDFs above.")

    st.divider()

    # Reset button
    if st.button("🗑️ Clear Knowledge Base", use_container_width=True):
        from src.utils.helpers import clear_directory
        pipeline.reset()
        clear_directory(VECTORSTORE_DIR)
        st.session_state.chat_history = []
        st.success("Knowledge base cleared.")
        st.rerun()


# ── Main Panel — Q&A ──────────────────────────────────────────────────────────
col_main, col_info = st.columns([3, 1])

with col_main:
    # Question input
    st.markdown('<div class="section-label">Ask a question</div>', unsafe_allow_html=True)
    user_question = st.text_area(
        label="question",
        label_visibility="collapsed",
        placeholder=(
            "e.g.  What attention mechanism does the Transformer use?\n"
            "      How does BERT differ from GPT?\n"
            "      Summarise the key contributions of the paper."
        ),
        height=120,
    )

    ask_col, clear_col = st.columns([4, 1])
    with ask_col:
        ask_btn = st.button("🔍 Ask", use_container_width=True)
    with clear_col:
        if st.button("Clear", use_container_width=True):
            st.session_state.chat_history = []
            st.rerun()

    # ── Answer generation ─────────────────────────────────────────────────────
    if ask_btn and user_question.strip():
        if not pipeline.is_ready():
            st.warning("⚠️ Please upload and index at least one PDF first.")
        elif not (api_key or os.getenv("GOOGLE_API_KEY")):
            st.warning("⚠️ Please enter your Google API key in the sidebar.")
        else:
            with st.spinner("Retrieving · Reranking · Generating …"):
                try:
                    result = pipeline.query(user_question.strip())
                    st.session_state.chat_history.insert(0, {
                        "question": user_question.strip(),
                        "result":   result,
                    })
                except Exception as e:
                    st.error(f"Query failed: {e}")
                    st.exception(e)

    # ── Chat history display ──────────────────────────────────────────────────
    for i, entry in enumerate(st.session_state.chat_history):
        q      = entry["question"]
        result = entry["result"]
        eval_m = result["evaluation"]

        # Question
        st.markdown(f"""
        <div class="card" style="margin-top:1.5rem;">
          <div class="section-label">Question</div>
          <div style="font-size:1.05rem;font-weight:500;">{q}</div>
        </div>
        """, unsafe_allow_html=True)

        # Answer
        st.markdown('<div class="section-label">Answer</div>', unsafe_allow_html=True)
        st.markdown(
            f'<div class="answer-box">{result["answer"]}</div>',
            unsafe_allow_html=True,
        )

        # Quick source chips
        source_chips = " ".join(
            f'<span class="source-chip">{s["paper_name"]}, p.{s["page_number"]}</span>'
            for s in result["sources"]
        )
        st.markdown(f"**Sources:** {source_chips}", unsafe_allow_html=True)

        # Expandable: detailed citations
        with st.expander(f"📎 View {len(result['sources'])} cited passages", expanded=(i == 0)):
            for j, source in enumerate(result["sources"], 1):
                st.markdown(f"""
**[{j}] {source['paper_name']}** — Page {source['page_number']}
*Section: {source['section'] or 'General'}  ·  Relevance: `{source['score']}`*

> {source['excerpt']}
""")
                if j < len(result["sources"]):
                    st.divider()

        # Expandable: evaluation metrics
        with st.expander("📊 Evaluation metrics"):
            m1, m2, m3, m4 = st.columns(4)

            def colour(v):
                if v >= 0.65: return "metric-good"
                if v >= 0.35: return "metric-med"
                return "metric-bad"

            m1.markdown(f"""
**Context Utilisation**
<span class="{colour(eval_m['context_utilisation'])}">{eval_m['context_utilisation']:.0%}</span>
""", unsafe_allow_html=True)

            m2.markdown(f"""
**Retrieval Relevance**
<span class="{colour(eval_m['retrieval_relevance'])}">{eval_m['retrieval_relevance']:.0%}</span>
""", unsafe_allow_html=True)

            m3.markdown(f"""
**Source Diversity**
<span class="{colour(eval_m['source_diversity'])}">{eval_m['source_diversity']:.0%}</span>
""", unsafe_allow_html=True)

            m4.markdown(f"""
**Overall Quality**
<span class="{colour(eval_m['overall_quality'])}">{eval_m['overall_quality']:.0%}</span>
""", unsafe_allow_html=True)

            if eval_m["uncertainty_flags"]:
                st.warning(
                    f"⚠️ Uncertainty detected: {', '.join(eval_m['uncertainty_flags'][:3])}"
                )

        st.divider()


# ── Right info panel ──────────────────────────────────────────────────────────
with col_info:
    st.markdown("""
    <div class="card card-accent">
      <div class="section-label">Pipeline</div>
      <div style="font-size:0.82rem;line-height:1.8;color:#8896b3;">
        📥 PyPDF Loader<br>
        🧹 Text Cleaner<br>
        ✂️ Smart Chunker<br>
        🧠 MiniLM Embedder<br>
        🗃️ FAISS Index<br>
        🔍 Hybrid Retrieval<br>
        🔥 Cross-Encoder Rerank<br>
        ✨ Gemini Flash 2.5<br>
        📊 Auto Evaluator
      </div>
    </div>
    """, unsafe_allow_html=True)

    if pipeline.is_ready():
        st.markdown("""
        <div class="card" style="background:#0e2a1a;border-color:#2a5a3a;">
          <div style="color:#7ee8a2;font-weight:600;">✓ Ready</div>
          <div style="color:#8896b3;font-size:0.8rem;">Knowledge base active</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="card" style="background:#2a1a0e;border-color:#5a3a2a;">
          <div style="color:#f7c94f;font-weight:600;">⏳ Awaiting PDFs</div>
          <div style="color:#8896b3;font-size:0.8rem;">Upload in sidebar</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
    <div class="card" style="margin-top:0.5rem;">
      <div class="section-label">Tips</div>
      <div style="font-size:0.8rem;color:#8896b3;line-height:1.7;">
        • Ask specific, focused questions<br>
        • Name the paper for targeted answers<br>
        • "Compare X and Y" works across papers<br>
        • Citations are auto-included
      </div>
    </div>
    """, unsafe_allow_html=True)
