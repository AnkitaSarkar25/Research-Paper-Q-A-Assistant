"""
app.py — Research Paper Q&A Assistant  (Lexis)
Aesthetic: Premium editorial / research-lab
  - Warm ivory main canvas with deep ink sidebar
  - Playfair Display headings · Lora body · JetBrains Mono code
  - Glowing teal accent, warm amber citations
  - Glassmorphism cards, animated answer panel
"""

import os
import sys
import logging
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).parent))

from src.utils.helpers import setup_logging, save_uploaded_file
from config import RAW_DIR, VECTORSTORE_DIR

setup_logging(logging.INFO)

st.set_page_config(
    page_title="Lexis — Research Q&A",
    page_icon="◎",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ═══════════════════════════════════════════════════════════════════════════════
#  DESIGN SYSTEM
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:ital,wght@0,400;0,600;0,700;1,400;1,600&family=Lora:ital,wght@0,400;0,500;1,400&family=JetBrains+Mono:wght@400;500&family=Outfit:wght@300;400;500;600&display=swap');

/* ── TOKENS ── */
:root {
  --canvas:       #FAFAF7;
  --canvas2:      #F3F3EE;
  --ink:          #1A1A2E;
  --ink-soft:     #3D3D5C;
  --ink-muted:    #7C7C9A;
  --sidebar-bg:   #0F1117;
  --sidebar2:     #161B27;
  --teal:         #00C9A7;
  --teal-dim:     #00876f;
  --teal-glow:    rgba(0,201,167,0.15);
  --amber:        #F59E0B;
  --amber-dim:    rgba(245,158,11,0.10);
  --rose:         #F43F5E;
  --border:       rgba(26,26,46,0.10);
  --border-dark:  rgba(255,255,255,0.07);
  --shadow-sm:    0 1px 3px rgba(26,26,46,0.06),0 1px 2px rgba(26,26,46,0.04);
  --shadow-md:    0 4px 16px rgba(26,26,46,0.08),0 2px 6px rgba(26,26,46,0.04);
}

/* ── BASE ── */
html, body { background-color: var(--canvas) !important; }
[data-testid="stAppViewContainer"] > .main { background-color: var(--canvas) !important; }
.stApp { font-family: 'Outfit', sans-serif; color: var(--ink); }
* { box-sizing: border-box; }
.block-container { padding: 0 2.5rem 3rem !important; max-width: 1400px !important; }

/* ── SIDEBAR ── */
[data-testid="stSidebar"],
[data-testid="stSidebar"] > div:first-child {
  background-color: var(--sidebar-bg) !important;
  border-right: 1px solid var(--border-dark) !important;
}
[data-testid="stSidebar"] * { color: #C8C8DC !important; }
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 { color: #EEEEF5 !important; }
[data-testid="stSidebar"] input[type="password"],
[data-testid="stSidebar"] input[type="text"] {
  background: var(--sidebar2) !important;
  border: 1px solid rgba(255,255,255,0.08) !important;
  border-radius: 8px !important;
  color: #EEEEF5 !important;
  font-family: 'JetBrains Mono', monospace !important;
  font-size: 0.78rem !important;
}
[data-testid="stSidebar"] input:focus {
  border-color: var(--teal) !important;
  box-shadow: 0 0 0 3px var(--teal-glow) !important;
  outline: none !important;
}
[data-testid="stSidebar"] .stButton > button {
  background: linear-gradient(135deg, var(--teal), var(--teal-dim)) !important;
  color: #fff !important;
  border: none !important;
  border-radius: 8px !important;
  font-family: 'Outfit', sans-serif !important;
  font-weight: 600 !important;
  font-size: 0.85rem !important;
  padding: 10px 16px !important;
  width: 100% !important;
  transition: all 0.2s ease !important;
  box-shadow: 0 2px 12px rgba(0,201,167,0.25) !important;
  cursor: pointer !important;
}
[data-testid="stSidebar"] .stButton > button:hover {
  transform: translateY(-1px) !important;
  box-shadow: 0 4px 20px rgba(0,201,167,0.35) !important;
}
[data-testid="stSidebar"] hr { border-color: var(--border-dark) !important; margin: 14px 0 !important; }
[data-testid="stSidebar"] [data-testid="stMetric"] {
  background: var(--sidebar2) !important;
  border: 1px solid var(--border-dark) !important;
  border-radius: 8px !important;
  padding: 10px 12px !important;
}
[data-testid="stSidebar"] [data-testid="stMetricLabel"] { color: #7C7CAA !important; font-size: 0.7rem !important; }
[data-testid="stSidebar"] [data-testid="stMetricValue"] { color: var(--teal) !important; font-size: 1.35rem !important; font-weight: 700 !important; }
[data-testid="stSidebar"] .streamlit-expanderHeader {
  background: var(--sidebar2) !important;
  border: 1px solid var(--border-dark) !important;
  border-radius: 8px !important;
  color: #C8C8DC !important;
  font-size: 0.8rem !important;
}
[data-testid="stSidebar"] [data-testid="stFileUploader"] {
  background: var(--sidebar2) !important;
  border: 1.5px dashed rgba(255,255,255,0.1) !important;
  border-radius: 12px !important;
  padding: 14px !important;
  transition: border-color 0.2s !important;
}
[data-testid="stSidebar"] [data-testid="stFileUploader"]:hover {
  border-color: rgba(0,201,167,0.3) !important;
}
[data-testid="collapsedControl"],button[kind="headerNoPadding"] {
  display: flex !important; visibility: visible !important; opacity: 1 !important;
}

/* ── SIDEBAR COMPONENTS ── */
.sb-brand { display:flex;align-items:center;gap:10px;padding:6px 0 18px;border-bottom:1px solid var(--border-dark);margin-bottom:16px; }
.sb-brand-icon { width:32px;height:32px;background:linear-gradient(135deg,var(--teal),var(--teal-dim));border-radius:8px;display:flex;align-items:center;justify-content:center;font-size:15px;color:#fff;flex-shrink:0;box-shadow:0 0 14px var(--teal-glow); }
.sb-brand-name { font-family:'Playfair Display',serif;font-size:1.2rem;font-weight:700;color:#EEEEF5 !important;letter-spacing:-0.02em; }
.sb-brand-sub  { font-size:0.62rem;color:var(--teal) !important;letter-spacing:0.14em;text-transform:uppercase;font-weight:500; }
.sb-section    { font-size:0.62rem;letter-spacing:0.14em;text-transform:uppercase;color:#5A5A7A !important;font-weight:700;margin:16px 0 8px; }
.sb-status     { display:flex;align-items:center;gap:8px;padding:9px 12px;border-radius:8px;font-size:0.8rem;font-weight:500;margin:10px 0; }
.sb-status.ready   { background:rgba(0,201,167,0.08);border:1px solid rgba(0,201,167,0.2);color:var(--teal) !important; }
.sb-status.waiting { background:rgba(245,158,11,0.08);border:1px solid rgba(245,158,11,0.2);color:var(--amber) !important; }
.sb-dot { width:7px;height:7px;border-radius:50%;flex-shrink:0; }
.sb-dot.ready   { background:var(--teal);box-shadow:0 0 6px var(--teal);animation:pulse 2s infinite; }
.sb-dot.waiting { background:var(--amber); }
.sb-hint { font-size:0.7rem;color:#5A5A7A !important;margin-top:-6px;margin-bottom:6px; }
.sb-clear { font-size:0.78rem !important;background:rgba(244,63,94,0.08) !important;color:#F43F5E !important;border:1px solid rgba(244,63,94,0.2) !important;box-shadow:none !important; }
.sb-paper-row { display:flex;justify-content:space-between;font-size:0.76rem;padding:5px 0;border-bottom:1px solid rgba(255,255,255,0.04); }
.sb-paper-name { color:#C8C8DC !important; }
.sb-paper-cnt  { color:var(--teal) !important;font-family:'JetBrains Mono',monospace; }

/* ── HERO ── */
.hero { position:relative;padding:2.8rem 0 2.2rem;margin-bottom:1.8rem;overflow:hidden; }
.hero::before { content:'';position:absolute;top:-80px;left:-60px;width:480px;height:320px;background:radial-gradient(ellipse,rgba(0,201,167,0.05) 0%,transparent 70%);pointer-events:none; }
.hero::after  { content:'';position:absolute;bottom:0;left:0;right:0;height:1px;background:linear-gradient(90deg,transparent,var(--border) 30%,var(--border) 70%,transparent); }
.hero-eye  { font-family:'JetBrains Mono',monospace;font-size:0.68rem;letter-spacing:0.18em;color:var(--teal);text-transform:uppercase;margin-bottom:10px;display:flex;align-items:center;gap:8px; }
.hero-eye::before { content:'';display:inline-block;width:22px;height:1.5px;background:var(--teal); }
.hero-h1   { font-family:'Playfair Display',serif;font-size:clamp(2rem,4vw,3.2rem);font-weight:700;color:var(--ink);letter-spacing:-0.03em;line-height:1.1;margin:0 0 12px; }
.hero-h1 em { font-style:italic;background:linear-gradient(135deg,var(--teal),var(--teal-dim));-webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text; }
.hero-sub  { font-family:'Lora',serif;font-size:1rem;color:var(--ink-soft);line-height:1.65;max-width:540px;font-style:italic; }
.tech-pills { display:flex;flex-wrap:wrap;gap:7px;margin-top:16px; }
.tpill { font-family:'JetBrains Mono',monospace;font-size:0.67rem;color:var(--ink-muted);background:var(--canvas2);border:1px solid var(--border);border-radius:100px;padding:3px 10px;letter-spacing:0.03em; }

/* ── QUERY BOX ── */
.query-wrap { background:#fff;border:1px solid var(--border);border-radius:18px;padding:1.8rem;box-shadow:var(--shadow-md);margin-bottom:1.8rem;position:relative;overflow:hidden; }
.query-wrap::before { content:'';position:absolute;top:0;left:0;right:0;height:3px;background:linear-gradient(90deg,var(--teal),var(--teal-dim),transparent); }
.query-lbl { font-family:'Outfit',sans-serif;font-size:0.67rem;font-weight:700;letter-spacing:0.14em;text-transform:uppercase;color:var(--ink-muted);margin-bottom:8px; }
.stTextArea textarea {
  background: var(--canvas) !important;
  border: 1.5px solid var(--border) !important;
  border-radius: 12px !important;
  color: var(--ink) !important;
  font-family: 'Lora', serif !important;
  font-size: 0.97rem !important;
  line-height: 1.65 !important;
  padding: 14px 16px !important;
  transition: all 0.2s ease !important;
  resize: none !important;
}
.stTextArea textarea:focus {
  border-color: var(--teal) !important;
  box-shadow: 0 0 0 4px var(--teal-glow) !important;
  background: #fff !important;
  outline: none !important;
}
.stTextArea textarea::placeholder { color: var(--ink-muted) !important; font-style: italic !important; }
.main-btns .stButton > button {
  background: linear-gradient(135deg, var(--ink), #2D2D4E) !important;
  color: #fff !important; border: none !important; border-radius: 10px !important;
  font-family: 'Outfit', sans-serif !important; font-weight: 600 !important;
  font-size: 0.88rem !important; padding: 11px 20px !important;
  transition: all 0.2s ease !important;
  box-shadow: 0 2px 8px rgba(26,26,46,0.2) !important;
  width: 100% !important; cursor: pointer !important;
}
.main-btns .stButton > button:hover {
  transform: translateY(-1px) !important;
  box-shadow: 0 5px 18px rgba(26,26,46,0.3) !important;
}
.main-btns .stButton:last-child > button {
  background: var(--canvas2) !important;
  color: var(--ink-muted) !important;
  border: 1px solid var(--border) !important;
  box-shadow: none !important;
  font-weight: 500 !important;
}
.main-btns .stButton:last-child > button:hover {
  background: #EAEAE5 !important;
  transform: none !important;
  box-shadow: none !important;
}

/* ── QA BLOCKS ── */
.qa-block { margin-bottom:1.8rem; animation:slideUp 0.35s cubic-bezier(0.16,1,0.3,1) both; }
@keyframes slideUp { from{opacity:0;transform:translateY(14px)} to{opacity:1;transform:translateY(0)} }
.q-card { background:#fff;border:1px solid var(--border);border-radius:12px;padding:1.1rem 1.5rem;display:flex;align-items:flex-start;gap:12px;box-shadow:var(--shadow-sm);margin-bottom:2px; }
.q-icon { width:28px;height:28px;background:var(--canvas2);border:1px solid var(--border);border-radius:50%;display:flex;align-items:center;justify-content:center;font-family:'Playfair Display',serif;font-size:13px;color:var(--ink-muted);flex-shrink:0;margin-top:2px; }
.q-text { font-family:'Outfit',sans-serif;font-size:0.97rem;font-weight:500;color:var(--ink);line-height:1.5; }
.a-card { background:linear-gradient(160deg,#fff 0%,#FAFFF9 100%);border:1px solid rgba(0,201,167,0.18);border-radius:12px;padding:1.6rem 1.8rem;box-shadow:var(--shadow-md);position:relative;overflow:hidden;margin-top:2px; }
.a-card::before { content:'';position:absolute;top:0;left:0;width:4px;height:100%;background:linear-gradient(180deg,var(--teal),transparent); }
.a-hdr { display:flex;align-items:center;gap:8px;margin-bottom:14px;padding-bottom:10px;border-bottom:1px solid rgba(0,201,167,0.10); }
.a-badge { font-family:'JetBrains Mono',monospace;font-size:0.63rem;letter-spacing:0.12em;text-transform:uppercase;color:var(--teal);background:rgba(0,201,167,0.08);border:1px solid rgba(0,201,167,0.2);border-radius:100px;padding:3px 10px; }
.a-meta  { font-size:0.7rem;color:var(--ink-muted);margin-left:auto;font-family:'Outfit',sans-serif; }
.a-body  { font-family:'Lora',serif;font-size:0.95rem;line-height:1.82;color:var(--ink-soft); }
.srcs-row { display:flex;flex-wrap:wrap;gap:6px;margin-top:14px;padding-top:14px;border-top:1px solid rgba(0,201,167,0.10);align-items:center; }
.srcs-lbl { font-family:'Outfit',sans-serif;font-size:0.68rem;font-weight:700;letter-spacing:0.08em;text-transform:uppercase;color:var(--ink-muted);margin-right:4px; }
.src-chip { display:inline-flex;align-items:center;gap:5px;background:var(--amber-dim);border:1px solid rgba(245,158,11,0.22);border-radius:100px;padding:3px 10px 3px 8px;font-family:'JetBrains Mono',monospace;font-size:0.7rem;color:#92670B;transition:background 0.15s;cursor:default; }
.src-chip:hover { background:rgba(245,158,11,0.18); }
.src-dot { width:5px;height:5px;border-radius:50%;background:var(--amber);flex-shrink:0; }

/* ── CITATIONS ── */
.cite-wrap { background:#fff;border:1px solid var(--border);border-radius:12px;overflow:hidden;box-shadow:var(--shadow-sm); }
.cite-item { padding:1.1rem 1.4rem;border-bottom:1px solid var(--border);transition:background 0.15s; }
.cite-item:last-child { border-bottom:none; }
.cite-item:hover { background:var(--canvas); }
.cite-hdr { display:flex;align-items:center;gap:8px;margin-bottom:8px;flex-wrap:wrap; }
.cite-num { display:inline-flex;align-items:center;justify-content:center;width:20px;height:20px;background:var(--amber-dim);border:1px solid rgba(245,158,11,0.22);border-radius:50%;font-family:'JetBrains Mono',monospace;font-size:0.63rem;color:#92670B;font-weight:700;flex-shrink:0; }
.cite-paper { font-family:'Outfit',sans-serif;font-weight:600;font-size:0.86rem;color:var(--ink); }
.cite-tag { font-family:'JetBrains Mono',monospace;font-size:0.67rem;color:var(--ink-muted);background:var(--canvas2);border-radius:4px;padding:2px 7px; }
.cite-score { margin-left:auto;font-family:'JetBrains Mono',monospace;font-size:0.66rem;color:var(--teal-dim);background:rgba(0,201,167,0.06);border:1px solid rgba(0,201,167,0.15);border-radius:100px;padding:2px 8px; }
.cite-excerpt { font-family:'Lora',serif;font-size:0.83rem;color:var(--ink-soft);line-height:1.65;padding:9px 12px;background:var(--canvas);border-left:3px solid rgba(245,158,11,0.3);border-radius:0 6px 6px 0;font-style:italic;margin:0; }

/* ── EVAL ── */
.eval-grid { display:grid;grid-template-columns:repeat(4,1fr);gap:10px;padding:4px 0; }
.eval-cell { background:#fff;border:1px solid var(--border);border-radius:10px;padding:14px 12px;text-align:center; }
.eval-lbl { font-family:'Outfit',sans-serif;font-size:0.65rem;font-weight:700;letter-spacing:0.06em;text-transform:uppercase;color:var(--ink-muted);margin-bottom:8px; }
.eval-val { font-family:'Playfair Display',serif;font-size:1.55rem;font-weight:700;line-height:1; }
.ev-good { color:#059669; } .ev-med { color:#D97706; } .ev-bad { color:var(--rose); }
.eval-bar { height:3px;border-radius:2px;margin-top:8px;background:var(--canvas2);overflow:hidden; }
.eval-fill { height:100%;border-radius:2px;transition:width 0.6s cubic-bezier(0.16,1,0.3,1); }
.ef-good { background:#059669; } .ef-med { background:#D97706; } .ef-bad { background:var(--rose); }
.unc-warn { display:flex;align-items:center;gap:8px;background:rgba(245,158,11,0.06);border:1px solid rgba(245,158,11,0.2);border-radius:8px;padding:10px 14px;font-size:0.8rem;color:#92670B;margin-top:10px; }

/* ── INFO PANEL ── */
.info-card { background:#fff;border:1px solid var(--border);border-radius:14px;padding:1.3rem;margin-bottom:12px;box-shadow:var(--shadow-sm); }
.info-title { font-family:'Outfit',sans-serif;font-size:0.65rem;font-weight:700;letter-spacing:0.14em;text-transform:uppercase;color:var(--ink-muted);margin-bottom:14px;display:flex;align-items:center;gap:6px; }
.info-title::after { content:'';flex:1;height:1px;background:var(--border); }
.pipe-step { display:flex;align-items:flex-start;gap:10px;padding:7px 0;border-bottom:1px solid var(--border); }
.pipe-step:last-child { border-bottom:none; }
.pipe-icon { width:26px;height:26px;background:var(--canvas2);border-radius:6px;display:flex;align-items:center;justify-content:center;font-size:11px;flex-shrink:0;margin-top:1px; }
.pipe-name { font-weight:500;font-size:0.8rem;color:var(--ink);font-family:'Outfit',sans-serif; }
.pipe-desc { font-size:0.69rem;color:var(--ink-muted);margin-top:1px; }
.tip-item  { display:flex;gap:8px;font-size:0.78rem;color:var(--ink-soft);padding:4px 0;line-height:1.45; }
.tip-arrow { color:var(--teal);font-weight:700;flex-shrink:0;margin-top:1px; }

/* ── EMPTY STATE ── */
.empty { text-align:center;padding:3.5rem 2rem;color:var(--ink-muted); }
.empty-icon  { font-size:2.8rem;margin-bottom:14px;opacity:0.35; }
.empty-title { font-family:'Playfair Display',serif;font-size:1.25rem;font-weight:600;color:var(--ink-soft);margin-bottom:8px; }
.empty-desc  { font-family:'Lora',serif;font-size:0.86rem;font-style:italic;line-height:1.65;max-width:340px;margin:0 auto; }

/* ── STREAMLIT EXPANDER ── */
.streamlit-expanderHeader {
  background: var(--canvas2) !important; border: 1px solid var(--border) !important;
  border-radius: 10px !important; color: var(--ink) !important;
  font-family: 'Outfit',sans-serif !important; font-size:0.83rem !important;
  font-weight: 500 !important; padding:10px 14px !important;
  transition: background 0.15s !important;
}
.streamlit-expanderHeader:hover { background:#EAEAE5 !important; }
.streamlit-expanderContent { border:1px solid var(--border) !important;border-top:none !important;border-radius:0 0 10px 10px !important; }

/* ── MISC ── */
hr { border-color:var(--border) !important;margin:1.4rem 0 !important; }
.stSpinner > div { border-top-color:var(--teal) !important; }
#MainMenu,footer { visibility:hidden; }
[data-testid="stDeployButton"] { display:none; }
@keyframes pulse { 0%,100%{box-shadow:0 0 6px var(--teal)}50%{box-shadow:0 0 14px var(--teal),0 0 22px rgba(0,201,167,0.25)} }
</style>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
#  STATE & PIPELINE
# ═══════════════════════════════════════════════════════════════════════════════
@st.cache_resource(show_spinner="Initialising Lexis …")
def get_pipeline():
    from src.pipeline import try_load_existing_index
    return try_load_existing_index()

pipeline = get_pipeline()
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# ═══════════════════════════════════════════════════════════════════════════════
#  SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════════
with st.sidebar:

    # Brand mark
    st.markdown("""
    <div class="sb-brand">
      <div class="sb-brand-icon">◎</div>
      <div>
        <div class="sb-brand-name">Lexis</div>
        <div class="sb-brand-sub">Research Q&A</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # Live status
    if pipeline.is_ready():
        stats = pipeline.get_stats()
        st.markdown(
            f'<div class="sb-status ready">'
            f'<div class="sb-dot ready"></div>'
            f'{stats.get("unique_papers",0)} paper(s) · {stats.get("total_chunks",0)} chunks'
            f'</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div class="sb-status waiting"><div class="sb-dot waiting"></div>No papers indexed</div>',
            unsafe_allow_html=True,
        )

    # ── API Key ───────────────────────────────────────────────────────────────
    st.markdown('<div class="sb-section">API Configuration</div>', unsafe_allow_html=True)
    api_key = st.text_input(
        "key", label_visibility="collapsed",
        type="password", placeholder="Google API Key  (AIza…)",
        help="aistudio.google.com/app/apikey",
    )
    if api_key:
        os.environ["GOOGLE_API_KEY"] = api_key
        st.markdown('<div class="sb-hint" style="color:var(--teal) !important;">✓ Key active</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="sb-hint">Required to generate answers</div>', unsafe_allow_html=True)

    st.divider()

    # ── Upload ────────────────────────────────────────────────────────────────
    st.markdown('<div class="sb-section">Upload Papers</div>', unsafe_allow_html=True)
    uploaded_files = st.file_uploader(
        "files", label_visibility="collapsed",
        type=["pdf"], accept_multiple_files=True,
    )
    if uploaded_files:
        st.markdown(
            f'<div class="sb-hint">{len(uploaded_files)} file(s) selected</div>',
            unsafe_allow_html=True,
        )

    ingest_btn = st.button("⚡  Process & Index PDFs", use_container_width=True)
    if ingest_btn:
        if not uploaded_files:
            st.warning("Select PDFs first.")
        elif not api_key and not os.getenv("GOOGLE_API_KEY"):
            st.warning("Enter your Google API key first.")
        else:
            paths = [save_uploaded_file(uf, RAW_DIR) for uf in uploaded_files]
            with st.spinner("Cleaning · Chunking · Embedding …"):
                try:
                    res = pipeline.ingest_pdfs(paths)
                    if res.get("new_chunks", 0) > 0:
                        st.success(f"Indexed {res['new_chunks']} chunks from {len(uploaded_files)} paper(s).")
                        st.rerun()
                    else:
                        st.info(res.get("message", "Already indexed."))
                except Exception as exc:
                    st.error(f"Ingestion error: {exc}")

    st.divider()

    # ── Knowledge base ────────────────────────────────────────────────────────
    st.markdown('<div class="sb-section">Knowledge Base</div>', unsafe_allow_html=True)
    if pipeline.is_ready():
        stats = pipeline.get_stats()
        c1, c2 = st.columns(2)
        c1.metric("Chunks", stats.get("total_chunks", 0))
        c2.metric("Papers", stats.get("unique_papers", 0))

        if stats.get("per_paper"):
            with st.expander("Paper breakdown"):
                rows = "".join(
                    f'<div class="sb-paper-row">'
                    f'<span class="sb-paper-name">{p[:22]}{"…" if len(p)>22 else ""}</span>'
                    f'<span class="sb-paper-cnt">{n}</span>'
                    f'</div>'
                    for p, n in stats["per_paper"].items()
                )
                st.markdown(rows, unsafe_allow_html=True)
    else:
        st.markdown('<div class="sb-hint">Upload PDFs above to begin.</div>', unsafe_allow_html=True)

    st.divider()

    # Clear
    if st.button("⊘  Clear Knowledge Base", use_container_width=True):
        from src.utils.helpers import clear_directory
        pipeline.reset()
        clear_directory(VECTORSTORE_DIR)
        st.session_state.chat_history = []
        st.rerun()

# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN CONTENT
# ═══════════════════════════════════════════════════════════════════════════════
col_main, col_info = st.columns([11, 4], gap="large")

with col_main:

    # ── Hero ──────────────────────────────────────────────────────────────────
    st.markdown("""
    <div class="hero">
      <div class="hero-eye">RAG · Retrieval-Augmented Generation</div>
      <h1 class="hero-h1">Ask anything about<br>your <em>research papers</em></h1>
      <p class="hero-sub">Upload papers, ask questions. Lexis retrieves, reranks, and synthesises
        answers with precise citations — grounded only in your documents.</p>
      <div class="tech-pills">
        <span class="tpill">Gemini Flash 2.5</span>
        <span class="tpill">FAISS · BM25 Hybrid</span>
        <span class="tpill">Cross-Encoder Reranking</span>
        <span class="tpill">MiniLM Embeddings</span>
        <span class="tpill">Auto Evaluation</span>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Query panel ───────────────────────────────────────────────────────────
    st.markdown('<div class="query-wrap"><div class="query-lbl">Your question</div>', unsafe_allow_html=True)
    user_question = st.text_area(
        "q", label_visibility="collapsed", height=108,
        placeholder=(
            "e.g. What attention mechanism does the Transformer use?\n"
            "     How do the authors evaluate their model?\n"
            "     Compare the findings across uploaded papers."
        ),
    )
    st.markdown('<div class="main-btns">', unsafe_allow_html=True)
    bc1, bc2 = st.columns([5, 1])
    with bc1:
        ask_btn = st.button("Search & Answer →", use_container_width=True)
    with bc2:
        clear_btn = st.button("Clear", use_container_width=True)
    st.markdown('</div></div>', unsafe_allow_html=True)

    if clear_btn:
        st.session_state.chat_history = []
        st.rerun()

    # Answer
    if ask_btn and user_question.strip():
        if not pipeline.is_ready():
            st.warning("Please upload and index at least one PDF first.")
        elif not (api_key or os.getenv("GOOGLE_API_KEY")):
            st.warning("Enter your Google API key in the sidebar.")
        else:
            with st.spinner("Retrieving · Reranking · Generating …"):
                try:
                    result = pipeline.query(user_question.strip())
                    st.session_state.chat_history.insert(0, {"question": user_question.strip(), "result": result})
                except Exception as exc:
                    st.error(f"Query failed: {exc}")
                    st.exception(exc)

    # ── History ───────────────────────────────────────────────────────────────
    if not st.session_state.chat_history:
        st.markdown("""
        <div class="empty">
          <div class="empty-icon">◎</div>
          <div class="empty-title">Ready when you are</div>
          <div class="empty-desc">Upload your research papers in the sidebar,
            then ask a question to get cited, grounded answers.</div>
        </div>
        """, unsafe_allow_html=True)

    for i, entry in enumerate(st.session_state.chat_history):
        q      = entry["question"]
        result = entry["result"]
        ev     = result["evaluation"]

        def _cls(v):
            return "good" if v >= 0.65 else ("med" if v >= 0.35 else "bad")

        # Question
        st.markdown(f"""
        <div class="qa-block">
          <div class="q-card">
            <div class="q-icon">?</div>
            <div class="q-text">{q}</div>
          </div>
        """, unsafe_allow_html=True)

        # Source chips
        chips = "".join(
            f'<span class="src-chip"><span class="src-dot"></span>'
            f'{s["paper_name"][:18]}{"…" if len(s["paper_name"])>18 else ""}, p.{s["page_number"]}'
            f'</span>'
            for s in result["sources"]
        )

        # Answer
        body = result["answer"].replace("\n", "<br>")
        st.markdown(f"""
          <div class="a-card">
            <div class="a-hdr">
              <span class="a-badge">◎ Lexis</span>
              <span class="a-meta">{ev['num_sources_used']} sources · {ev['num_unique_papers']} paper(s)</span>
            </div>
            <div class="a-body">{body}</div>
            <div class="srcs-row">
              <span class="srcs-lbl">Sources</span>
              {chips}
            </div>
          </div>
        </div>
        """, unsafe_allow_html=True)

        # Citations expander
        with st.expander(f"📎  {len(result['sources'])} cited passages", expanded=(i == 0)):
            items = ""
            for j, s in enumerate(result["sources"], 1):
                sec_tag = f'<span class="cite-tag">{s["section"][:22]}</span>' if s["section"] else ""
                items += f"""
                <div class="cite-item">
                  <div class="cite-hdr">
                    <span class="cite-num">{j}</span>
                    <span class="cite-paper">{s["paper_name"]}</span>
                    <span class="cite-tag">p. {s["page_number"]}</span>
                    {sec_tag}
                    <span class="cite-score">↑ {s["score"]}</span>
                  </div>
                  <blockquote class="cite-excerpt">{s["excerpt"]}</blockquote>
                </div>"""
            st.markdown(f'<div class="cite-wrap">{items}</div>', unsafe_allow_html=True)

        # Evaluation expander
        with st.expander("📊  Quality metrics"):
            metrics = [
                ("Context Utilisation", ev["context_utilisation"]),
                ("Retrieval Relevance", ev["retrieval_relevance"]),
                ("Source Diversity",    ev["source_diversity"]),
                ("Overall Quality",     ev["overall_quality"]),
            ]
            cells = ""
            for lbl, val in metrics:
                c = _cls(val)
                cells += (
                    f'<div class="eval-cell">'
                    f'<div class="eval-lbl">{lbl}</div>'
                    f'<div class="eval-val ev-{c}">{val:.0%}</div>'
                    f'<div class="eval-bar"><div class="eval-fill ef-{c}" style="width:{int(val*100)}%"></div></div>'
                    f'</div>'
                )
            st.markdown(f'<div class="eval-grid">{cells}</div>', unsafe_allow_html=True)

            if ev["uncertainty_flags"]:
                flags = ", ".join(f'"{f}"' for f in ev["uncertainty_flags"][:3])
                st.markdown(
                    f'<div class="unc-warn">⚠ Uncertainty detected — phrases: {flags}</div>',
                    unsafe_allow_html=True,
                )

        if i < len(st.session_state.chat_history) - 1:
            st.markdown("<hr>", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
#  RIGHT COLUMN
# ═══════════════════════════════════════════════════════════════════════════════
with col_info:

    pipeline_steps = [
        ("📥", "PDF Ingestion",       "PyPDF page extraction"),
        ("🧹", "Text Cleaning",       "Noise & ligature removal"),
        ("✂️", "Smart Chunking",      "Section-aware + overlap"),
        ("🧠", "MiniLM Embedding",    "384-dim local vectors"),
        ("🗃️", "FAISS Index",         "Persisted vector store"),
        ("🔍", "Hybrid Retrieval",    "70% semantic + 30% BM25"),
        ("🔥", "Cross-Encoder Rerank","Top-10 → Top-5"),
        ("✨", "Gemini Flash 2.5",    "Grounded generation"),
        ("📊", "Auto Evaluation",     "3-metric quality score"),
    ]
    steps_html = "".join(
        f'<div class="pipe-step">'
        f'<div class="pipe-icon">{icon}</div>'
        f'<div><div class="pipe-name">{name}</div><div class="pipe-desc">{desc}</div></div>'
        f'</div>'
        for icon, name, desc in pipeline_steps
    )
    st.markdown(
        f'<div class="info-card"><div class="info-title">RAG Pipeline</div>{steps_html}</div>',
        unsafe_allow_html=True,
    )

    tips = [
        "Ask about methods, results, or contributions for precise answers.",
        'Say "In the BERT paper…" for paper-targeted questions.',
        '"Compare X and Y" synthesises across multiple papers.',
        "All answers include automatic page-level citations.",
        "Your index persists across sessions — no need to re-upload.",
    ]
    tips_html = "".join(
        f'<div class="tip-item"><span class="tip-arrow">→</span><span>{t}</span></div>'
        for t in tips
    )
    st.markdown(
        f'<div class="info-card"><div class="info-title">Tips</div>{tips_html}</div>',
        unsafe_allow_html=True,
    )
