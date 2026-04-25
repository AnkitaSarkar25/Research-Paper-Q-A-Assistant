"""
app.py — Lexis · Research Paper Q&A
Design: Clean light research UI — white/grey canvas, teal accents,
        editorial serif headings, monospaced data elements, industry-standard UX.
"""

import os
import re
import sys
import hashlib
import logging
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as _st_components

sys.path.insert(0, str(Path(__file__).parent))

from src.utils.helpers import setup_logging, save_uploaded_file
from config import RAW_DIR, VECTORSTORE_DIR, GOOGLE_API_KEY, SARVAM_API_KEY

setup_logging(logging.INFO)

if GOOGLE_API_KEY:
    os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY

st.set_page_config(
    page_title="Research Paper Q&A",
    page_icon="◎",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ═══════════════════════════════════════════════════════════════════════════════
#  HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def _make_entry_id(question: str, answer: str) -> str:
    """Stable content-derived key — immune to list-position shifts."""
    payload = f"{question.strip()}|||{answer[:120]}"
    return hashlib.sha1(payload.encode()).hexdigest()[:16]


def _normalise_score(raw: float) -> tuple:
    """
    Cross-encoder scores are unbounded logits (often in the range −15 … +5).
    Sigmoid-map them to a human-readable 0–100 relevance percentage.
    Returns (float_pct, display_str).
    """
    import math
    pct = 1.0 / (1.0 + math.exp(-raw))   # sigmoid → (0, 1)
    pct = round(pct * 100, 1)
    return pct, f"{pct:.0f}%"


def _strip_html_to_text(raw: str) -> str:
    """
    Use Python's stdlib html.parser to extract only visible text nodes,
    completely ignoring all tags, attributes, scripts, and styles.
    This is the only truly reliable way to handle malformed / truncated HTML.
    """
    from html.parser import HTMLParser

    class _TextExtractor(HTMLParser):
        # Tags whose *content* we discard entirely (UI chrome, not prose)
        _SKIP = {"script", "style", "head"}

        def __init__(self):
            super().__init__(convert_charrefs=True)
            self._parts: list[str] = []
            self._skip_depth: int = 0
            # If the excerpt contains a <blockquote>, we ONLY want its text.
            # Two-pass: first check, then extract.
            self._in_blockquote: int = 0
            self._bq_parts: list[str] = []
            self._has_blockquote: bool = False

        def handle_starttag(self, tag, attrs):
            if tag in self._SKIP:
                self._skip_depth += 1
            if tag == "blockquote":
                self._in_blockquote += 1
                self._has_blockquote = True

        def handle_endtag(self, tag):
            if tag in self._SKIP and self._skip_depth:
                self._skip_depth -= 1
            if tag == "blockquote" and self._in_blockquote:
                self._in_blockquote -= 1

        def handle_data(self, data):
            if self._skip_depth:
                return
            self._parts.append(data)
            if self._in_blockquote:
                self._bq_parts.append(data)

        def get_text(self) -> str:
            # Prefer blockquote content if present (that's the actual passage)
            source = self._bq_parts if self._has_blockquote else self._parts
            return " ".join(source)

    parser = _TextExtractor()
    try:
        parser.feed(raw)
        return parser.get_text()
    except Exception:
        # Absolute last resort — regex strip
        return re.sub(r"<[^>]*>", " ", raw)


def _clean_excerpt(text: str) -> str:
    """
    Bulletproof excerpt cleaner. Handles every known corruption pattern:

    1. Plain text              → pass-through with artefact fixes
    2. Stray inline HTML tags  → stripped via html.parser
    3. Full rendered HTML block (cite-item + blockquote, possibly truncated)
                               → html.parser extracts only blockquote prose
    4. Truncated mid-tag HTML  → html.parser tolerates unclosed tags natively

    Post-extraction: entity decode, ligature fix, PDF spacing artefacts,
    UI-chrome token scrub, whitespace collapse, word-boundary truncation.
    """
    if not text or not text.strip():
        return ""

    # ── Step 1: HTML extraction (only when markup is present) ────────────────
    if re.search(r"<[a-zA-Z/]", text):
        text = _strip_html_to_text(text)

    # ── Step 2: scrub residual UI-chrome text (score badges, CSS tokens) ─────
    # These appear as visible text inside tag content when the excerpt field
    # captured a full rendered block, e.g. "↑ 0%" or class-name literals.
    text = re.sub(r"[↑↓]\s*\d+\s*%", "", text)
    # CSS class-name tokens that appear as text (cite-score, score-low, etc.)
    text = re.sub(
        r"\b(?:cite|score|eval|src|sb|pipe|tip|hero|query|unc|ef|ev|qa|a|q)"
        r"-(?:wrap|item|hdr|num|paper|tag|score|excerpt|body|card|badge|meta"
        r"|high|mid|low|good|med|bad|fill|bar|cell|lbl|val|icon|text|"
        r"grid|warn|step|name|desc|arrow|empty)\b",
        "", text
    )
    # Stray numeric or named attribute remnants (e.g.  style="width:80%")
    text = re.sub(r'\w[\w-]*\s*=\s*["\'][^"\']*["\']', "", text)

    # ── Step 3: ligature normalisation ───────────────────────────────────────
    for lig, rep in {
        "\ufb00": "ff", "\ufb01": "fi", "\ufb02": "fl",
        "\ufb03": "ffi", "\ufb04": "ffl", "\ufb06": "st",
        "\u00e6": "ae",  "\u0153": "oe",
    }.items():
        text = text.replace(lig, rep)

    # ── Step 4: PDF merge artefacts ──────────────────────────────────────────
    text = re.sub(r"([a-z])([A-Z])", r"\1 \2", text)
    text = re.sub(r"([.,;:!?])([A-Za-z])", r"\1 \2", text)

    # ── Step 5: collapse whitespace ──────────────────────────────────────────
    text = re.sub(r"\s+", " ", text).strip()

    # ── Step 6: truncate at word boundary ────────────────────────────────────
    if len(text) > 320:
        text = text[:320].rsplit(" ", 1)[0] + " …"
    
    text = re.sub(r"\bhow to cite this article\b.*", "", text, flags=re.I)
    text = re.sub(r"\bdownloaded from\b.*", "", text, flags=re.I)
    text = re.sub(r"\bterms and conditions\b.*", "", text, flags=re.I)

    return text



# ─────────────────────────────────────────────────────────────────────────────
#  CITATION RENDERER
#  Uses st.components.v1.html() instead of st.markdown(unsafe_allow_html=True)
#  because st.markdown inside st.expander silently escapes HTML in many
#  Streamlit versions, causing raw tags to appear as visible text.
#  components.v1.html() renders into a sandboxed <iframe> and ALWAYS works.
# ─────────────────────────────────────────────────────────────────────────────

_CITE_CSS = """
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600&family=JetBrains+Mono:wght@400;500&display=swap');
*{box-sizing:border-box;margin:0;padding:0;}
body{background:#fff;font-family:'DM Sans',sans-serif;color:#111827;padding:2px 0;}
.cite-wrap{background:#fff;border:1px solid rgba(0,0,0,0.10);border-radius:10px;overflow:hidden;}
.cite-item{padding:14px 18px;border-bottom:1px solid rgba(0,0,0,0.07);}
.cite-item:last-child{border-bottom:none;}
.cite-hdr{display:flex;align-items:center;gap:8px;margin-bottom:10px;flex-wrap:wrap;}
.cite-num{display:inline-flex;align-items:center;justify-content:center;width:20px;height:20px;
  background:rgba(180,83,9,0.07);border:1px solid rgba(180,83,9,0.18);border-radius:50%;
  font-family:'JetBrains Mono',monospace;font-size:0.61rem;color:#B45309;font-weight:700;flex-shrink:0;}
.cite-paper{font-weight:600;font-size:0.85rem;color:#111827;}
.cite-tag{font-family:'JetBrains Mono',monospace;font-size:0.64rem;color:#6B7280;
  background:#F1F3F8;border:1px solid rgba(0,0,0,0.08);border-radius:4px;padding:2px 7px;}
.cite-score{margin-left:auto;font-family:'JetBrains Mono',monospace;font-size:0.63rem;border-radius:100px;padding:2px 8px;}
.score-high{color:#059669;background:rgba(5,150,105,0.08);border:1px solid rgba(5,150,105,0.2);}
.score-mid {color:#D97706;background:rgba(217,119,6,0.08);border:1px solid rgba(217,119,6,0.2);}
.score-low {color:#6B7280;background:#F1F3F8;border:1px solid rgba(0,0,0,0.08);}
.cite-excerpt{font-size:0.83rem;color:#374151;line-height:1.72;padding:10px 14px;
  background:#F8F9FC;border-left:3px solid rgba(180,83,9,0.25);border-radius:0 6px 6px 0;
  font-style:italic;word-break:break-word;}
.cite-empty{font-size:0.78rem;color:#9CA3AF;font-style:italic;padding:4px 0;}
"""

def _render_citations(sources: list) -> None:
    """
    Render citation cards using st.components.v1.html().

    This bypasses the st.markdown / st.expander HTML-escaping bug entirely.
    The component renders into an iframe, so it always shows proper HTML
    regardless of Streamlit version or nesting depth.

    Args:
        sources: List of source dicts from pipeline.query() result.
    """
    import html as _html
    import math

    def _sigmoid_pct(raw: float) -> tuple:
        pct = round(100 / (1 + math.exp(-raw)), 1)
        cls = "score-high" if pct >= 65 else "score-mid" if pct >= 40 else "score-low"
        return pct, f"{pct:.0f}%", cls

    items_html = ""
    rendered = 0

    for s in sources:
        # Clean the excerpt — strip ALL html, ligatures, PDF noise
        excerpt = _clean_excerpt(str(s.get("excerpt", "") or ""))
        prose   = re.sub(r"\s+", "", excerpt)
        if len(prose) < 25:
            continue
        rendered += 1

        try:
            raw_score = float(s.get("score", 0.0))
        except (TypeError, ValueError):
            raw_score = 0.0
        pct, pct_str, score_cls = _sigmoid_pct(raw_score)

        # html.escape every dynamic value — excerpt is plain text after
        # _clean_excerpt but may still contain &, <, > from paper content
        name_e    = _html.escape(str(s.get("paper_name",  "Unknown")))
        page_e    = _html.escape(str(s.get("page_number", "?")))
        excerpt_e = _html.escape(excerpt)

        sec_raw = str(s.get("section", "") or "").strip()
        sec_tag = (
            f'<span class="cite-tag">{_html.escape(sec_raw[:26])}</span>'
            if sec_raw else ""
        )

        items_html += f"""
        <div class="cite-item">
          <div class="cite-hdr">
            <span class="cite-num">{rendered}</span>
            <span class="cite-paper">{name_e}</span>
            <span class="cite-tag">p.&nbsp;{page_e}</span>
            {sec_tag}
            <span class="cite-score {score_cls}">&#8593; {pct_str}</span>
          </div>
          <div class="cite-excerpt">{excerpt_e}</div>
        </div>"""

    if not items_html:
        items_html = (
            '<div class="cite-item">'
            '<span class="cite-empty">No readable excerpts available for these passages.</span>'
            '</div>'
        )

    full_html = f"""<!DOCTYPE html>
<html><head>
<meta charset="utf-8">
<style>{_CITE_CSS}</style>
</head><body>
<div class="cite-wrap">{items_html}</div>
</body></html>"""

    # Height: ~110px per item + padding; minimum 120px
    height = max(120, rendered * 130 + 30)
    _st_components.html(full_html, height=height, scrolling=True)

# ═══════════════════════════════════════════════════════════════════════════════
#  DESIGN SYSTEM — Clean Light Research UI
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Sans:ital,opsz,wght@0,9..40,300;0,9..40,400;0,9..40,500;0,9..40,600;1,9..40,400&family=JetBrains+Mono:wght@400;500;600&display=swap');

/* ─── TOKENS ─────────────────────────────────────────────────────────── */
:root {
  --bg:           #F8F9FC;
  --bg2:          #FFFFFF;
  --bg3:          #F1F3F8;
  --bg4:          #E8ECF4;
  --surface:      #FFFFFF;
  --surface2:     #F4F6FB;
  --text:         #111827;
  --text-soft:    #374151;
  --text-muted:   #6B7280;
  --text-dim:     #9CA3AF;
  --teal:         #0D9488;
  --teal-mid:     #0F766E;
  --teal-dim:     #134E4A;
  --teal-glow:    rgba(13,148,136,0.10);
  --teal-glow2:   rgba(13,148,136,0.05);
  --gold:         #B45309;
  --gold-dim:     rgba(180,83,9,0.07);
  --gold-border:  rgba(180,83,9,0.18);
  --good:         #059669;
  --med:          #D97706;
  --bad:          #DC2626;
  --border:       rgba(0,0,0,0.07);
  --border-mid:   rgba(0,0,0,0.11);
  --border-teal:  rgba(13,148,136,0.18);
  --shadow:       0 1px 6px rgba(0,0,0,0.06), 0 4px 20px rgba(0,0,0,0.04);
  --shadow-sm:    0 1px 3px rgba(0,0,0,0.06);
  --shadow-teal:  0 0 0 1px rgba(13,148,136,0.10);
  --r-sm: 8px; --r-md: 12px; --r-lg: 18px; --r-xl: 24px;
}

/* ─── BASE ───────────────────────────────────────────────────────────── */
html, body,
[data-testid="stAppViewContainer"],
[data-testid="stAppViewContainer"] > .main {
  background: var(--bg) !important;
}
.stApp { font-family: 'DM Sans', sans-serif; color: var(--text); background: var(--bg) !important; }
* { box-sizing: border-box; }
.block-container { padding: 0 2.5rem 4rem !important; max-width: 1440px !important; }

/* ─── SIDEBAR ────────────────────────────────────────────────────────── */
[data-testid="stSidebar"],
[data-testid="stSidebar"] > div:first-child {
  background: var(--bg2) !important;
  border-right: 1px solid var(--border-mid) !important;
}
[data-testid="stSidebar"] * { color: var(--text-soft) !important; }
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 { color: var(--text) !important; }
[data-testid="stSidebar"] input[type="text"],
[data-testid="stSidebar"] input[type="password"] {
  background: var(--bg3) !important; border: 1px solid var(--border-mid) !important;
  border-radius: var(--r-sm) !important; color: var(--text) !important;
  font-family: 'JetBrains Mono', monospace !important; font-size: 0.76rem !important;
}
[data-testid="stSidebar"] input:focus {
  border-color: var(--teal) !important; box-shadow: 0 0 0 3px var(--teal-glow) !important; outline: none !important;
}
[data-testid="stSidebar"] .stButton > button {
  background: var(--teal) !important;
  color: #fff !important; border: none !important; border-radius: var(--r-sm) !important;
  font-family: 'DM Sans', sans-serif !important; font-weight: 600 !important;
  font-size: 0.82rem !important; padding: 9px 16px !important; width: 100% !important;
  transition: all 0.2s !important; box-shadow: 0 1px 4px rgba(13,148,136,0.2) !important; cursor: pointer !important;
}
[data-testid="stSidebar"] .stButton > button:hover {
  background: var(--teal-mid) !important; transform: translateY(-1px) !important; box-shadow: 0 4px 12px rgba(13,148,136,0.22) !important;
}
[data-testid="stSidebar"] hr { border-color: var(--border) !important; margin: 12px 0 !important; }
[data-testid="stSidebar"] [data-testid="stMetric"] {
  background: var(--bg3) !important; border: 1px solid var(--border) !important;
  border-radius: var(--r-sm) !important; padding: 10px 12px !important;
}
[data-testid="stSidebar"] [data-testid="stMetricLabel"] {
  color: var(--text-muted) !important; font-size: 0.67rem !important;
  letter-spacing: 0.08em !important; text-transform: uppercase !important;
}
[data-testid="stSidebar"] [data-testid="stMetricValue"] {
  color: var(--teal) !important; font-size: 1.3rem !important; font-weight: 700 !important;
}
[data-testid="stSidebar"] .streamlit-expanderHeader {
  background: var(--bg3) !important; border: 1px solid var(--border) !important;
  border-radius: var(--r-sm) !important; color: var(--text-soft) !important; font-size: 0.79rem !important;
}
[data-testid="stSidebar"] [data-testid="stFileUploader"] {
  background: var(--bg3) !important; border: 1.5px dashed rgba(13,148,136,0.25) !important;
  border-radius: var(--r-md) !important; padding: 14px !important; transition: border-color 0.2s !important;
}
[data-testid="stSidebar"] [data-testid="stFileUploader"]:hover { border-color: rgba(13,148,136,0.45) !important; }
[data-testid="collapsedControl"], button[kind="headerNoPadding"] {
  display: flex !important; visibility: visible !important; opacity: 1 !important;
}

/* ─── SIDEBAR COMPONENTS ─────────────────────────────────────────────── */
.sb-brand { display:flex;align-items:center;gap:12px;padding:4px 0 20px;border-bottom:1px solid var(--border);margin-bottom:18px; }
.sb-brand-icon { width:34px;height:34px;background:var(--teal);border-radius:9px;display:flex;align-items:center;justify-content:center;font-size:16px;color:#fff;flex-shrink:0;box-shadow:0 2px 8px rgba(13,148,136,0.22); }
.sb-brand-name { font-family:'DM Serif Display',serif;font-size:1.25rem;color:var(--text) !important;letter-spacing:-0.02em;line-height:1; }
.sb-brand-sub  { font-size:0.59rem;color:var(--teal) !important;letter-spacing:0.16em;text-transform:uppercase;font-weight:600;margin-top:2px; }
.sb-section    { font-size:0.59rem;letter-spacing:0.16em;text-transform:uppercase;color:var(--text-dim) !important;font-weight:700;margin:18px 0 8px; }
.sb-status     { display:flex;align-items:center;gap:8px;padding:8px 12px;border-radius:var(--r-sm);font-size:0.78rem;font-weight:500;margin:8px 0; }
.sb-status.ready   { background:rgba(13,148,136,0.07);border:1px solid rgba(13,148,136,0.18);color:var(--teal) !important; }
.sb-status.waiting { background:rgba(217,119,6,0.06);border:1px solid rgba(217,119,6,0.18);color:var(--med) !important; }
.sb-dot { width:6px;height:6px;border-radius:50%;flex-shrink:0; }
.sb-dot.ready   { background:var(--teal);box-shadow:0 0 6px var(--teal);animation:pulse 2s infinite; }
.sb-dot.waiting { background:var(--med); }
.sb-hint        { font-size:0.69rem;color:var(--text-muted) !important;margin-top:-4px;margin-bottom:6px;line-height:1.5; }
.sb-api-ok      { color:var(--good) !important; }
.sb-api-warn    { color:var(--med) !important; }
.sb-paper-row   { display:flex;justify-content:space-between;font-size:0.73rem;padding:5px 0;border-bottom:1px solid var(--border); }
.sb-paper-name  { color:var(--text-soft) !important; }
.sb-paper-cnt   { color:var(--teal) !important;font-family:'JetBrains Mono',monospace; }

/* ─── HERO ───────────────────────────────────────────────────────────── */
.hero { position:relative;padding:3rem 0 2.2rem;margin-bottom:2rem; }
.hero::before { content:'';position:absolute;top:-60px;left:-80px;width:500px;height:400px;background:radial-gradient(ellipse,rgba(13,148,136,0.05) 0%,transparent 65%);pointer-events:none; }
.hero::after  { content:'';position:absolute;bottom:0;left:0;right:0;height:1px;background:linear-gradient(90deg,transparent,var(--border-mid) 30%,var(--border-mid) 70%,transparent); }
.hero-eyebrow { font-family:'JetBrains Mono',monospace;font-size:0.65rem;letter-spacing:0.22em;color:var(--teal);text-transform:uppercase;margin-bottom:12px;display:flex;align-items:center;gap:10px; }
.hero-eyebrow::before { content:'';display:inline-block;width:28px;height:1px;background:var(--teal); }
.hero-h1 { font-family:'DM Serif Display',serif;font-size:clamp(2.2rem,4vw,3.4rem);font-weight:400;color:var(--text);letter-spacing:-0.03em;line-height:1.08;margin:0 0 14px; }
.hero-h1 em { font-style:italic;color:var(--teal);-webkit-text-fill-color:var(--teal); }
.hero-sub { font-size:1rem;color:var(--text-soft);line-height:1.72;max-width:520px; }
.tech-pills { display:flex;flex-wrap:wrap;gap:6px;margin-top:18px; }
.tpill { font-family:'JetBrains Mono',monospace;font-size:0.63rem;color:var(--text-muted);background:var(--bg2);border:1px solid var(--border-mid);border-radius:100px;padding:3px 10px;letter-spacing:0.02em;transition:border-color 0.2s,color 0.2s; }
.tpill:hover { border-color:var(--teal);color:var(--teal); }

/* ─── QUERY BOX ──────────────────────────────────────────────────────── */
.query-wrap { background:var(--surface);border:1px solid var(--border-mid);border-radius:var(--r-xl);padding:1.8rem;box-shadow:var(--shadow);margin-bottom:2rem;position:relative;overflow:hidden; }
.query-wrap::before { content:'';position:absolute;top:0;left:0;right:0;height:3px;background:var(--teal);border-radius:var(--r-xl) var(--r-xl) 0 0; }
.query-lbl { font-size:0.61rem;font-weight:700;letter-spacing:0.16em;text-transform:uppercase;color:var(--text-muted);margin-bottom:10px; }
.stTextArea textarea {
  background: var(--bg3) !important; border: 1.5px solid var(--border-mid) !important;
  border-radius: var(--r-md) !important; color: var(--text) !important;
  font-family: 'DM Sans', sans-serif !important; font-size: 0.96rem !important;
  line-height: 1.7 !important; padding: 14px 16px !important;
  transition: all 0.2s !important; resize: none !important;
}
.stTextArea textarea:focus {
  border-color: var(--teal) !important; box-shadow: 0 0 0 3px var(--teal-glow) !important;
  background: var(--bg2) !important; outline: none !important;
}
.stTextArea textarea::placeholder { color: var(--text-dim) !important; font-style: italic !important; }

.main-btns .stButton > button {
  background: var(--teal) !important;
  color: #fff !important; border: none !important; border-radius: var(--r-md) !important;
  font-family: 'DM Sans', sans-serif !important; font-weight: 600 !important;
  font-size: 0.88rem !important; padding: 11px 20px !important;
  transition: all 0.2s !important; box-shadow: 0 1px 4px rgba(13,148,136,0.18) !important;
  width: 100% !important; cursor: pointer !important; letter-spacing: 0.01em !important;
}
.main-btns .stButton > button:hover { background: var(--teal-mid) !important; transform: translateY(-1px) !important; box-shadow: 0 4px 14px rgba(13,148,136,0.24) !important; }
.main-btns .stButton:last-child > button {
  background: var(--bg3) !important; color: var(--text-muted) !important;
  border: 1px solid var(--border-mid) !important; box-shadow: none !important; font-weight: 500 !important;
}
.main-btns .stButton:last-child > button:hover { background: var(--bg4) !important; color: var(--text-soft) !important; transform: none !important; }

/* ─── Q&A BLOCKS ─────────────────────────────────────────────────────── */
.qa-block { margin-bottom:2rem;animation:slideUp 0.38s cubic-bezier(0.16,1,0.3,1) both; }
@keyframes slideUp { from{opacity:0;transform:translateY(16px)} to{opacity:1;transform:translateY(0)} }

.q-card { display:flex;align-items:flex-start;gap:12px;background:var(--bg3);border:1px solid var(--border-mid);border-radius:var(--r-md);padding:1rem 1.4rem;margin-bottom:3px;box-shadow:var(--shadow-sm); }
.q-icon { width:26px;height:26px;flex-shrink:0;background:var(--bg4);border:1px solid var(--border-mid);border-radius:50%;display:flex;align-items:center;justify-content:center;font-family:'DM Serif Display',serif;font-size:12px;color:var(--text-muted);margin-top:1px; }
.q-text { font-size:0.96rem;font-weight:500;color:var(--text);line-height:1.55; }

.a-card { background:var(--surface);border:1px solid var(--border-teal);border-radius:var(--r-md);padding:1.6rem 1.8rem;box-shadow:var(--shadow);position:relative;overflow:hidden; }
.a-card::before { content:'';position:absolute;top:0;left:0;width:3px;height:100%;background:var(--teal);border-radius:3px 0 0 3px; }
.a-hdr { display:flex;align-items:center;gap:10px;margin-bottom:16px;padding-bottom:12px;border-bottom:1px solid var(--border); }
.a-badge { font-family:'JetBrains Mono',monospace;font-size:0.61rem;letter-spacing:0.12em;text-transform:uppercase;color:var(--teal);background:rgba(13,148,136,0.08);border:1px solid rgba(13,148,136,0.2);border-radius:100px;padding:3px 10px; }
.a-meta  { font-size:0.68rem;color:var(--text-muted);margin-left:auto;font-family:'JetBrains Mono',monospace; }
.a-body  { font-size:0.94rem;line-height:1.84;color:var(--text-soft); }

.srcs-row { display:flex;flex-wrap:wrap;gap:6px;margin-top:16px;padding-top:14px;border-top:1px solid var(--border);align-items:center; }
.srcs-lbl { font-size:0.61rem;font-weight:700;letter-spacing:0.1em;text-transform:uppercase;color:var(--text-dim);margin-right:4px; }
.src-chip { display:inline-flex;align-items:center;gap:5px;background:var(--gold-dim);border:1px solid var(--gold-border);border-radius:100px;padding:3px 10px 3px 7px;font-family:'JetBrains Mono',monospace;font-size:0.67rem;color:var(--gold);transition:background 0.15s;cursor:default; }
.src-chip:hover { background:rgba(180,83,9,0.12); }
.src-dot { width:4px;height:4px;border-radius:50%;background:var(--gold);flex-shrink:0; }

/* ─── CITATIONS ──────────────────────────────────────────────────────── */
.cite-wrap { background:var(--bg2);border:1px solid var(--border-mid);border-radius:var(--r-md);overflow:hidden; }
.cite-item { padding:1.1rem 1.4rem;border-bottom:1px solid var(--border);transition:background 0.15s; }
.cite-item:last-child { border-bottom:none; }
.cite-item:hover { background:var(--bg3); }
.cite-hdr { display:flex;align-items:center;gap:8px;margin-bottom:10px;flex-wrap:wrap; }
.cite-num { display:inline-flex;align-items:center;justify-content:center;width:20px;height:20px;background:var(--gold-dim);border:1px solid var(--gold-border);border-radius:50%;font-family:'JetBrains Mono',monospace;font-size:0.61rem;color:var(--gold);font-weight:700;flex-shrink:0; }
.cite-paper { font-family:'DM Sans',sans-serif;font-weight:600;font-size:0.84rem;color:var(--text); }
.cite-tag { font-family:'JetBrains Mono',monospace;font-size:0.64rem;color:var(--text-muted);background:var(--bg3);border:1px solid var(--border);border-radius:4px;padding:2px 7px; }

/* Score badge — normalised, coloured by relevance tier */
.cite-score { margin-left:auto;font-family:'JetBrains Mono',monospace;font-size:0.63rem;border-radius:100px;padding:2px 8px; }
.cite-score.score-high { color:var(--good);background:rgba(5,150,105,0.08);border:1px solid rgba(5,150,105,0.2); }
.cite-score.score-mid  { color:var(--med);background:rgba(217,119,6,0.08);border:1px solid rgba(217,119,6,0.2); }
.cite-score.score-low  { color:var(--text-muted);background:var(--bg3);border:1px solid var(--border); }

.cite-excerpt { font-size:0.82rem;color:var(--text-soft);line-height:1.72;padding:10px 14px;background:var(--bg3);border-left:3px solid var(--gold-border);border-radius:0 6px 6px 0;font-style:italic;margin:0;word-break:break-word; }
.cite-excerpt-empty { display:block;font-size:0.78rem;color:var(--text-dim);font-style:italic;padding:6px 0 2px;letter-spacing:0.01em; }

/* ─── EVALUATION ─────────────────────────────────────────────────────── */
.eval-grid { display:grid;grid-template-columns:repeat(4,1fr);gap:10px;padding:4px 0; }
.eval-cell { background:var(--bg3);border:1px solid var(--border);border-radius:var(--r-sm);padding:14px 12px;text-align:center; }
.eval-lbl  { font-size:0.62rem;font-weight:700;letter-spacing:0.06em;text-transform:uppercase;color:var(--text-muted);margin-bottom:8px; }
.eval-val  { font-family:'DM Serif Display',serif;font-size:1.5rem;font-weight:400;line-height:1; }
.ev-good { color:var(--good); } .ev-med { color:var(--med); } .ev-bad { color:var(--bad); }
.eval-bar  { height:3px;border-radius:2px;margin-top:8px;background:var(--bg4);overflow:hidden; }
.eval-fill { height:100%;border-radius:2px;transition:width 0.6s cubic-bezier(0.16,1,0.3,1); }
.ef-good { background:var(--good); } .ef-med { background:var(--med); } .ef-bad { background:var(--bad); }
.unc-warn { display:flex;align-items:center;gap:8px;background:rgba(217,119,6,0.05);border:1px solid rgba(217,119,6,0.16);border-radius:var(--r-sm);padding:10px 14px;font-size:0.78rem;color:var(--med);margin-top:10px; }

/* ─── INFO PANEL ─────────────────────────────────────────────────────── */
.info-card { background:var(--surface);border:1px solid var(--border-mid);border-radius:var(--r-lg);padding:1.3rem;margin-bottom:14px;box-shadow:var(--shadow-sm); }
.info-title { font-size:0.59rem;font-weight:700;letter-spacing:0.16em;text-transform:uppercase;color:var(--text-muted);margin-bottom:14px;display:flex;align-items:center;gap:8px; }
.info-title::after { content:'';flex:1;height:1px;background:var(--border); }
.pipe-step { display:flex;align-items:flex-start;gap:10px;padding:7px 0;border-bottom:1px solid var(--border); }
.pipe-step:last-child { border-bottom:none; }
.pipe-icon { width:26px;height:26px;flex-shrink:0;background:var(--bg3);border:1px solid var(--border-mid);border-radius:6px;display:flex;align-items:center;justify-content:center;font-size:11px;margin-top:1px; }
.pipe-name { font-weight:500;font-size:0.79rem;color:var(--text);font-family:'DM Sans',sans-serif; }
.pipe-desc { font-size:0.68rem;color:var(--text-muted);margin-top:1px; }
.tip-item  { display:flex;gap:8px;font-size:0.77rem;color:var(--text-soft);padding:4px 0;line-height:1.5; }
.tip-arrow { color:var(--teal);font-weight:700;flex-shrink:0;margin-top:1px; }

/* ─── EXPANDERS ──────────────────────────────────────────────────────── */
.streamlit-expanderHeader {
  background: var(--bg3) !important; border: 1px solid var(--border-mid) !important;
  border-radius: var(--r-sm) !important; color: var(--text-soft) !important;
  font-family: 'DM Sans', sans-serif !important; font-size: 0.81rem !important;
  font-weight: 500 !important; padding: 10px 14px !important; transition: background 0.15s !important;
}
.streamlit-expanderHeader:hover { background: var(--bg4) !important; }
.streamlit-expanderContent { border: 1px solid var(--border-mid) !important; border-top: none !important; border-radius: 0 0 var(--r-sm) var(--r-sm) !important; background: var(--bg2) !important; }

/* ─── EMPTY STATE ────────────────────────────────────────────────────── */
.empty { text-align:center;padding:4rem 2rem;color:var(--text-muted); }
.empty-icon  { font-size:2.6rem;margin-bottom:14px;opacity:0.3; }
.empty-title { font-family:'DM Serif Display',serif;font-size:1.3rem;font-weight:400;color:var(--text-soft);margin-bottom:8px; }
.empty-desc  { font-size:0.87rem;font-style:italic;color:var(--text-muted);line-height:1.7;max-width:340px;margin:0 auto; }

/* ─── MISC ───────────────────────────────────────────────────────────── */
audio { border-radius: var(--r-sm) !important; width: 100% !important; }
hr { border-color: var(--border) !important; margin: 1.6rem 0 !important; }
.stSpinner > div { border-top-color: var(--teal) !important; }
#MainMenu, footer { visibility: hidden; }
[data-testid="stDeployButton"] { display: none; }
@keyframes pulse {
  0%,100% { box-shadow: 0 0 4px var(--teal); }
  50%      { box-shadow: 0 0 10px var(--teal), 0 0 18px rgba(13,148,136,0.2); }
}
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
#  STATE  (idempotent init)
# ═══════════════════════════════════════════════════════════════════════════════
_DEFAULTS = {
    "chat_history":           [],
    "tts_speaker":            "anushka",
    "tts_language":           "en-IN",
    "tts_pace":               1.0,
    "tts_audio_cache":        {},   # {entry_id: bytes} — stable key, never shifts
    "stt_pending_transcript": "",
    "stt_last_audio_hash":    None,
    "stt_auto_submit":        False,
    "stt_recorder_key":       0,
}
for _k, _v in _DEFAULTS.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v


# ═══════════════════════════════════════════════════════════════════════════════
#  PIPELINE
# ═══════════════════════════════════════════════════════════════════════════════
@st.cache_resource(show_spinner="Initialising research …")
def get_pipeline():
    from src.pipeline import try_load_existing_index
    return try_load_existing_index()

pipeline = get_pipeline()


# ═══════════════════════════════════════════════════════════════════════════════
#  VOICE
# ═══════════════════════════════════════════════════════════════════════════════
SPEAKERS = [
    "anushka","abhilash","manisha","vidya","arya","karun",
    "hitesh","aditya","ritu","priya","neha","rahul",
    "pooja","rohan","simran","kavya",
]
LANGUAGES = {
    "English (India)": "en-IN","Hindi":"hi-IN","Bengali":"bn-IN",
    "Tamil":"ta-IN","Telugu":"te-IN","Kannada":"kn-IN",
    "Malayalam":"ml-IN","Marathi":"mr-IN","Gujarati":"gu-IN","Punjabi":"pa-IN",
}

def _sarvam_key() -> str:
    return SARVAM_API_KEY.strip() if SARVAM_API_KEY else ""

def _voice_enabled() -> bool:
    return bool(_sarvam_key())

def _do_tts(text: str, entry_id: str) -> None:
    if not _voice_enabled():
        st.warning("SARVAM_API_KEY is not set in your .env file.")
        return
    clean = re.sub(r"[*`#]", "", text)
    if len(clean) > 2000:
        clean = clean[:2000] + "…"
    with st.spinner("🔊 Synthesising speech …"):
        try:
            from src.voice.sarvam_voice import synthesize_speech
            audio_bytes = synthesize_speech(
                text=clean, api_key=_sarvam_key(),
                speaker=st.session_state.tts_speaker,
                language_code=st.session_state.tts_language,
                pace=st.session_state.tts_pace,
            )
            if audio_bytes:
                st.session_state.tts_audio_cache[entry_id] = audio_bytes
                st.rerun()
            else:
                st.error("TTS returned empty audio.")
        except Exception as exc:
            st.error(f"TTS error: {exc}")
            logging.exception("TTS synthesis failed")


# ═══════════════════════════════════════════════════════════════════════════════
#  SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div class="sb-brand">
      <div class="sb-brand-icon">◎</div>
      <div>
        <div class="sb-brand-name">Research Paper Q&amp;A</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    if pipeline.is_ready():
        stats = pipeline.get_stats()
        st.markdown(
            f'<div class="sb-status ready"><div class="sb-dot ready"></div>'
            f'{stats.get("unique_papers",0)} paper(s) · {stats.get("total_chunks",0)} chunks</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div class="sb-status waiting"><div class="sb-dot waiting"></div>No papers indexed</div>',
            unsafe_allow_html=True,
        )

    st.markdown('<div class="sb-section">API Status</div>', unsafe_allow_html=True)
    for ok, sym, msg in [
        (bool(GOOGLE_API_KEY), "✓", "Google API key loaded"),
        (bool(GOOGLE_API_KEY) is False, "⚠", "GOOGLE_API_KEY missing in .env"),
        (bool(SARVAM_API_KEY), "✓", "Sarvam API key loaded"),
        (bool(SARVAM_API_KEY) is False, "⚠", "SARVAM_API_KEY missing in .env"),
    ]:
        if ok:
            cls = "sb-api-ok" if sym == "✓" else "sb-api-warn"
            st.markdown(f'<div class="sb-hint {cls}">{sym} {msg}</div>', unsafe_allow_html=True)

    with st.expander("🔊 TTS Settings"):
        st.session_state.tts_speaker = st.selectbox(
            "Speaker voice", SPEAKERS,
            index=SPEAKERS.index(st.session_state.tts_speaker),
        )
        lang_choice = st.selectbox("TTS Language", list(LANGUAGES.keys()), index=0)
        st.session_state.tts_language = LANGUAGES[lang_choice]
        st.session_state.tts_pace = st.slider(
            "Speaking pace", min_value=0.5, max_value=2.0,
            value=float(st.session_state.tts_pace), step=0.1,
        )

    st.divider()
    st.markdown('<div class="sb-section">Upload Papers</div>', unsafe_allow_html=True)
    uploaded_files = st.file_uploader(
        "files", label_visibility="collapsed", type=["pdf"], accept_multiple_files=True,
    )
    if uploaded_files:
        st.markdown(f'<div class="sb-hint">{len(uploaded_files)} file(s) selected</div>', unsafe_allow_html=True)

    if st.button("⚡  Process & Index PDFs", use_container_width=True):
        if not uploaded_files:
            st.warning("Select PDFs first.")
        elif not os.getenv("GOOGLE_API_KEY"):
            st.warning("GOOGLE_API_KEY is not set in your .env file.")
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
                    logging.exception("PDF ingestion failed")

    st.divider()
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
                    f'<span class="sb-paper-cnt">{n}</span></div>'
                    for p, n in stats["per_paper"].items()
                )
                st.markdown(rows, unsafe_allow_html=True)
    else:
        st.markdown('<div class="sb-hint">Upload PDFs above to begin.</div>', unsafe_allow_html=True)

    st.divider()
    if st.button("⊘  Clear Knowledge Base", use_container_width=True):
        from src.utils.helpers import clear_directory
        pipeline.reset()
        clear_directory(VECTORSTORE_DIR)
        st.session_state.chat_history    = []
        st.session_state.tts_audio_cache = {}
        st.rerun()

def _is_valid_excerpt(text: str) -> bool:
    if not text:
        return False
    text = _clean_excerpt(text)
    # Reject if too short or looks like noise
    return len(text.strip()) > 30
# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════════════
col_main, col_info = st.columns([11, 4], gap="large")

with col_main:

    # Hero
    st.markdown("""
    <div class="hero">
      <div class="hero-eyebrow">RAG · Retrieval-Augmented Generation</div>
      <h1 class="hero-h1">Ask anything about<br>your <em>research papers</em></h1>
      <p class="hero-sub">Upload research papers. Ask questions. 
                Get precise, citation-backed answers grounded only in your documents.</p>
      <div class="tech-pills">
        <span class="tpill">Gemini Flash 2.5</span>
        <span class="tpill">FAISS · BM25 Hybrid</span>
        <span class="tpill">Cross-Encoder Reranking</span>
        <span class="tpill">MiniLM Embeddings</span>
        <span class="tpill">Auto Evaluation</span>
        <span class="tpill">🎙 Sarvam STT</span>
        <span class="tpill">🔊 Sarvam TTS</span>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # Query
    st.markdown('<div class="query-wrap"><div class="query-lbl">Your question</div>', unsafe_allow_html=True)
    user_question = st.text_area(
        "q", label_visibility="collapsed", height=108,
        placeholder=(
            "e.g. What attention mechanism does the Transformer use?\n"
            "     How do the authors evaluate their model?\n"
            "     Compare the findings across uploaded papers."
        ),
        value=st.session_state.stt_pending_transcript,
    )

    stt_col, hint_col = st.columns([1, 4])
    with stt_col:
        audio_input = st.audio_input(
            "🎙 Record",
            key=f"stt_audio_widget_{st.session_state.stt_recorder_key}",
            help="Record your question — Sarvam AI will transcribe it",
        )
    with hint_col:
        if _voice_enabled():
            st.markdown(
                '<div style="padding-top:28px;font-size:0.73rem;color:var(--text-muted);">'
                '🎙 Click the mic to ask by voice. Sarvam AI will auto-transcribe above.</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<div style="padding-top:28px;font-size:0.73rem;color:var(--text-muted);">'
                '🔒 Set <b>SARVAM_API_KEY</b> in <code>.env</code> to enable voice input.</div>',
                unsafe_allow_html=True,
            )

    if audio_input is not None and _voice_enabled():
        audio_bytes = audio_input.read()
        if audio_bytes and len(audio_bytes) > 500:
            audio_hash = hashlib.md5(audio_bytes).hexdigest()
            if audio_hash != st.session_state.stt_last_audio_hash:
                st.session_state.stt_last_audio_hash = audio_hash
                with st.spinner("🎙 Transcribing …"):
                    try:
                        from src.voice.sarvam_voice import transcribe_audio
                        transcript = transcribe_audio(
                            audio_bytes=audio_bytes, api_key=_sarvam_key(), language_code="unknown",
                        )
                        if transcript:
                            st.session_state.stt_pending_transcript = transcript
                            st.session_state.stt_auto_submit = True
                            st.rerun()
                        else:
                            st.warning("Could not transcribe audio. Please try again.")
                    except Exception as exc:
                        st.error(f"STT error: {exc}")
                        logging.exception("STT transcription failed")

    st.markdown('<div class="main-btns">', unsafe_allow_html=True)
    bc1, bc2 = st.columns([5, 1])
    with bc1:
        ask_btn = st.button("Search & Answer →", use_container_width=True)
    with bc2:
        clear_btn = st.button("Clear", use_container_width=True)
    st.markdown('</div></div>', unsafe_allow_html=True)

    if clear_btn:
        st.session_state.chat_history             = []
        st.session_state.tts_audio_cache          = {}
        st.session_state.stt_pending_transcript   = ""
        st.session_state.stt_last_audio_hash      = None
        st.session_state.stt_auto_submit          = False
        st.session_state.stt_recorder_key        += 1
        st.rerun()

    should_query = (ask_btn and user_question.strip()) or (
        st.session_state.stt_auto_submit and st.session_state.stt_pending_transcript.strip()
    )
    query_text = user_question.strip() or st.session_state.stt_pending_transcript.strip()

    if should_query:
        st.session_state.stt_pending_transcript = ""
        st.session_state.stt_auto_submit        = False
        if not pipeline.is_ready():
            st.warning("Please upload and index at least one PDF first.")
        elif not os.getenv("GOOGLE_API_KEY"):
            st.warning("GOOGLE_API_KEY is not set in your .env file.")
        else:
            with st.spinner("Retrieving · Reranking · Generating …"):
                try:
                    result   = pipeline.query(query_text)
                    entry_id = _make_entry_id(query_text, result["answer"])
                    st.session_state.chat_history.insert(
                        0, {"question": query_text, "result": result, "entry_id": entry_id}
                    )
                    st.session_state.stt_last_audio_hash = None
                    st.session_state.stt_recorder_key   += 1
                    st.rerun()
                except Exception as exc:
                    st.error(f"Query failed: {exc}")
                    logging.exception("Pipeline query failed")

    # Chat history
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
        q        = entry["question"]
        result   = entry["result"]
        ev       = result["evaluation"]
        entry_id = entry.get("entry_id") or _make_entry_id(q, result["answer"])

        def _cls(v: float) -> str:
            return "good" if v >= 0.65 else ("med" if v >= 0.35 else "bad")

        # Question card
        st.markdown(f"""
        <div class="qa-block">
          <div class="q-card">
            <div class="q-icon">?</div>
            <div class="q-text">{q}</div>
          </div>
        """, unsafe_allow_html=True)

        chips = "".join(
            f'<span class="src-chip"><span class="src-dot"></span>'
            f'{s["paper_name"][:20]}{"…" if len(s["paper_name"])>20 else ""}'
            f'&nbsp;p.{s["page_number"]}</span>'
            for s in result["sources"]
        )
        body = result["answer"].replace("\n", "<br>")
        st.markdown(f"""
          <div class="a-card">
            <div class="a-hdr">
              <span class="a-badge">◎ Research Paper Q&A</span>
              <span class="a-meta">{ev['num_sources_used']} sources · {ev['num_unique_papers']} paper(s)</span>
            </div>
            <div class="a-body">{body}</div>
            <div class="srcs-row"><span class="srcs-lbl">Sources</span>{chips}</div>
          </div>
        </div>
        """, unsafe_allow_html=True)

        # TTS — keyed by stable entry_id
        has_audio = entry_id in st.session_state.tts_audio_cache
        tts_btn_col, tts_audio_col = st.columns([1, 5])
        with tts_btn_col:
            btn_label = "🔄 Replay" if has_audio else "🔊 Listen"
            if _voice_enabled():
                if st.button(btn_label, key=f"tts_{entry_id}",
                             help="Hear this answer spoken aloud via Sarvam AI"):
                    _do_tts(result["answer"], entry_id)
            else:
                st.button("🔊 Listen", key=f"tts_{entry_id}", disabled=True,
                          help="Set SARVAM_API_KEY in .env to enable TTS")
        with tts_audio_col:
            if has_audio:
                st.audio(st.session_state.tts_audio_cache[entry_id], format="audio/mp3")

        # Citations — rendered via st.components.v1.html() to guarantee
        # proper HTML rendering inside st.expander (st.markdown inside expanders
        # silently escapes HTML in many Streamlit versions).
        with st.expander(f"📎  {len(result['sources'])} cited passages", expanded=(i == 0)):
            _render_citations(result["sources"])

        # Quality metrics
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
            if ev.get("uncertainty_flags"):
                flags = ", ".join(f'"{f}"' for f in ev["uncertainty_flags"][:3])
                st.markdown(
                    f'<div class="unc-warn">⚠ Uncertainty detected — {flags}</div>',
                    unsafe_allow_html=True,
                )

        if i < len(st.session_state.chat_history) - 1:
            st.markdown("<hr>", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
#  RIGHT COLUMN
# ═══════════════════════════════════════════════════════════════════════════════
with col_info:
    pipeline_steps = [
        ("📥","PDF Ingestion",        "PyPDF page extraction"),
        ("🧹","Text Cleaning",        "Noise & ligature removal"),
        ("✂️","Smart Chunking",       "Section-aware + overlap"),
        ("🧠","MiniLM Embedding",     "384-dim local vectors"),
        ("🗃️","FAISS Index",          "Persisted vector store"),
        ("🔍","Hybrid Retrieval",     "70% semantic · 30% BM25"),
        ("🔥","Cross-Encoder Rerank", "Top-10 → Top-5"),
        ("✨","Gemini Flash 2.5",     "Grounded generation"),
        ("📊","Auto Evaluation",      "3-metric quality score"),
        ("🎙","Sarvam STT",           "Voice → text"),
        ("🔊","Sarvam TTS",           "Answer → spoken audio"),
    ]
    steps_html = "".join(
        f'<div class="pipe-step"><div class="pipe-icon">{ic}</div>'
        f'<div><div class="pipe-name">{nm}</div><div class="pipe-desc">{dc}</div></div></div>'
        for ic, nm, dc in pipeline_steps
    )
    st.markdown(
        f'<div class="info-card"><div class="info-title">RAG Pipeline</div>{steps_html}</div>',
        unsafe_allow_html=True,
    )

    tips = [
        "Ask about methods, results, or contributions.",
        '"In the BERT paper…" targets a specific paper.',
        '"Compare X and Y" synthesises across papers.',
        "Answers include automatic page-level citations.",
        "Index persists — no need to re-upload papers.",
        "🎙 Mic icon below the query box for voice input.",
        "🔊 Listen under any answer for spoken playback.",
    ]
    tips_html = "".join(
        f'<div class="tip-item"><span class="tip-arrow">→</span><span>{t}</span></div>'
        for t in tips
    )
    st.markdown(
        f'<div class="info-card"><div class="info-title">Tips</div>{tips_html}</div>',
        unsafe_allow_html=True,
    )