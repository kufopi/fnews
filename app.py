import os
import tempfile
import streamlit as st
from pathlib import Path
from modules.text_detector  import predict as text_predict, predict_batch, confidence_tier as text_tier
from modules.image_verifier import predict as img_predict,  confidence_tier as img_tier
from modules.video_verifier import predict as vid_predict,  confidence_tier as vid_tier

st.set_page_config(
    page_title="Detector Media Intelligence",
    page_icon="◈",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ─────────────────────────────────────────────────────────────────────────────
# GLOBAL STYLES
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Outfit:wght@300;400;600;700;900&display=swap');

:root {
  --bg:       #080B0F;
  --surface:  #0D1117;
  --surface2: #131920;
  --border:   #1E2A35;
  --border2:  #253040;
  --text:     #C8D8E8;
  --muted:    #4A6070;
  --accent:   #00D4FF;
  --fake:     #FF3B5C;
  --real:     #00E5A0;
  --ai:       #FF8C00;
  --deep:     #B06EFF;
  --mono:     'Space Mono', monospace;
  --sans:     'Outfit', sans-serif;
}

html, body, [class*="css"] {
  font-family: var(--sans);
  background: var(--bg) !important;
  color: var(--text);
}

/* Scanline overlay */
.stApp::before {
  content: '';
  position: fixed;
  inset: 0;
  background: repeating-linear-gradient(
    0deg,
    transparent,
    transparent 2px,
    rgba(0,212,255,0.012) 2px,
    rgba(0,212,255,0.012) 4px
  );
  pointer-events: none;
  z-index: 9999;
}

.stApp { background: var(--bg) !important; }
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 2rem !important; max-width: 820px !important; }

/* ── HERO ── */
.hero { position: relative; padding: 2.5rem 0 2rem; margin-bottom: 1.5rem; }
.hero-eyebrow {
  font-family: var(--mono);
  font-size: .65rem;
  letter-spacing: .2em;
  color: var(--accent);
  text-transform: uppercase;
  margin-bottom: .6rem;
  opacity: .7;
}
.hero-title {
  font-family: var(--sans);
  font-size: clamp(2.4rem, 6vw, 3.4rem);
  font-weight: 900;
  line-height: 1;
  letter-spacing: -.03em;
  color: #E8F4FF;
  margin: 0 0 .5rem;
}
.hero-title span { color: var(--accent); }
.hero-sub { font-family: var(--mono); font-size: .72rem; color: var(--muted); letter-spacing: .08em; }
.hero-line {
  width: 100%;
  height: 1px;
  background: linear-gradient(90deg, var(--accent) 0%, transparent 60%);
  margin-top: 1.8rem;
  opacity: .4;
}
.hero::after {
  content: '◈';
  position: absolute;
  top: 2.5rem;
  right: 0;
  font-size: 2rem;
  color: var(--accent);
  opacity: .12;
}

/* ── MODULE BADGES ── */
.module-row { display: flex; gap: .5rem; margin-bottom: 1.8rem; flex-wrap: wrap; }
.mod-badge {
  font-family: var(--mono);
  font-size: .62rem;
  letter-spacing: .12em;
  padding: .3rem .75rem;
  border: 1px solid var(--border2);
  border-radius: 2px;
  color: var(--muted);
  background: var(--surface);
}
.mod-badge.active { border-color: var(--accent); color: var(--accent); background: rgba(0,212,255,.06); }

/* ── TABS ── */
.stTabs [data-baseweb="tab-list"] {
  background: var(--surface) !important;
  border: 1px solid var(--border) !important;
  border-radius: 4px !important;
  padding: .25rem !important;
  gap: .2rem !important;
}
.stTabs [data-baseweb="tab"] {
  font-family: var(--mono) !important;
  font-size: .7rem !important;
  letter-spacing: .08em !important;
  color: var(--muted) !important;
  border-radius: 3px !important;
  padding: .5rem 1.1rem !important;
  border: none !important;
  background: transparent !important;
  transition: all .2s !important;
}
.stTabs [aria-selected="true"] {
  background: rgba(0,212,255,.1) !important;
  color: var(--accent) !important;
  border: 1px solid rgba(0,212,255,.25) !important;
}
.stTabs [data-baseweb="tab-highlight"] { display: none !important; }
.stTabs [data-baseweb="tab-border"]    { display: none !important; }

/* ── INPUTS ── */
.stTextArea textarea {
  font-family: var(--mono) !important;
  font-size: .8rem !important;
  background: var(--surface) !important;
  border: 1px solid var(--border) !important;
  border-radius: 4px !important;
  color: var(--text) !important;
  line-height: 1.7 !important;
  transition: border-color .2s !important;
}
.stTextArea textarea:focus {
  border-color: var(--accent) !important;
  box-shadow: 0 0 0 2px rgba(0,212,255,.08) !important;
}
.stTextArea label {
  font-family: var(--mono) !important;
  font-size: .68rem !important;
  letter-spacing: .1em !important;
  color: var(--muted) !important;
  text-transform: uppercase !important;
}
[data-testid="stFileUploader"] {
  border: 1px dashed var(--border2) !important;
  border-radius: 4px !important;
  background: var(--surface) !important;
  transition: border-color .2s !important;
}
[data-testid="stFileUploader"]:hover { border-color: var(--accent) !important; }
[data-testid="stFileUploader"] label {
  font-family: var(--mono) !important;
  font-size: .68rem !important;
  letter-spacing: .1em !important;
  color: var(--muted) !important;
  text-transform: uppercase !important;
}

/* ── BUTTONS ── */
.stButton > button {
  font-family: var(--mono) !important;
  font-size: .72rem !important;
  letter-spacing: .12em !important;
  font-weight: 700 !important;
  text-transform: uppercase !important;
  background: transparent !important;
  color: var(--accent) !important;
  border: 1px solid var(--accent) !important;
  border-radius: 3px !important;
  padding: .55rem 1.6rem !important;
  transition: all .2s !important;
}
.stButton > button:hover {
  background: rgba(0,212,255,.1) !important;
  box-shadow: 0 0 16px rgba(0,212,255,.15) !important;
}

/* ── RESULT CARDS ── */
@keyframes card-in {
  from { opacity: 0; transform: translateY(10px); }
  to   { opacity: 1; transform: translateY(0); }
}
@keyframes scan-pulse {
  0%, 100% { opacity: 0; }
  50%       { opacity: .4; }
}

.vcard {
  position: relative;
  border-radius: 4px;
  padding: 1.6rem 1.8rem 1.4rem;
  margin-top: 1.2rem;
  animation: card-in .35s ease both;
  overflow: hidden;
}
.vcard::before {
  content: '';
  position: absolute;
  top: 0; left: 0; right: 0;
  height: 2px;
}
.vcard::after {
  content: '';
  position: absolute;
  left: 0; right: 0;
  height: 40px;
  pointer-events: none;
  animation: scan-pulse 2.4s ease-in-out infinite .5s;
  top: 40%;
}
.vcard-fake  { background: rgba(255,59,92,.06);  border: 1px solid rgba(255,59,92,.3); }
.vcard-real  { background: rgba(0,229,160,.05);  border: 1px solid rgba(0,229,160,.25); }
.vcard-ai    { background: rgba(255,140,0,.06);  border: 1px solid rgba(255,140,0,.28); }
.vcard-deep  { background: rgba(176,110,255,.06);border: 1px solid rgba(176,110,255,.28); }
.vcard-fake::before  { background: var(--fake); }
.vcard-real::before  { background: var(--real); }
.vcard-ai::before    { background: var(--ai);   }
.vcard-deep::before  { background: var(--deep); }
.vcard-fake::after   { background: linear-gradient(transparent, rgba(255,59,92,.05), transparent); }
.vcard-real::after   { background: linear-gradient(transparent, rgba(0,229,160,.05), transparent); }
.vcard-ai::after     { background: linear-gradient(transparent, rgba(255,140,0,.05), transparent); }
.vcard-deep::after   { background: linear-gradient(transparent, rgba(176,110,255,.05), transparent); }

.vcard-header { display: flex; align-items: flex-start; justify-content: space-between; margin-bottom: 1rem; }
.vcard-verdict {
  font-family: var(--sans);
  font-size: 1.9rem;
  font-weight: 900;
  letter-spacing: -.02em;
  line-height: 1;
}
.vcard-fake  .vcard-verdict { color: var(--fake); }
.vcard-real  .vcard-verdict { color: var(--real); }
.vcard-ai    .vcard-verdict { color: var(--ai);   }
.vcard-deep  .vcard-verdict { color: var(--deep); }

.vcard-conf-pill {
  font-family: var(--mono);
  font-size: .65rem;
  letter-spacing: .1em;
  padding: .25rem .7rem;
  border-radius: 2px;
  border: 1px solid currentColor;
  opacity: .75;
}
.vcard-fake  .vcard-conf-pill { color: var(--fake); }
.vcard-real  .vcard-conf-pill { color: var(--real); }
.vcard-ai    .vcard-conf-pill { color: var(--ai);   }
.vcard-deep  .vcard-conf-pill { color: var(--deep); }

.vcard-meta {
  font-family: var(--mono);
  font-size: .66rem;
  color: var(--muted);
  letter-spacing: .06em;
  margin-bottom: 1rem;
}

/* Progress bar */
.vbar-wrap { background: var(--border); border-radius: 1px; height: 3px; margin-bottom: 1.2rem; overflow: hidden; }
@keyframes bar-grow {
  from { width: 0%; }
  to   { width: var(--w); }
}
.vbar-inner {
  height: 3px;
  border-radius: 1px;
  animation: bar-grow .6s cubic-bezier(.4,0,.2,1) .2s both;
}
.vcard-fake  .vbar-inner { background: var(--fake); }
.vcard-real  .vbar-inner { background: var(--real); }
.vcard-ai    .vbar-inner { background: var(--ai);   }
.vcard-deep  .vbar-inner { background: var(--deep); }

/* Signal grid */
.sig-section-label {
  font-family: var(--mono);
  font-size: .6rem;
  letter-spacing: .15em;
  color: var(--muted);
  text-transform: uppercase;
  padding-top: .6rem;
  padding-bottom: .4rem;
  border-top: 1px solid var(--border);
  opacity: .6;
}
.sig-grid { display: grid; gap: .45rem; }
.sig-item { display: grid; grid-template-columns: 130px 1fr 44px; align-items: center; gap: .6rem; }
.sig-label { font-family: var(--mono); font-size: .6rem; color: var(--muted); letter-spacing: .05em; }
.sig-track { background: var(--border); border-radius: 1px; height: 3px; overflow: hidden; }
.sig-fill  { height: 3px; border-radius: 1px; }
.sig-num   { font-family: var(--mono); font-size: .62rem; color: var(--text); text-align: right; opacity: .55; }

/* ── SECTION LABEL ── */
.sec-label {
  font-family: var(--mono);
  font-size: .6rem;
  letter-spacing: .18em;
  color: var(--muted);
  text-transform: uppercase;
  margin: 1rem 0 .6rem;
  display: flex;
  align-items: center;
  gap: .6rem;
}
.sec-label::after { content: ''; flex: 1; height: 1px; background: var(--border); }

/* ── BATCH LIST ── */
.batch-item {
  display: grid;
  grid-template-columns: 28px 58px 1fr 38px;
  align-items: center;
  gap: .75rem;
  padding: .65rem 0;
  border-bottom: 1px solid var(--border);
  animation: card-in .2s ease both;
}
.batch-idx  { font-family: var(--mono); font-size: .6rem; color: var(--muted); text-align: right; }
.batch-badge {
  font-family: var(--mono);
  font-size: .58rem;
  font-weight: 700;
  letter-spacing: .1em;
  padding: .2rem .5rem;
  border-radius: 2px;
  text-align: center;
}
.b-fake { background: rgba(255,59,92,.15); color: var(--fake); border: 1px solid rgba(255,59,92,.3); }
.b-real { background: rgba(0,229,160,.12); color: var(--real); border: 1px solid rgba(0,229,160,.25); }
.batch-text { font-size: .82rem; color: var(--text); line-height: 1.4; opacity: .85; }
.batch-pct  { font-family: var(--mono); font-size: .65rem; color: var(--muted); text-align: right; }

/* ── METRICS ── */
[data-testid="stMetric"] {
  background: var(--surface) !important;
  border: 1px solid var(--border) !important;
  border-radius: 4px !important;
  padding: .9rem 1.1rem !important;
}
[data-testid="stMetricLabel"] {
  font-family: var(--mono) !important;
  font-size: .6rem !important;
  letter-spacing: .1em !important;
  color: var(--muted) !important;
  text-transform: uppercase !important;
}
[data-testid="stMetricValue"] {
  font-family: var(--sans) !important;
  font-size: 1.5rem !important;
  font-weight: 700 !important;
  color: var(--text) !important;
}

/* ── INFO / ALERT ── */
[data-testid="stAlert"] {
  background: var(--surface) !important;
  border: 1px solid var(--border2) !important;
  border-radius: 4px !important;
  font-family: var(--mono) !important;
  font-size: .7rem !important;
  color: var(--muted) !important;
}

/* ── EXPANDER ── */
[data-testid="stExpander"] {
  border: 1px solid var(--border) !important;
  border-radius: 4px !important;
  background: var(--surface) !important;
  margin-top: .8rem !important;
}
[data-testid="stExpander"] summary {
  font-family: var(--mono) !important;
  font-size: .67rem !important;
  letter-spacing: .1em !important;
  text-transform: uppercase !important;
  color: var(--muted) !important;
  padding: .75rem 1rem !important;
}
[data-testid="stExpander"] summary:hover { color: var(--accent) !important; }

/* ── TABLE ── */
table { width: 100%; border-collapse: collapse; }
th {
  font-family: var(--mono);
  font-size: .62rem;
  letter-spacing: .1em;
  text-transform: uppercase;
  color: var(--muted);
  border-bottom: 1px solid var(--border);
  padding: .5rem .75rem;
  text-align: left;
}
td {
  font-size: .8rem;
  color: var(--text);
  padding: .5rem .75rem;
  border-bottom: 1px solid var(--border);
  opacity: .8;
}
tr:last-child td { border-bottom: none; }

/* ── IMAGE / VIDEO ── */
[data-testid="stImage"] img {
  border: 1px solid var(--border) !important;
  border-radius: 4px !important;
}
video {
  border: 1px solid var(--border) !important;
  border-radius: 4px !important;
  width: 100% !important;
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# HERO HEADER
# ─────────────────────────────────────────────────────────────────────────────

st.markdown("""
<div class="hero">
  <div class="hero-eyebrow">Final Year Project</div>
  <h1 class="hero-title">Multimedia Fake News<span> Detection</h1>
  <p class="hero-sub">TEXT ANALYSIS · IMAGE FORENSICS · VIDEO DEEPFAKE DETECTION · ALL LOCAL</p>
  <div class="hero-line"></div>
</div>
<div class="module-row">
  <div class="mod-badge active">◈ MODULE 01 — RoBERTa NLP</div>
  <div class="mod-badge active">◈ MODULE 02 — 6-Signal FFT</div>
  <div class="mod-badge active">◈ MODULE 03 — Temporal CV</div>
  <div class="mod-badge">○ NO EXTERNAL APIS</div>
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _vcard(label: str, conf: float, tier_fn, kind: str,
           signals: dict = None) -> str:
    colour_map = {
        "fake": "var(--fake)", "real": "var(--real)",
        "ai":   "var(--ai)",   "deep": "var(--deep)",
    }
    colour = colour_map.get(kind, "var(--accent)")
    tier   = tier_fn(conf)
    pct    = conf * 100

    sig_html = ""
    if signals:
        sig_html = '<div class="sig-section-label">Signal Breakdown</div><div class="sig-grid">'
        for name, val in signals.items():
            w   = int(val * 100)
            lbl = name.replace("_", " ").upper()
            sig_html += (
                f'<div class="sig-item">'
                f'<span class="sig-label">{lbl}</span>'
                f'<div class="sig-track"><div class="sig-fill" '
                f'style="width:{w}%;background:{colour}"></div></div>'
                f'<span class="sig-num">{val:.0%}</span>'
                f'</div>'
            )
        sig_html += "</div>"

    return (
        f'<div class="vcard vcard-{kind}">'
        f'  <div class="vcard-header">'
        f'    <div class="vcard-verdict">{label}</div>'
        f'    <div class="vcard-conf-pill">{conf:.0%} CONFIDENCE</div>'
        f'  </div>'
        f'  <div class="vcard-meta">TIER: {tier.upper()} &nbsp;·&nbsp; SCORE: {conf:.4f}</div>'
        f'  <div class="vbar-wrap">'
        f'    <div class="vbar-inner" style="--w:{pct:.1f}%"></div>'
        f'  </div>'
        f'  {sig_html}'
        f'</div>'
    )


def _sec(label: str) -> str:
    return f'<p class="sec-label">{label}</p>'


def _mono_note(text: str) -> str:
    return (
        f'<p style="font-family:var(--mono);font-size:.62rem;'
        f'color:var(--muted);padding-top:.5rem;">{text}</p>'
    )


# ─────────────────────────────────────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────────────────────────────────────

tab_text, tab_img, tab_vid, tab_batch = st.tabs([
    "01 · TEXT", "02 · IMAGE", "03 · VIDEO", "04 · BATCH",
])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — TEXT
# ══════════════════════════════════════════════════════════════════════════════
with tab_text:
    st.markdown(_sec("Input — Paste Article or Headline"), unsafe_allow_html=True)
    st.info("First run downloads the RoBERTa model (~500 MB) from HuggingFace and caches it locally.")

    text_input = st.text_area(
        "Article text",
        height=190,
        placeholder="Paste a headline or full article body here…\n\ne.g. Scientists discover vaccine effective against all flu strains…",
        key="text_input",
        label_visibility="collapsed",
    )

    col_btn, col_note = st.columns([1, 3])
    with col_btn:
        run_text = st.button("▶  RUN ANALYSIS", key="run_text")
    with col_note:
        st.markdown(
            _mono_note("MODEL: hamzab/roberta-fake-news-classification · LIAR DATASET"),
            unsafe_allow_html=True,
        )

    if run_text:
        if not text_input.strip():
            st.warning("Enter some text before running analysis.")
        else:
            with st.spinner("Running RoBERTa inference…"):
                r = text_predict(text_input)
            label, conf = r["label"], r["confidence"]
            kind = "fake" if label == "FAKE" else "real"
            st.markdown(_sec("Verdict"), unsafe_allow_html=True)
            st.markdown(_vcard(label, conf, text_tier, kind), unsafe_allow_html=True)

            with st.expander("▸  INTERPRETATION & CAVEATS"):
                if label == "FAKE":
                    st.markdown(f"""
**{conf:.0%} confidence — FAKE**

Language patterns match known misinformation signatures in the LIAR training dataset:
sensationalist tone, unverified superlative claims, or vocabulary fingerprints from
flagged sources. Cross-reference with Reuters, AP, BBC, or primary sources.

> ⚠ This is a **style classifier**, not a fact-checker.  
> A well-written lie scores low. A dramatic but true headline may score high.
""")
                else:
                    st.markdown(f"""
**{conf:.0%} confidence — REAL**

Writing style aligns with credible journalism patterns — measured tone, attributed
claims, institutional vocabulary. Still verify independently: this model cannot
access the internet or check facts in a knowledge base.
""")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — IMAGE
# ══════════════════════════════════════════════════════════════════════════════
with tab_img:
    st.markdown(_sec("Input — Upload Image File"), unsafe_allow_html=True)
    st.markdown(
        _mono_note(
            "DETECTS: STABLE DIFFUSION · MIDJOURNEY · DALL-E · FIREFLY · ANY GAN/DIFFUSION OUTPUT<br>"
            "METHOD: FFT SPECTRAL · SRM NOISE · LBP TEXTURE · DCT BLOCKING · CHANNEL CORR · EXIF"
        ),
        unsafe_allow_html=True,
    )

    uploaded_img = st.file_uploader(
        "Drop image here or click to browse  ·  JPG PNG WEBP BMP",
        type=["jpg", "jpeg", "png", "webp", "bmp"],
        key="img_upload",
        label_visibility="collapsed",
    )

    if uploaded_img:
        col_img, col_ctrl = st.columns([3, 2])
        with col_img:
            st.markdown(_sec("Preview"), unsafe_allow_html=True)
            st.image(uploaded_img, use_container_width=True)
        with col_ctrl:
            st.markdown(_sec("File Info"), unsafe_allow_html=True)
            sz_kb = len(uploaded_img.getvalue()) / 1024
            st.markdown(f"""
<div style="font-family:var(--mono);font-size:.65rem;line-height:2.2;color:var(--muted);
            background:var(--surface);border:1px solid var(--border);
            border-radius:4px;padding:.8rem 1rem;margin-bottom:1rem;">
  <span style="color:var(--text);">NAME</span>&nbsp;&nbsp;{uploaded_img.name}<br>
  <span style="color:var(--text);">SIZE</span>&nbsp;&nbsp;{sz_kb:.1f} KB<br>
  <span style="color:var(--text);">TYPE</span>&nbsp;&nbsp;{uploaded_img.type or "—"}
</div>""", unsafe_allow_html=True)
            run_img = st.button("▶  ANALYZE IMAGE", key="run_img")

        if run_img:
            img_bytes = uploaded_img.getvalue()
            with st.spinner("Running 6-signal forensic analysis…"):
                r = img_predict(img_bytes)

            label, conf = r["label"], r["confidence"]
            if label == "AI_GENERATED":
                kind, display = "ai",   "AI GENERATED"
            elif label == "REAL":
                kind, display = "real", "REAL PHOTO"
            else:
                kind, display = "fake", "UNKNOWN"

            st.markdown(_sec("Forensic Verdict"), unsafe_allow_html=True)
            st.markdown(
                _vcard(display, conf, img_tier, kind, signals=r.get("signals")),
                unsafe_allow_html=True,
            )

            if r.get("warning"):
                st.warning(r["warning"])

            with st.expander("▸  SIGNAL REFERENCE TABLE"):
                st.markdown("""
| Signal | Mechanism | AI Indicator |
|---|---|---|
| **Frequency** | Radial power spectrum slope (1/f²) | Flat / stepped high-freq band |
| **Noise** | SRM high-pass residual statistics | Near-zero or uniform residual |
| **Texture** | LBP histogram entropy (256 bins) | Unnaturally high entropy |
| **DCT Blocking** | 8-px boundary gradient ratio | Missing JPEG block structure |
| **Colour Corr.** | Inter-channel noise cross-correlation | Independently synthesised channels |
| **EXIF** | Metadata presence + AI tool signatures | Absent tags or AI software string |
""")
                if r.get("exif_info"):
                    st.markdown(_sec("EXIF Data"), unsafe_allow_html=True)
                    st.json(r["exif_info"])
                else:
                    st.markdown(
                        _mono_note("— No EXIF metadata found. AI images typically carry none. —"),
                        unsafe_allow_html=True,
                    )


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — VIDEO
# ══════════════════════════════════════════════════════════════════════════════
with tab_vid:
    st.markdown(_sec("Input — Upload Video File"), unsafe_allow_html=True)
    st.markdown(
        _mono_note(
            "DETECTS: FACE SWAP · GAN DEEPFAKE · DIFFUSION FACE REPLACEMENT<br>"
            "METHOD: TEXTURE SEAM · TEMPORAL FLICKER · LANDMARK GEOMETRY · OPTICAL FLOW · COLOUR · HF NOISE"
        ),
        unsafe_allow_html=True,
    )
    st.info(
        "First run attempts to download the MediaPipe FaceLandmarker model (~30 MB). "
        "Falls back to OpenCV Haar cascade if network is restricted."
    )

    uploaded_vid = st.file_uploader(
        "Drop video here or click to browse  ·  MP4 MOV AVI MKV WEBM",
        type=["mp4", "mov", "avi", "mkv", "webm"],
        key="vid_upload",
        label_visibility="collapsed",
    )

    if uploaded_vid:
        st.markdown(_sec("Preview"), unsafe_allow_html=True)
        st.video(uploaded_vid)

        col_vbtn, col_vnote = st.columns([1, 3])
        with col_vbtn:
            run_vid = st.button("▶  ANALYZE VIDEO", key="run_vid")
        with col_vnote:
            st.markdown(
                _mono_note("SAMPLES 32 FRAMES EVENLY · 10% TRIMMED MEAN ENSEMBLE"),
                unsafe_allow_html=True,
            )

        if run_vid:
            suffix = Path(uploaded_vid.name).suffix or ".mp4"
            with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
                tmp.write(uploaded_vid.getvalue())
                tmp_path = tmp.name

            with st.spinner("Extracting frames · Running temporal forensic analysis…"):
                r = vid_predict(tmp_path)
            os.unlink(tmp_path)

            st.markdown(_sec("Forensic Verdict"), unsafe_allow_html=True)

            if r["label"] == "UNKNOWN":
                st.error(f"Analysis incomplete: {r['warning']}")
            else:
                label, conf = r["label"], r["confidence"]
                kind = "deep" if label == "DEEPFAKE" else "real"
                st.markdown(
                    _vcard(label, conf, vid_tier, kind, signals=r.get("signals")),
                    unsafe_allow_html=True,
                )
                if r.get("warning"):
                    st.warning(r["warning"])

            st.markdown(_sec("Video Metadata"), unsafe_allow_html=True)
            mc1, mc2, mc3, mc4 = st.columns(4)
            mc1.metric("Total Frames",   r.get("frame_count", "—"))
            mc2.metric("Analysed",        r.get("analysed_frames", "—"))
            mc3.metric("With Face",       r.get("faces_detected", "—"))
            mc4.metric("Duration",        f"{r.get('duration_sec', 0):.1f}s")

            with st.expander("▸  SIGNAL REFERENCE TABLE"):
                st.markdown("""
| Signal | Mechanism | Deepfake Indicator |
|---|---|---|
| **Texture Seam** | Laplacian variance at face-oval boundary | Unnatural edge discontinuity |
| **Temporal Flicker** | Frame-to-frame brightness Δ in face crop | Generator per-frame noise |
| **Landmark Geometry** | 478-pt symmetry · EAR · velocity jitter | Over-regularised / erratic |
| **Optical Flow** | Face vs background Farneback flow ratio | Decoupled or frozen face region |
| **Colour Stats** | HSV saturation inter-frame variance | Flicker · over-saturation |
| **HF Noise** | SRM residual std variance across frames | Generator instability |
""")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — BATCH TEXT
# ══════════════════════════════════════════════════════════════════════════════
with tab_batch:
    st.markdown(_sec("Input — One Headline or Article per Line"), unsafe_allow_html=True)
    st.info("First run downloads the RoBERTa model (~500 MB) from HuggingFace and caches it locally.")

    batch_input = st.text_area(
        "Batch input",
        height=200,
        placeholder=(
            "Paste headlines, one per line:\n\n"
            "Covid vaccine found to be 100% effective against all variants\n"
            "Fed raises rates by 25bps amid inflation concerns\n"
            "ALIENS LAND IN TEXAS — GOVERNMENT COVER-UP EXPOSED"
        ),
        key="batch_input",
        label_visibility="collapsed",
    )

    run_batch = st.button("▶  ANALYZE BATCH", key="run_batch")

    if run_batch:
        lines = [l.strip() for l in batch_input.strip().split("\n") if l.strip()]
        if not lines:
            st.warning("Enter at least one line.")
        else:
            with st.spinner(f"Classifying {len(lines)} item{'s' if len(lines)>1 else ''}…"):
                results = predict_batch(lines)

            fake_n = sum(1 for r in results if r["label"] == "FAKE")
            real_n = len(results) - fake_n

            st.markdown(_sec("Summary"), unsafe_allow_html=True)
            sc1, sc2, sc3 = st.columns(3)
            sc1.metric("Total Analysed",  len(results))
            sc2.metric("Flagged Fake",    fake_n)
            sc3.metric("Classified Real", real_n)

            st.markdown(_sec("Results"), unsafe_allow_html=True)
            for i, (t, r) in enumerate(zip(lines, results)):
                badge_cls = "b-fake" if r["label"] == "FAKE" else "b-real"
                truncated = t[:95] + "…" if len(t) > 95 else t
                st.markdown(f"""
<div class="batch-item" style="animation-delay:{i * 0.04}s">
  <span class="batch-idx">{i+1:02d}</span>
  <span class="batch-badge {badge_cls}">{r['label']}</span>
  <span class="batch-text">{truncated}</span>
  <span class="batch-pct">{r['confidence']:.0%}</span>
</div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# CREDITS FOOTER  — fully inline styles (no class dependencies)
# ══════════════════════════════════════════════════════════════════════════════

# Shared inline style tokens
_C_BG      = "#0D1117"
_C_BORDER  = "#1E2A35"
_C_BORDER2 = "#253040"
_C_TEXT    = "#C8D8E8"
_C_MUTED   = "#4A6070"
_C_ACCENT  = "#00D4FF"
_MONO      = "Space Mono, monospace"
_SANS      = "Outfit, sans-serif"

LIBS = [
    ("Streamlit",     "Web UI framework"),
    ("PyTorch",       "Deep learning runtime"),
    ("Transformers",  "RoBERTa NLP model"),
    ("OpenCV (cv2)",  "Image & video processing"),
    ("NumPy",         "Numerical computation"),
    ("SciPy",         "Signal & statistical analysis"),
    ("Pillow (PIL)",  "Image I/O & conversion"),
    ("MediaPipe",     "Face landmark detection"),
    ("piexif",        "EXIF metadata parsing"),
]

def _chip(name: str, role: str) -> str:
    return (
        f'<div style="padding:.5rem .75rem;background:{_C_BG};border:1px solid {_C_BORDER2};'
        f'border-radius:3px;display:flex;flex-direction:column;gap:.2rem;">'
        f'<span style="font-family:{_MONO};font-size:.68rem;font-weight:700;color:{_C_TEXT};">{name}</span>'
        f'<span style="font-family:{_MONO};font-size:.58rem;color:{_C_MUTED};">{role}</span>'
        f'</div>'
    )

def _person(role_label: str, name: str, sub: str, mat: str) -> str:
    return (
        f'<div style="padding:1rem 1.25rem;background:{_C_BG};border:1px solid {_C_BORDER2};'
        f'border-radius:4px;flex:1;">'
        f'<div style="font-family:{_MONO};font-size:.58rem;letter-spacing:.18em;'
        f'text-transform:uppercase;color:{_C_ACCENT};opacity:.7;margin-bottom:.4rem;">{role_label}</div>'
        f'<div style="font-family:{_SANS};font-size:1rem;font-weight:700;color:{_C_TEXT};">{name}</div>'
        f'<div style="font-family:{_SANS};font-size:1rem;font-weight:700;color:{_C_TEXT};">{mat}</div>'
        f'<div style="font-family:{_MONO};font-size:.65rem;color:{_C_MUTED};margin-top:.2rem;">{sub}</div>'
        f'</div>'
    )
def _personn(role_label: str, name: str, sub: str) -> str:
    return (
        f'<div style="padding:1rem 1.25rem;background:{_C_BG};border:1px solid {_C_BORDER2};'
        f'border-radius:4px;flex:1;">'
        f'<div style="font-family:{_MONO};font-size:.58rem;letter-spacing:.18em;'
        f'text-transform:uppercase;color:{_C_ACCENT};opacity:.7;margin-bottom:.4rem;">{role_label}</div>'
        f'<div style="font-family:{_SANS};font-size:1rem;font-weight:700;color:{_C_TEXT};">{name}</div>'
        
        f'<div style="font-family:{_MONO};font-size:.65rem;color:{_C_MUTED};margin-top:.2rem;">{sub}</div>'
        f'</div>'
    )

# Build chips HTML (3 per row using flexbox rows)
chip_rows = ""
for i in range(0, len(LIBS), 3):
    row_chips = "".join(_chip(n, r) for n, r in LIBS[i:i+3])
    chip_rows += (
        f'<div style="display:flex;gap:.5rem;margin-bottom:.5rem;">{row_chips}</div>'
    )

credits_html = f"""
<div style="margin-top:4rem;border-top:1px solid {_C_BORDER};padding-top:2rem;padding-bottom:3rem;">

  <!-- Section header -->
  <div style="font-family:{_MONO};font-size:.6rem;letter-spacing:.18em;text-transform:uppercase;
              color:{_C_MUTED};margin-bottom:1.25rem;display:flex;align-items:center;gap:.6rem;">
    Project Credits
    <div style="flex:1;height:1px;background:{_C_BORDER};"></div>
  </div>

  <!-- People row -->
  <div style="display:flex;gap:1rem;margin-bottom:1.5rem;">
    {_person("Developer", "Idowu Christianah Toluwaleyi", "Department of Computer Science","Matric No: 22/CS/09")}
    {_personn("Supervisor", "Dr. Kokobioko", "Lecturer I &nbsp;·&nbsp; Department of Computer Science")}
  </div>

  <!-- Divider -->
  <div style="height:1px;background:{_C_BORDER};margin:1.25rem 0;"></div>

  <!-- Libraries label -->
  <div style="font-family:{_MONO};font-size:.58rem;letter-spacing:.18em;text-transform:uppercase;
              color:{_C_ACCENT};opacity:.7;margin-bottom:.85rem;">Technologies &amp; Libraries</div>

  <!-- Chip grid -->
  {chip_rows}

  <!-- Divider -->
  <div style="height:1px;background:{_C_BORDER};margin:1.5rem 0 1rem;"></div>

  <!-- Footer line -->
  <div style="font-family:{_MONO};font-size:.6rem;color:{_C_MUTED};letter-spacing:.1em;
              text-align:center;opacity:.45;">
    Idowu Christianah Toluwaleyi &nbsp;·&nbsp; Multimedia Fake News Detection &nbsp;·&nbsp; Computer Science Department
  </div>

</div>
"""

st.markdown(credits_html, unsafe_allow_html=True)