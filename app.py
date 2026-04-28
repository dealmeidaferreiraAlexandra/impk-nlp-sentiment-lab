# Developed by Alexandra de Almeida Ferreira

import json
from pathlib import Path

import pandas as pd
import streamlit as st
import torch
from PIL import Image
import io
import time

# =============================
# OPTIONAL PDF (SAFE IMPORT)
# =============================
try:
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib import colors
    REPORTLAB_AVAILABLE = True
except:
    REPORTLAB_AVAILABLE = False

from src.config import TRADITIONAL_MODEL_PATH, METRICS_PATH
from src.traditional import load_traditional_pipeline, predict_traditional_text
from src.bert_utils import load_bert_components, predict_bert_text
from src.utils import clean_text, read_uploaded_text

st.set_page_config(page_title="NLP Sentiment Lab", layout="wide")

# =============================
# STYLE (UNCHANGED)
# =============================
st.markdown("""<style>
.stApp { background:#020617; color:#e2e8f0; }
.left-panel { border-right:1px solid #1f2231; padding-right:12px; }
.right-panel { background:#050a18; padding:20px; border-radius:16px; }
.stTextArea textarea, .stTextInput input {
    background:#020617 !important;
    border:1px solid #1f2231 !important;
    color:#e2e8f0 !important;
}
[data-testid="stFileUploader"] [data-testid="stFileUploaderDropzone"] {
    background:#020617 !important;
    border:1px solid #1f2231 !important;
    border-radius:12px !important;
}
[data-testid="stFileUploader"] [data-testid="stFileUploaderDropzone"] * {
    background:transparent !important;
}
[data-testid="stFileUploader"] [data-testid="stFileUploaderDropzone"] button {
    background:#020617 !important;
    border:1px solid #1f2231 !important;
    color:#e2e8f0 !important;
}
.stButton>button {
    width:100%;
    background:linear-gradient(90deg,#6366f1,#8b5cf6);
    border-radius:10px;
    border:none;
}
.pipe { border:1px solid #1f2231; border-radius:12px; padding:12px; text-align:center; background:#020617; }
.active { border:1px solid #6366f1; box-shadow:0 0 20px rgba(99,102,241,0.6); }
.card { border:1px solid #1f2231; border-radius:14px; padding:16px; margin-top:20px; background:#020617; }
.footer { text-align:center; opacity:0.6; margin-top:40px; }
.small-muted { opacity:0.7; font-size: 13px; }
</style>""", unsafe_allow_html=True)

# =============================
# HEADER
# =============================
st.title("🧠 NLP Sentiment Lab")
st.caption("TF-IDF + Logistic Regression vs BERT | Text Classification + Sentiment Comparison")

# =============================
# MODEL LOADERS
# =============================
@st.cache_resource
def get_traditional_pipeline():
    return load_traditional_pipeline(TRADITIONAL_MODEL_PATH)

@st.cache_resource
def get_bert_components():
    return load_bert_components()

# =============================
# PIPELINE
# =============================
def render_pipeline(stage: str):
    p1, p2, p3, p4 = st.columns(4)

    def pipe(col, icon, title, desc, key):
        active = stage == key
        with col:
            st.markdown(f"""
            <div class="pipe {'active' if active else ''}">
                {icon}<br>
                <b>{title}</b><br>
                <span style="opacity:0.6;font-size:11px;">{desc}</span>
            </div>
            """, unsafe_allow_html=True)

    pipe(p1, "✍️", "INPUT", "Paste review or text", "input")
    pipe(p2, "🧹", "CLEAN", "Normalize text", "clean")
    pipe(p3, "🧠", "PREDICT", "Run both models", "predict")
    pipe(p4, "📊", "COMPARE", "Compare results", "compare")

def render_prediction_card(title: str, result: dict):
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown(f"### {title}")
    st.write(f"**Sentiment:** {result['label']}")
    st.progress(float(result["confidence"]))
    st.write(f"Confidence: {result['confidence']*100:.2f}%")
    st.write(f"Positive probability: {result['positive_prob']*100:.2f}%")
    st.write(f"Negative probability: {result['negative_prob']*100:.2f}%")
    st.markdown("</div>", unsafe_allow_html=True)

# =============================
# STATE
# =============================
if "stage" not in st.session_state:
    st.session_state.stage = "input"

if "results" not in st.session_state:
    st.session_state.results = None

if "input_key" not in st.session_state:
    st.session_state.input_key = "text_input_0"

if "upload_key" not in st.session_state:
    st.session_state.upload_key = "upload_0"

# =============================
# LAYOUT
# =============================
left, right = st.columns([0.2, 0.8])

# =============================
# LEFT PANEL
# =============================
with left:
    st.markdown('<div class="left-panel">', unsafe_allow_html=True)

    text_input = st.text_area(
        "Paste a review, tweet, or short paragraph",
        height=180,
        placeholder="Example: This movie was surprisingly good, with great performances and a strong ending.",
        key=st.session_state.input_key
    )
    st.caption("Max preview: 600 chars")

    uploaded = st.file_uploader(
        "Optional upload (.txt / .csv)",
        type=["txt", "csv"],
        key=st.session_state.upload_key
    )

    view_mode = st.radio(
        "View mode",
        ["Compare both", "TF-IDF + Logistic Regression", "BERT"],
    )

    run = st.button("Run analysis")

    st.subheader("System")
    st.write("🟢 Ready" if (text_input.strip() or uploaded) else "🟡 Waiting input")

    st.markdown("</div>", unsafe_allow_html=True)

# =============================
# RIGHT PANEL
# =============================
with right:
    st.markdown('<div class="right-panel">', unsafe_allow_html=True)

    render_pipeline(st.session_state.stage)

    st.markdown("""
    <div class="card">
        <h3>🚀 Sentiment Lab</h3>
        Compare classical NLP vs Transformer models.
    </div>
    """, unsafe_allow_html=True)

    if run:
        st.session_state.stage = "clean"
        st.rerun()

    if st.session_state.stage == "clean":
        raw_text = text_input.strip() or (read_uploaded_text(uploaded).strip() if uploaded else "")

        if not raw_text:
            st.warning("Please provide input.")
            st.session_state.stage = "input"
        else:
            cleaned_text = clean_text(raw_text)
            st.session_state._temp = (raw_text, cleaned_text)
            st.session_state.stage = "predict"
            st.rerun()

    if st.session_state.stage == "predict":
        raw_text, cleaned_text = st.session_state._temp

        trad = predict_traditional_text(cleaned_text, get_traditional_pipeline())
        tok, model, device = get_bert_components()
        bert = predict_bert_text(cleaned_text, tok, model, device=device)

        st.session_state.results = {
            "raw_text": raw_text,
            "cleaned_text": cleaned_text,
            "traditional": trad,
            "bert": bert
        }

        st.session_state.stage = "compare"
        st.rerun()

    if st.session_state.results:

        data = st.session_state.results

        st.markdown("## 🧠 Results")

        if len(data["raw_text"]) > 600:
            st.warning("Preview limited to 600 characters.")

        st.markdown(f"<div class='card'><b>Original text</b><br><br>{data['raw_text'][:600]}</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='card'><b>Cleaned text</b><br><br>{data['cleaned_text'][:600]}</div>", unsafe_allow_html=True)

        c1, c2 = st.columns(2)
        with c1:
            render_prediction_card("TF-IDF", data["traditional"])
        with c2:
            render_prediction_card("BERT", data["bert"])

        # =============================
        # BUTTONS
        # =============================
        c1, c2, c3 = st.columns(3)

        with c1:
            report_with_meta = {
                **data,
                "metadata": {
                    "author": "Alexandra de Almeida Ferreira",
                    "project": "NLP Sentiment Lab",
                    "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "links": {
                        "github": "https://github.com/dealmeidaferreiraAlexandra",
                        "linkedin": "https://www.linkedin.com/in/dealmeidaferreira",
                    },
                    "model_versions": {
                        "traditional": "tfidf_logreg_v1",
                        "bert": "bert-base-uncased"
                    }
                }
            }

            st.download_button(
                "⬇️ JSON",
                json.dumps(report_with_meta, indent=2, ensure_ascii=False),
                "report.json",
                use_container_width=True
            )

        with c2:
            if REPORTLAB_AVAILABLE:

                def bar(p):
                    total = 20
                    filled = int(p * total)
                    return "█"*filled + "░"*(total-filled)

                t = data["traditional"]
                b = data["bert"]

                buffer = io.BytesIO()
                doc = SimpleDocTemplate(buffer)
                styles = getSampleStyleSheet()

                title = ParagraphStyle('title', parent=styles['Title'], textColor=colors.HexColor("#6366f1"))
                section = ParagraphStyle('section', parent=styles['Heading2'], textColor=colors.HexColor("#8b5cf6"))

                elements = [
                    Paragraph("NLP Sentiment Lab Report", title),
                    Spacer(1, 12),

                    Paragraph("Input Text", section),
                    Paragraph(data["raw_text"], styles["Normal"]),
                    Spacer(1, 12),

                    Paragraph("TF-IDF", section),
                    Paragraph(f"Sentiment: {t['label']}", styles["Normal"]),
                    Paragraph(f"Confidence: {t['confidence']*100:.2f}%", styles["Normal"]),
                    Paragraph(f"Positive: {t['positive_prob']*100:.2f}%", styles["Normal"]),
                    Paragraph(f"Negative: {t['negative_prob']*100:.2f}%", styles["Normal"]),
                    Paragraph(bar(t["positive_prob"]), styles["Normal"]),
                    Spacer(1, 12),

                    Paragraph("BERT", section),
                    Paragraph(f"Sentiment: {b['label']}", styles["Normal"]),
                    Paragraph(f"Confidence: {b['confidence']*100:.2f}%", styles["Normal"]),
                    Paragraph(f"Positive: {b['positive_prob']*100:.2f}%", styles["Normal"]),
                    Paragraph(f"Negative: {b['negative_prob']*100:.2f}%", styles["Normal"]),
                    Paragraph(bar(b["positive_prob"]), styles["Normal"]),
                    Spacer(1, 12),
                ]

                if t["label"] != b["label"]:
                    elements.append(Paragraph("⚠ Models DISAGREE", styles["Normal"]))

                elements += [
                    Spacer(1, 20),
                    Paragraph("Developed by Alexandra de Almeida Ferreira", styles["Normal"]),
                    Paragraph("GitHub: github.com/dealmeidaferreiraAlexandra", styles["Normal"]),
                    Paragraph("LinkedIn: linkedin.com/in/dealmeidaferreira", styles["Normal"]),
                ]

                doc.build(elements)

                st.download_button(
                    "📄 PDF",
                    buffer.getvalue(),
                    "report.pdf",
                    use_container_width=True
                )
            else:
                st.info("PDF unavailable (missing reportlab)")

        with c3:
            if st.button("🔄 Reset", use_container_width=True):

                st.session_state.input_key = f"text_input_{time.time()}"
                st.session_state.upload_key = f"upload_{time.time()}"

                st.session_state.results = None
                st.session_state.stage = "input"

                st.rerun()

    st.markdown("</div>", unsafe_allow_html=True)

# =============================
# FOOTER
# =============================
st.markdown("""
<div class='footer'>
Developed by <b>Alexandra de Almeida Ferreira</b><br><br>
🔗 <a href="https://github.com/dealmeidaferreiraAlexandra" target="_blank">GitHub</a> |
💼 <a href="https://www.linkedin.com/in/dealmeidaferreira" target="_blank">LinkedIn</a>
</div>
""", unsafe_allow_html=True)
