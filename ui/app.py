"""
ui/app.py
----------
Streamlit chat interface for the AI Legal Assistant.
Includes: query intent display, human review gate, audit trail,
emergency detection, multilingual selector, sample queries.

Run with:
    streamlit run ui/app.py
"""

import sys
import json
import base64
import os
import io
import tempfile
import time
sys.path.append(".")

# Check if running within streamlit
import streamlit as st
if not st.runtime.exists():
    print("\n" + "="*50)
    print("  ❌ ERROR: Please run this app using Streamlit:")
    print("     streamlit run ui/app.py")
    print("="*50 + "\n")
    sys.exit(1)

from agents.orchestrator import Orchestrator
from config import client, MODEL_NAME
try:
    from gtts import gTTS
except ImportError:
    st.error("Please install gTTS: pip install gTTS")
try:
    from fpdf import FPDF
except ImportError:
    st.error("Please install fpdf: pip install fpdf2")

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Legal Assistant — Indian Laws",
    page_icon="⚖️",
    layout="centered",
    initial_sidebar_state="expanded",
)

# ── Custom CSS with Animations ────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');

    /* Global Typography */
    html, body, [class*="css"]  {
        font-family: 'Plus Jakarta Sans', sans-serif !important;
    }

    /* Main Title Styling & Alignment */
    h1 {
        font-weight: 800 !important;
        text-align: center;
        padding-bottom: 5px;
        animation: fadeInDown 0.6s ease-out;
    }
    
    h4 {
        text-align: center;
        font-weight: 500 !important;
        margin-bottom: 25px !important;
        animation: fadeIn 0.8s ease-in;
        opacity: 0.8;
    }

    /* Message Bubbles - Effect & Animation */
    @keyframes fadeInUp {
        from { opacity: 0; transform: translateY(15px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    @keyframes fadeInDown {
        from { opacity: 0; transform: translateY(-15px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    .stChatMessage {
        animation: fadeInUp 0.5s cubic-bezier(0.2, 0.8, 0.2, 1) forwards;
        border-radius: 12px;
        padding: 5px 10px;
        margin-bottom: 12px;
        box-shadow: 0 2px 6px rgba(0,0,0,0.1);
        border: 1px solid rgba(255,255,255,0.1);
        transition: all 0.3s ease;
    }
    
    /* Elegant Sidebar */
    [data-testid="stSidebar"] {
        border-right: 1px solid rgba(255,255,255,0.1);
    }
    
    /* Button Effects */
    .stButton > button {
        font-weight: 600;
        border-radius: 8px;
        padding: 10px 20px;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    }
    .stButton > button:hover {
        transform: translateY(-2px);
    }

    /* Floating Intent Badge Pill */
    .intent-pill {
        display: inline-block;
        padding: 6px 16px;
        border-radius: 20px;
        font-size: 13px;
        font-weight: 700;
        background: rgba(29, 78, 216, 0.2);
        color: #60a5fa;
        margin-bottom: 16px;
        border: 1px solid #3b82f6;
        box-shadow: 0 2px 6px rgba(37,99,235,0.1);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .intent-pill:hover {
        transform: translateY(-2px);
    }

    /* Review gate warning box with pulse */
    @keyframes softPulse {
        0% { border-left-color: #f59e0b; box-shadow: 0 0 0 rgba(245,158,11,0); }
        50% { border-left-color: #fbbf24; box-shadow: 0 0 8px rgba(245,158,11,0.2); }
        100% { border-left-color: #f59e0b; box-shadow: 0 0 0 rgba(245,158,11,0); }
    }

    .review-gate {
        background: rgba(245, 158, 11, 0.1);
        border-left: 5px solid #f59e0b;
        border-radius: 6px 12px 12px 6px;
        padding: 12px 16px;
        font-size: 14px;
        font-weight: 500;
        color: #fcd34d;
        margin-top: 16px;
        animation: softPulse 2.5s infinite;
    }

    /* Clean Source citation cards */
    .source-item {
        background: rgba(255,255,255,0.05);
        border-left: 3px solid #cbd5e1;
        border-radius: 6px;
        padding: 10px 14px;
        font-size: 13px;
        margin-bottom: 8px;
        transition: all 0.2s ease;
    }
    .source-item:hover {
        border-left-color: #3b82f6;
        background: rgba(255,255,255,0.1);
        transform: translateX(4px);
    }

    /* Beautiful Audit ID */
    .audit-id {
        font-family: 'JetBrains Mono', 'Courier New', monospace;
        font-size: 11px;
        background: rgba(255,255,255,0.1);
        padding: 4px 8px;
        border-radius: 6px;
        border: 1px solid rgba(255,255,255,0.2);
    }

    /* Elegant Expander styling */
    .streamlit-expanderHeader {
        font-weight: 600 !important;
        border-radius: 8px;
        padding: 8px 12px !important;
        transition: background 0.3s;
    }

    /* Audio & File Uploader Effect */
    [data-testid="stFileUploadDropzone"], audio {
        border-radius: 12px;
        transition: all 0.3s ease;
    }
    [data-testid="stFileUploadDropzone"]:hover {
        border-color: #3b82f6;
        transform: scale(1.01);
    }
    
    /* Input field styling */
    .stChatInput {
        border-radius: 24px;
        transition: all 0.3s ease;
    }

    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

INTENT_LABELS = {
    "consumer": "Consumer law", "tenancy":  "Tenancy law",
    "labour":   "Labour law",   "criminal": "Criminal law",
    "family":   "Family law",   "property": "Property law",
    "general":  "General legal query",
}

@st.cache_resource(show_spinner="Loading legal knowledge base...")
def load_bot():
    return Orchestrator()

bot = load_bot()

# Helper Functions
def text_to_speech(text, lang_code):
    try:
        gtts_lang = "en"
        maps = {"hi":"hi", "bn":"bn", "te":"te", "mr":"mr", "ta":"ta", "ur":"ur", "gu":"gu", "kn":"kn", "ml":"ml"}
        if lang_code in maps: gtts_lang = maps[lang_code]
        tts = gTTS(text=text, lang=gtts_lang)
        fp = io.BytesIO()
        tts.write_to_fp(fp)
        return fp.getvalue()
    except Exception as e:
        return None

def generate_pdf(text):
    safe_text = text.encode('latin-1', 'replace').decode('latin-1')
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.multi_cell(0, 10, txt=safe_text)
    return pdf.output(dest='S').encode('latin-1')

def generate_dossier():
    dossier = "==== PRO-BONO LEGAL DOSSIER ====\n\n"
    for msg in st.session_state.messages:
        role = "CLIENT" if msg["role"] == "user" else "AI ASSISTANT"
        dossier += f"[{role}]:\n{msg['content']}\n\n"
    dossier += "=================================\n"
    dossier += "Transmitted securely for legal aid review."
    return dossier.encode('utf-8')

def parse_audio(audio_bytes):
    from google.genai import types
    from config import client, MODEL_NAME
    prompt = "Please transcribe this audio exactly in the language it is spoken, returning only the text."
    resp = client.models.generate_content(
        model=MODEL_NAME,
        contents=[
            types.Part.from_bytes(data=audio_bytes, mime_type="audio/wav"),
            prompt
        ]
    )
    return resp.text

def analyze_document(file_bytes, mime_type):
    from google.genai import types
    from config import client, MODEL_NAME
    prompt = "You are a legal assistant. Please read this document, extract its key points, and explain what it means in extremely simple language."
    resp = client.models.generate_content(
        model=MODEL_NAME,
        contents=[
            types.Part.from_bytes(data=file_bytes, mime_type=mime_type),
            prompt
        ]
    )
    return resp.text

# ── Session state init ────────────────────────────────────────────────────────
if "messages"     not in st.session_state:
    st.session_state.messages     = []
if "audit_trail"  not in st.session_state:
    st.session_state.audit_trail  = []

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚖️ AI Legal Assistant")
    st.caption("Indian Laws · Free · Plain Language")
    st.markdown("---")

    language = st.selectbox(
        "Response language ✨",
        options=["English", "Hindi", "Bengali", "Telugu", "Marathi", "Tamil", "Urdu", "Gujarati", "Kannada", "Odia", "Malayalam", "Punjabi", "Assamese"],
        index=0
    )
    lang_code_map = {
        "English": "en", "Hindi": "hi", "Bengali": "bn", "Telugu": "te", "Marathi": "mr",
        "Tamil": "ta", "Urdu": "ur", "Gujarati": "gu", "Kannada": "kn", "Odia": "or",
        "Malayalam": "ml", "Punjabi": "pa", "Assamese": "as"
    }
    lang_code = lang_code_map.get(language, "en")

    st.markdown("---")
    st.markdown("### 💡 Try these examples")
    sample_queries = [
        "My landlord won't return my security deposit",
        "I bought a defective phone, what are my rights?",
        "My employer hasn't paid salary for 2 months",
        "How do I file a consumer complaint?",
    ]
    for sample in sample_queries:
        if st.button(sample, use_container_width=True):
            st.session_state.pending_query = sample

    st.markdown("---")
    # Dossier generation (NALSA)
    st.markdown("### 🤝 Free Legal Aid")
    st.caption("Pack your case to send to a lawyer or NALSA.")
    if st.download_button(
        "📥 Download Case Dossier", 
        data=generate_dossier() if len(st.session_state.messages) > 0 else b"No case history yet.",
        file_name="Case_Dossier.txt", 
        mime="text/plain", 
        use_container_width=True
    ):
        st.toast("Dossier downloaded successfully! 🚀", icon="✅")
        st.balloons()

# ── Main header ───────────────────────────────────────────────────────────────
st.markdown('<h1>⚖️ AI Legal Assistant for Indian Laws</h1>', unsafe_allow_html=True)
st.markdown('<h4>Know your rights. Generate drafts. Talk in your language.</h4>', unsafe_allow_html=True)

with st.expander("🚨 Emergency legal helplines", expanded=False):
    col1, col2, col3 = st.columns(3)
    with col1: st.markdown("**NALSA (Free Legal Aid)**\n\n📞 15100")
    with col2: st.markdown("**Women Helpline**\n\n📞 1091 / 181")
    with col3: st.markdown("**Police**\n\n📞 100")

# Input methods at the top, or chat
colA, colB = st.columns([1,1])
with colA:
    audio_input = st.audio_input("🎙️ Speak your issue (Voice Input)")
with colB:
    doc_upload = st.file_uploader("📄 Upload Notice/Agreement (OCR)", type=["png", "jpg", "pdf"])

st.markdown("---")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("audio_data"):
            st.audio(msg["audio_data"], format="audio/mp3")
        if msg.get("draft_text"):
            st.download_button("📄 Download Draft (PDF)", data=generate_pdf(msg["draft_text"]), file_name="Legal_Draft.pdf", mime="application/pdf", key="draft_"+str(id(msg)))

query = st.chat_input("Or type your legal situation here...")
if "pending_query" in st.session_state:
    query = st.session_state.pop("pending_query")

# Handle Document Upload
if doc_upload and doc_upload not in st.session_state.get('processed_docs', []):
    if 'processed_docs' not in st.session_state:
        st.session_state.processed_docs = []
    st.session_state.processed_docs.append(doc_upload)
    with st.spinner("🔍 Magically analyzing your document..."):
        file_bytes = doc_upload.read()
        mime = "application/pdf" if doc_upload.name.endswith(".pdf") else "image/png"
        explanation = analyze_document(file_bytes, mime)
        st.session_state.messages.append({"role": "user", "content": f"Uploaded File: {doc_upload.name}"})
        ans = f"**Document Analysis**\n\n{explanation}"
        st.session_state.messages.append({"role": "assistant", "content": ans})
        st.rerun()

# Handle Audio Input
if audio_input and audio_input not in st.session_state.get('processed_audio', []):
    if 'processed_audio' not in st.session_state:
        st.session_state.processed_audio = []
    st.session_state.processed_audio.append(audio_input)
    with st.spinner("🎧 Transcribing your audio..."):
        audio_bytes = audio_input.read()
        transcribed_text = parse_audio(audio_bytes)
        st.session_state.audio_draft = transcribed_text
        st.rerun()

# ── Allow user to Edit & Confirm Audio Transcription ──
if "audio_draft" in st.session_state:
    st.info("🎤 Audio transcribed! You can edit the text below before sending:")
    edited_query = st.text_area("Edit your issue:", value=st.session_state.audio_draft, height=80)
    col_send, col_discard, _ = st.columns([2, 2, 4])
    with col_send:
        if st.button("🚀 Send to Assistant", type="primary", use_container_width=True):
            st.session_state.pending_query = edited_query
            del st.session_state.audio_draft
            st.rerun()
    with col_discard:
        if st.button("🗑️ Discard", use_container_width=True):
            del st.session_state.audio_draft
            st.toast("Audio discarded.", icon="🗑️")
            st.rerun()

# Process Text/Audio Query
if query:
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        with st.spinner("⚖️ Searching through Indian law..."):
            time.sleep(0.5) # Slight pause for effect
            result = bot.run(query, language=lang_code)
        
        if result.get("emergency"): 
            st.error("🆘 **Urgent situation detected! Please call the helpline immediately.**")
        
        if result.get("intent"): 
            st.markdown(f'<div class="intent-pill">🏷️ Type: {INTENT_LABELS.get(result["intent"], result["intent"])}</div>', unsafe_allow_html=True)
            
        st.markdown(result["answer"])

        # Auto generate audio for the response
        audio_data = None
        with st.spinner("🔊 Generating Voice Output..."):
            audio_data = text_to_speech(result["answer"], lang_code)
            if audio_data:
                st.audio(audio_data, format="audio/mp3")
        
        # If user intent is consumer or tenancy, let's offer a draft
        draft_text = None
        if result.get("safe") and result.get("intent") in ["consumer", "tenancy"]:
            with st.spinner("✍️ Auto-generating a professional legal draft..."):
                draft_prompt = f"Draft a formal complaint letter or legal notice regarding: {query}. Keep it professional."
                draft_resp = client.models.generate_content(model=MODEL_NAME, contents=draft_prompt)
                draft_text = draft_resp.text
                st.success("✨ A legal draft has been instantly generated for you!")
                st.download_button("📄 Download Signed Draft (PDF)", data=generate_pdf(draft_text), file_name="Legal_Draft.pdf", mime="application/pdf")
                st.toast("Tap to download your drafted file!", icon="📄")

        if result.get("review_flag"):
            st.markdown('<div class="review-gate">⚠️ <b>Human review recommended</b>—This case involves complex facts.</div>', unsafe_allow_html=True)

        st.session_state.messages.append({
            "role": "assistant",
            "content": result["answer"],
            "audio_data": audio_data,
            "draft_text": draft_text
        })
        st.session_state.audit_trail.append(result)

        # Draw Audit Trail nicely
        with st.expander("📋 View Reasoning & Audit Trail"):
            st.markdown(f"**Audit ID:** `<span class='audit-id'>{result.get('audit_id', 'N/A')}</span>`", unsafe_allow_html=True)
            for src in result.get("sources", []):
                # Split source to show a short preview as the expander label, and the full text inside
                preview = (src[:80] + "...") if len(src) > 80 else src
                with st.expander(f"📖 {preview}"):
                    st.markdown(src)





