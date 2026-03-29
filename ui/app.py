"""
ui/app_v2_tabs.py
-----------------
PHASE 2: Updated Streamlit UI with 3-tab layout:
  Tab 1: Simple Query (original chat interface)
  Tab 2: FIR Analyzer (FIR photo → 4 legal documents + checklist)
  Tab 3: Loophole Finder (case facts → legal vulnerabilities + risk analysis)

Run with:
    streamlit run ui/app_v2_tabs.py
"""

import sys
import json
import base64
import os
import io
import tempfile
import time
import zipfile
from datetime import datetime

sys.path.append(".")

# Check if running within streamlit
import streamlit as st
try:
    if not st.runtime.exists():
        print("\n" + "="*50)
        print("  ERROR: Please run this app using Streamlit:")
        print("     streamlit run ui/app_v2_tabs.py")
        print("="*50 + "\n")
        sys.exit(1)
except:
    pass  # During import tests, ignore streamlit check

from agents.orchestrator import Orchestrator
from agents.fir_processor import FIRProcessor
from agents.loophole_finder import LoopholeFinder
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
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS with Animations (from original) ────────────────────────────────
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
    
    /* Tab styling */
    [data-baseweb="tab-list"] {
        gap: 8px;
    }
    
    [data-testid="stTab"] {
        border-radius: 8px;
        font-weight: 600;
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

# ── Load Agents ───────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading legal knowledge base...")
def load_orchestrator():
    return Orchestrator()

@st.cache_resource(show_spinner="Loading FIR Processor...")
def load_fir_processor():
    return FIRProcessor()

@st.cache_resource(show_spinner="Loading Loophole Finder...")
def load_loophole_finder():
    return LoopholeFinder()

orchestrator = load_orchestrator()
fir_processor = load_fir_processor()
loophole_finder = load_loophole_finder()

# ── Helper Functions ──────────────────────────────────────────────────────────
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
    pdf.set_font("Helvetica", size=11)
    pdf.set_margins(10, 10, 10)
    pdf.multi_cell(0, 4, text=safe_text)
    return bytes(pdf.output())

# ── Helper: Get correct MIME type from filename ──────────────────────────────
def get_mime_type_from_filename(filename):
    """
    Derive correct MIME type from filename extension.
    Streamlit's file.type is unreliable, so we use filename instead.
    """
    ext = filename.lower().split('.')[-1]
    mime_map = {
        'jpg': 'image/jpeg',
        'jpeg': 'image/jpeg',
        'png': 'image/png',
        'pdf': 'application/pdf',
    }
    return mime_map.get(ext, 'image/jpeg')

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
    prompt = "You are a legal assistant. Please read this document, extract its key points, and explain what it means in extremely simple language."
    resp = client.models.generate_content(
        model=MODEL_NAME,
        contents=[
            types.Part.from_bytes(data=file_bytes, mime_type=mime_type),
            prompt
        ]
    )
    return resp.text

def create_zip_with_pdfs(pdfs_dict, facts):
    """Create a ZIP file containing all PDFs and metadata"""
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, 'w') as zip_file:
        # Add PDFs
        for doc_name, pdf_content in pdfs_dict.items():
            if pdf_content and len(pdf_content) > 0:
                zip_file.writestr(f"{doc_name}.pdf", pdf_content)
        
        # Add metadata JSON
        metadata = {
            "generated_at": datetime.now().isoformat(),
            "fir_number": facts.get("fir_number", "N/A"),
            "location": facts.get("location", "N/A"),
            "crime_type": facts.get("crime_type", "N/A"),
            "documents_included": list(pdfs_dict.keys())
        }
        zip_file.writestr("metadata.json", json.dumps(metadata, indent=2))
    
    return zip_buffer.getvalue()

# ── Session state init ────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "audit_trail" not in st.session_state:
    st.session_state.audit_trail = []
if "fir_processed_count" not in st.session_state:
    st.session_state.fir_processed_count = 0
if "loopholes_found_count" not in st.session_state:
    st.session_state.loopholes_found_count = 0
if "docs_generated_count" not in st.session_state:
    st.session_state.docs_generated_count = 0

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
    st.markdown("### 📊 Hackathon Metrics")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("FIRs Processed", st.session_state.fir_processed_count)
        st.metric("Loopholes Found", st.session_state.loopholes_found_count)
    with col2:
        st.metric("Docs Generated", st.session_state.docs_generated_count)

    st.markdown("---")
    st.markdown("### 💡 Try these examples")
    sample_queries = [
        "My landlord won't return my security deposit",
        "I bought a defective phone, what are my rights?",
        "My employer hasn't paid salary for 2 months",
        "How do I file a consumer complaint?",
    ]
    for sample in sample_queries:
        if st.button(sample, width='stretch'):
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
        width='stretch'
    ):
        st.toast("Dossier downloaded successfully! 🚀", icon="✅")

    st.markdown("---")
    with st.expander("🚨 Emergency legal helplines", expanded=False):
        st.markdown("**NALSA (Free Legal Aid)**\n\n📞 15100")
        st.markdown("**Women Helpline**\n\n📞 1091 / 181")
        st.markdown("**Police**\n\n📞 100")

# ── Main header ───────────────────────────────────────────────────────────────
st.markdown('<h1>⚖️ AI Legal Assistant for Indian Laws</h1>', unsafe_allow_html=True)
st.markdown('<h4>Know your rights. Generate drafts. Analyze loopholes. Talk in your language.</h4>', unsafe_allow_html=True)

# ── TAB STRUCTURE ─────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["💬 Simple Query", "📸 FIR Analyzer", "🔍 Loophole Finder"])

# ═════════════════════════════════════════════════════════════════════════════
# TAB 1: SIMPLE QUERY (Original Chat Interface)
# ═════════════════════════════════════════════════════════════════════════════
with tab1:
    st.markdown("### Chat with the AI Legal Assistant")
    
    colA, colB = st.columns([1, 1])
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
            if st.button("🚀 Send to Assistant", type="primary", width='stretch'):
                st.session_state.pending_query = edited_query
                del st.session_state.audio_draft
                st.rerun()
        with col_discard:
            if st.button("🗑️ Discard", width='stretch'):
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
                time.sleep(0.5)
                result = orchestrator.run(query, language=lang_code)
            
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
                st.markdown(f"**Audit ID:** `{result.get('audit_id', 'N/A')}`")
                for src in result.get("sources", []):
                    preview = (src[:80] + "...") if len(src) > 80 else src
                    with st.expander(f"📖 {preview}"):
                        st.markdown(src)


# ═════════════════════════════════════════════════════════════════════════════
# TAB 2: FIR ANALYZER (Transform FIR Photo → Legal Documents)
# ═════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown("### 📸 FIR Analyzer: Transform Photos into Legal Documents")
    st.caption("Upload a photo of your FIR (First Information Report) and we'll automatically generate 4 legal documents ready to file.")
    
    col_info1, col_info2, col_info3, col_info4 = st.columns(4)
    with col_info1:
        st.metric("📄 Documents", "4 PDFs")
    with col_info2:
        st.metric("⚡ Speed", "<10 sec")
    with col_info3:
        st.metric("✅ Accuracy", "AI Vision")
    with col_info4:
        st.metric("📋 Formats", "Ready-to-file")
    
    st.markdown("---")
    
    fir_upload = st.file_uploader(
        "📸 Upload FIR Photo (JPG/PNG)",
        type=["jpg", "jpeg", "png"],
        key="fir_uploader",
        accept_multiple_files=False
    )
    
    if fir_upload:
        st.markdown("---")
        
        # Show uploaded image
        col_img, col_info = st.columns([1, 1])
        with col_img:
            st.image(fir_upload, caption="Uploaded FIR Photo", width='stretch')
        
        with col_info:
            st.info("🔍 Image will be analyzed for:\n- FIR details extraction\n- Crime classification\n- Incident location\n- Victim & accused info\n- Legal sections applicable")
        
        st.markdown("---")
        
        if st.button("🚀 Generate Legal Documents", type="primary", width='stretch'):
            progress_placeholder = st.empty()
            status_placeholder = st.empty()
            
            try:
                # Convert image to bytes
                image_bytes = fir_upload.read()
                
                # Get correct MIME type from filename (not file.type which is unreliable)
                mime_type = get_mime_type_from_filename(fir_upload.name)
                
                # Process through FIRProcessor
                with progress_placeholder.container():
                    progress = st.progress(0)
                    status_placeholder.info("📸 Stage 1/7: Extracting facts from FIR photo...")
                
                start_time = time.time()
                result = fir_processor.process(image_bytes, mime_type=mime_type)
                elapsed = time.time() - start_time
                
                # Update metrics
                st.session_state.fir_processed_count += 1
                st.session_state.docs_generated_count += 4
                
                # Show results
                st.success(f"✅ Documents generated in {elapsed:.1f}s!")
                
                st.markdown("---")
                st.markdown("### 📋 Extracted FIR Details")
                
                col_facts, col_laws = st.columns([1, 1])
                
                with col_facts:
                    st.markdown("**Extracted Facts:**")
                    if result.get("facts_extracted"):
                        for key, value in result["facts_extracted"].items():
                            st.caption(f"**{key}:** {value}")
                
                with col_laws:
                    st.markdown("**Applicable Laws:**")
                    if result.get("laws_identified"):
                        for law in result["laws_identified"][:3]:
                            st.caption(f"• {law}")
                
                st.markdown("---")
                st.markdown("### 📊 Risk Assessment")
                
                if result.get("risk_assessment"):
                    risk = result["risk_assessment"]
                    col_risk1, col_risk2, col_risk3 = st.columns(3)
                    
                    with col_risk1:
                        st.metric("Bail Eligibility", risk.get("bail_probability", "N/A"))
                    with col_risk2:
                        st.metric("Custody Risk", risk.get("custody_likelihood", "N/A"))
                    with col_risk3:
                        st.metric("Case Severity", risk.get("severity_level", "N/A"))
                
                st.markdown("---")
                st.markdown("### 📄 Generated Documents")
                
                # Display PDFs and provide downloads
                doc_cols = st.columns(4)
                doc_names = ["complaint", "bail", "petition", "cover_letter"]
                doc_labels = ["📋 Complaint", "⚖️ Bail Application", "📑 Petition", "📮 Cover Letter"]
                
                for idx, (doc_col, doc_name, doc_label) in enumerate(zip(doc_cols, doc_names, doc_labels)):
                    with doc_col:
                        if result.get("pdfs", {}).get(doc_name):
                            pdf_bytes = result["pdfs"][doc_name]
                            st.download_button(
                                f"⬇️ {doc_label}",
                                data=pdf_bytes,
                                file_name=f"{doc_name}.pdf",
                                mime="application/pdf",
                                width='stretch'
                            )
                            st.caption(f"{len(pdf_bytes) / 1024:.1f} KB")
                        else:
                            st.warning(f"⚠️ {doc_label}\nGeneration failed")
                
                st.markdown("---")
                
                # Download all as ZIP
                st.markdown("### 📦 Download All Documents")
                zip_data = create_zip_with_pdfs(
                    result.get("pdfs", {}),
                    result.get("facts_extracted", {})
                )
                st.download_button(
                    "📥 Download All PDFs + Metadata (ZIP)",
                    data=zip_data,
                    file_name=f"FIR_Documents_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip",
                    mime="application/zip",
                    width='stretch'
                )
                
                st.markdown("---")
                st.markdown("### ✅ Next Steps")
                
                if result.get("next_steps"):
                    # If it's a string from the FIR processor, display it directly
                    if isinstance(result["next_steps"], str):
                        st.markdown(result["next_steps"])
                    else:
                        # Fallback in case it's a list
                        for i, step in enumerate(result["next_steps"], 1):
                            st.markdown(f"**{i}. {step}**")
                
                st.success("🎉 All documents are ready! Download and file them immediately.")
                st.balloons()
                
            except Exception as e:
                st.error(f"❌ Error processing FIR: {str(e)}")
                st.info("💡 Tips:\n- Ensure the image is clear\n- Photo should show the full FIR form\n- Try a different angle if text is not readable")


# ═════════════════════════════════════════════════════════════════════════════
# TAB 3: LOOPHOLE FINDER (Analyze Case for Legal Vulnerabilities)
# ═════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown("### 🔍 Loophole Finder: Detect Legal Vulnerabilities & Risk Assessment")
    st.caption("Analyze a case to identify legal loopholes, model opponent strategies, and get risk scores.")
    
    col_metric1, col_metric2, col_metric3, col_metric4 = st.columns(4)
    with col_metric1:
        st.metric("🎯 Analysis", "7-Stage")
    with col_metric2:
        st.metric("🔐 Loopholes", "Detected")
    with col_metric3:
        st.metric("📊 Risks", "Scored")
    with col_metric4:
        st.metric("⏱️ Time", "<10 sec")
    
    st.markdown("---")
    
    st.markdown("### Enter Case Details")
    
    # Case facts input
    case_facts_text = st.text_area(
        "📝 Describe your case (facts, history, key arguments):",
        placeholder="Example: I was charged with fraud under section 420 IPC. The complainant claims I misrepresented property details, but the property has been registered in my name for 5 years with no prior objection...",
        height=200
    )
    
    col_crime, col_jurisdiction = st.columns(2)
    
    with col_crime:
        crime_type = st.selectbox(
            "Crime Type:",
            ["---", "Fraud", "Theft", "Assault", "Criminal Intimidation", "Cheating", 
             "Defamation", "Sexual Assault", "Attempt to Murder", "Other"]
        )
    
    with col_jurisdiction:
        jurisdiction = st.selectbox(
            "Jurisdiction (State):",
            ["---", "Delhi", "Maharashtra", "Karnataka", "Tamil Nadu", "Gujarat", 
             "Uttar Pradesh", "West Bengal", "Other"]
        )
    
    st.markdown("---")
    
    if st.button("🚀 Analyze for Loopholes & Risks", type="primary", width='stretch'):
        if not case_facts_text or crime_type == "---":
            st.warning("⚠️ Please provide case details and select a crime type.")
        else:
            progress_placeholder = st.empty()
            
            try:
                progress_placeholder.info("🔍 Analyzing case for legal vulnerabilities...")
                
                start_time = time.time()
                
                # Prepare case facts string for analysis (loophole_finder.analyze() expects STRING, not dict)
                combined_case_facts = f"{case_facts_text}\n\nCrime Type: {crime_type}\nJurisdiction: {jurisdiction}"
                
                # Process through LoopholeFinder
                result = loophole_finder.analyze(combined_case_facts)
                elapsed = time.time() - start_time
                
                # Update metrics
                st.session_state.loopholes_found_count += len(result.get("loopholes", []))
                
                # Clear progress and show results
                progress_placeholder.success(f"✅ Analysis completed in {elapsed:.1f}s!")
                
                st.markdown("---")
                st.markdown("### 🔓 Detected Loopholes")
                
                if result.get("loopholes"):
                    for i, loophole in enumerate(result["loopholes"], 1):
                        with st.expander(f"**Loophole {i}: {loophole.get('loophole_name', 'Unknown')}** (Risk: {loophole.get('opponent_risk_percentage', 'N/A')}%)"):
                            st.markdown(f"**Section:** {loophole.get('affected_section', 'N/A')}")
                            st.markdown(f"**Description:** {loophole.get('description', 'N/A')}")
                            st.markdown(f"**How Opponent Exploits:** {loophole.get('how_opponent_exploits', 'N/A')}")
                            with st.expander("📖 Law Excerpt", expanded=False):
                                st.markdown(loophole.get('law_excerpt', 'N/A'))
                else:
                    st.info("✅ No major loopholes detected in this case.")
                
                st.markdown("---")
                st.markdown("### ⚖️ Opponent Strategy Analysis")
                
                if result.get("opponent_strategies"):
                    for i, strategy in enumerate(result["opponent_strategies"], 1):
                        with st.expander(f"**Risk {i}: {strategy.get('loophole', 'Unknown')}** ({strategy.get('likelihood_percentage', 'N/A')}% likely)"):
                            st.markdown(f"**Their Argument:** {strategy.get('opponent_argument', 'N/A')}")
                            st.markdown(f"**Your Counter-Argument:** {strategy.get('counterargument', 'N/A')[:500]}")
                            
                            if strategy.get('supporting_precedents'):
                                with st.expander("📚 Precedents They'll Cite", expanded=False):
                                    for precedent in strategy['supporting_precedents']:
                                        st.markdown(f"• {precedent}")
                            
                            if strategy.get('legal_reasoning'):
                                with st.expander("⚔️ Their Legal Reasoning", expanded=False):
                                    st.markdown(strategy['legal_reasoning'])
                
                st.markdown("---")
                st.markdown("### 📊 Risk Assessment Scores")
                
                if result.get("risk_scores"):
                    scores = result["risk_scores"]
                    
                    col_win, col_custody, col_appeal = st.columns(3)
                    
                    with col_win:
                        win_prob = scores.get("win_probability", 0)
                        st.metric(
                            "Win Probability",
                            f"{win_prob}%",
                            delta="Higher is better"
                        )
                    
                    with col_custody:
                        custody_risk = scores.get("custody_risk", 0)
                        st.metric(
                            "Custody Risk",
                            f"{custody_risk}%",
                            delta="Lower is better",
                            delta_color="inverse"
                        )
                    
                    with col_appeal:
                        appeal_prob = scores.get("appeal_probability", 0)
                        st.metric(
                            "Appeal Success",
                            f"{appeal_prob}%",
                            delta="Higher is better"
                        )
                
                st.markdown("---")
                st.markdown("### 🛡️ Counter-Strategies for Your Defense")
                
                if result.get("counter_strategies"):
                    for i, counter in enumerate(result["counter_strategies"], 1):
                        with st.expander(f"**Defense {i}: {counter.get('loophole', 'Strategy ' + str(i))}**"):
                            st.markdown(f"**How to Counter:** {counter.get('counterargument', 'N/A')}")
                else:
                    st.info("Strategies will be displayed based on identified loopholes.")
                
                st.markdown("---")
                st.markdown("### ⚖️ Suggested Legislative Amendments")
                
                if result.get("amendments"):
                    for i, amendment in enumerate(result["amendments"], 1):
                        st.markdown(f"**Amendment {i}:** {amendment}")
                else:
                    st.info("No specific amendments suggested for this case type.")
                
                st.markdown("---")
                st.markdown("### 📋 Full Strategy Report")
                
                if result.get("strategy_report"):
                    with st.expander("📖 View Complete Analysis Report", expanded=False):
                        st.markdown(result["strategy_report"])
                    
                    # Download report as PDF
                    report_pdf = generate_pdf(result["strategy_report"])
                    st.download_button(
                        "📥 Download Strategy Report (PDF)",
                        data=report_pdf,
                        file_name=f"Loophole_Strategy_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                        mime="application/pdf",
                        width='stretch'
                    )
                
                st.success("✅ Analysis complete! Use these insights to strengthen your case.")
                
            except Exception as e:
                st.error(f"❌ Error analyzing case: {str(e)}")
                st.info("💡 Tips:\n- Provide detailed case facts\n- Mention specific laws and sections\n- Include key dates and parties involved")


# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style="text-align: center; opacity: 0.7; font-size: 12px;">
    <p>⚖️ AI Legal Assistant | Always consult a qualified advocate for legal matters</p>
    <p>Powered by Google Gemini · FAISS · Indian Legal Database</p>
</div>
""", unsafe_allow_html=True)
