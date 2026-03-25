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

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Legal Assistant — Indian Laws",
    page_icon="⚖️",
    layout="centered",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Intent badge pill */
    .intent-pill {
        display: inline-block;
        padding: 2px 10px;
        border-radius: 12px;
        font-size: 12px;
        font-weight: 500;
        background: #E6F1FB;
        color: #185FA5;
        margin-bottom: 6px;
    }

    /* Review gate warning box */
    .review-gate {
        background: #FAEEDA;
        border-left: 3px solid #BA7517;
        border-radius: 0 6px 6px 0;
        padding: 8px 12px;
        font-size: 13px;
        color: #633806;
        margin-top: 8px;
    }

    /* Source citation item */
    .source-item {
        background: #F1EFE8;
        border-radius: 6px;
        padding: 5px 10px;
        font-size: 12px;
        color: #444441;
        margin-bottom: 4px;
    }

    /* Audit ID code block */
    .audit-id {
        font-family: monospace;
        font-size: 11px;
        background: #F1EFE8;
        padding: 3px 7px;
        border-radius: 4px;
        color: #5F5E5A;
    }

    /* Hide Streamlit default footer */
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# ── Intent label map ──────────────────────────────────────────────────────────
INTENT_LABELS = {
    "consumer": "Consumer law",
    "tenancy":  "Tenancy law",
    "labour":   "Labour law",
    "criminal": "Criminal law",
    "family":   "Family law",
    "property": "Property law",
    "general":  "General legal query",
}

# ── Load orchestrator once (cached) ───────────────────────────────────────────
@st.cache_resource(show_spinner="Loading legal knowledge base...")
def load_bot():
    return Orchestrator()

bot = load_bot()

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

    # Language selector
    language = st.selectbox(
        "Response language",
        options=["English", "Hindi"],
        index=0,
        help="Hindi translation is powered by Gemini. IndicTrans2 can be added for higher quality."
    )
    lang_code = "hi" if language == "Hindi" else "en"

    st.markdown("---")
    st.markdown("### Try these examples")

    sample_queries = [
        "My landlord won't return my security deposit",
        "I bought a defective phone, what are my rights?",
        "My employer hasn't paid salary for 2 months",
        "I received a legal notice, what do I do?",
        "My landlord increased rent without notice",
        "How do I file a consumer complaint?",
        "My cheque bounced, what can I do?",
        "What is Section 138 of Negotiable Instruments Act?",
    ]

    for sample in sample_queries:
        if st.button(sample, use_container_width=True):
            st.session_state.pending_query = sample

    st.markdown("---")

    # Session stats
    total = len(st.session_state.audit_trail)
    if total > 0:
        blocked   = sum(1 for a in st.session_state.audit_trail if not a["safe"])
        emergency = sum(1 for a in st.session_state.audit_trail if a["emergency"])
        flagged   = sum(1 for a in st.session_state.audit_trail if a.get("review_flag"))

        st.markdown("### Session stats")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Queries",   total)
            st.metric("Blocked",   blocked)
        with col2:
            st.metric("Emergency", emergency)
            st.metric("Flagged",   flagged)

        st.markdown("---")
        st.download_button(
            label="Download audit log",
            data=json.dumps(st.session_state.audit_trail, indent=2, ensure_ascii=False),
            file_name="audit_trail.json",
            mime="application/json",
            use_container_width=True,
        )

    st.markdown("---")
    st.markdown("### About")
    st.caption(
        "Built for ET AI Hackathon 2026 · PS5\n\n"
        "Uses Gemini 1.5 Flash + FAISS + HuggingFace embeddings "
        "over Indian legal corpus. Every decision is logged."
    )

# ── Main header ───────────────────────────────────────────────────────────────
st.title("⚖️ AI Legal Assistant for Indian Laws")
st.caption(
    "Know your rights under Indian law — free, instant, in plain language. "
    "Covers IPC · CrPC · Consumer Protection · Tenancy · Labour and more."
)

# Emergency helpline quick reference
with st.expander("🆘 Emergency legal helplines", expanded=False):
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("**NALSA (Free Legal Aid)**\n\n📞 15100")
    with col2:
        st.markdown("**Women Helpline**\n\n📞 1091 / 181")
    with col3:
        st.markdown("**Police**\n\n📞 100")

st.markdown("---")

# ── Render chat history ───────────────────────────────────────────────────────
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ── Get query input ───────────────────────────────────────────────────────────
query = st.chat_input("Describe your legal situation in detail...")

# Handle sidebar sample button clicks
if "pending_query" in st.session_state:
    query = st.session_state.pop("pending_query")

# ── Process query ─────────────────────────────────────────────────────────────
if query:
    # Show user message immediately
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    # Run the full agent pipeline
    with st.chat_message("assistant"):
        with st.spinner("Searching Indian law database and reasoning..."):
            result = bot.run(query, language=lang_code)

        # ── Emergency banner (top priority) ───────────────────────────────────
        if result.get("emergency"):
            st.error(
                "🆘 **Urgent situation detected.** "
                "Emergency helpline numbers are included in the response. "
                "Please reach out for immediate help."
            )

        # ── Intent classification badge ────────────────────────────────────────
        if result.get("intent"):
            label = INTENT_LABELS.get(result["intent"], result["intent"])
            st.markdown(
                f'<div class="intent-pill">Query type: {label}</div>',
                unsafe_allow_html=True,
            )

        # ── Main answer ────────────────────────────────────────────────────────
        st.markdown(result["answer"])

        # ── Human review gate ─────────────────────────────────────────────────
        if result.get("review_flag"):
            st.markdown(
                '<div class="review-gate">'
                "⚠️ <strong>Human review recommended</strong> — "
                "This query involves a complex area (property/family law) where "
                "professional verification is strongly advised before acting."
                "</div>",
                unsafe_allow_html=True,
            )

        # ── Compliance block message ───────────────────────────────────────────
        if not result["safe"]:
            violations = result["compliance_detail"].get("violations_found", [])
            if violations:
                st.error(
                    f"🚫 Response blocked · Violation: {', '.join(violations)}"
                )

        # ── Audit trail expander ───────────────────────────────────────────────
        with st.expander("📋 How I answered this (Audit Trail)", expanded=False):

            # Top row: audit ID + compliance status
            col1, col2, col3 = st.columns([2, 1, 1])
            with col1:
                st.markdown("**Audit ID**")
                st.markdown(
                    f'<span class="audit-id">{result["audit_id"]}</span>',
                    unsafe_allow_html=True,
                )
            with col2:
                st.markdown("**Compliance**")
                st.markdown("✅ Passed" if result["safe"] else "❌ Blocked")
            with col3:
                st.markdown("**Emergency**")
                st.markdown("🆘 Yes" if result.get("emergency") else "— No")

            # Query processing info
            st.markdown("---")
            st.markdown("**Query classified as:**")
            st.caption(INTENT_LABELS.get(result.get("intent", "general"), "General"))

            # Sources
            st.markdown("**Legal sources retrieved:**")
            if result["sources"]:
                for src in result["sources"]:
                    st.markdown(
                        f'<div class="source-item">{src}</div>',
                        unsafe_allow_html=True,
                    )
            else:
                st.caption("No sources retrieved.")

            # Compliance detail
            st.markdown("**Compliance check reason:**")
            st.caption(result["compliance_detail"].get("reason", "N/A"))

            if result["compliance_detail"].get("violations_found"):
                st.warning(
                    "Violations found: "
                    + ", ".join(result["compliance_detail"]["violations_found"])
                )

            # Human review flag
            if result.get("review_flag"):
                st.info("This response was flagged for human review.")

    # ── Store in session state ─────────────────────────────────────────────────
    st.session_state.messages.append({
        "role":    "assistant",
        "content": result["answer"],
    })
    st.session_state.audit_trail.append({
        "query":        query,
        "audit_id":     result["audit_id"],
        "safe":         result["safe"],
        "emergency":    result.get("emergency", False),
        "review_flag":  result.get("review_flag", False),
        "intent":       result.get("intent", "general"),
        "sources":      result["sources"],
        "language":     lang_code,
    })