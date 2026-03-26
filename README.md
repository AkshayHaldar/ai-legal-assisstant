# ⚖️ AI Legal Assistant for Indian Laws

**ET AI Hackathon 2026 — Problem Statement 5: Domain-Specialized AI Agents with Compliance Guardrails**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge.svg)](https://share.streamlit.io/)

An intelligent, multi-agent AI system designed to decode complex Indian laws into plain, actionable language for everyday citizens. No jargon. No fees. Just pure clarity. 

Designed specifically to improve **Access to Justice (A2J)** for the common Indian citizen.

---

## 🚀 The 6 Unique Magic Features

We didn't just build a legal search engine; we built an interactive AI lawyer. Our standout features include:

### 1. 🎙️ Multilingual Voice Input & Output
Why type a long legal issue when you can just talk?
*   **Speech-to-Text**: Click the microphone icon to speak your issue. Uses Gemini's Multimodal engine to transcribe seamlessly in English, Hindi, Bengali, Tamil, etc.
*   **Text-to-Speech (TTS)**: The AI automatically generates a voice output reading its legal advice aloud in your chosen regional language.

### 2. 📄 Instant Document Analysis (OCR)
People are often confused by dense legal notices.
*   **Upload Feature**: Users can upload pictures (JPG/PNG) or PDFs of a Rent Agreement, Police FIR, or Legal Notice.
*   **Plain English Summary**: The AI extracts the difficult terms and explains what the document *actually* means in simple, conversational language.

### 3. ✍️ Automated Legal Draft Generation (PDF)
Knowing your rights is only step one; taking action is step two.
*   If the AI detects your issue is a Consumer Complaint or Tenancy dispute, it automatically writes a formal **Legal Notice Draft**.
*   Users can download the securely formatted `Legal_Draft.pdf` immediately with one click.

### 4. 🤝 Free Legal Aid Dossier (NALSA Handoff)
Severe or highly complex cases inherently require human lawyers.
*   With a single click on **"Download Case Dossier"**, the entire chat history and the AI's preliminary findings are packaged into a professional text file. 
*   This can be forwarded directly to a free legal aid clinic (like NALSA - 15100).

### 5. ❓ Active Clarifying Questions
If a user just says "My landlord is troubling me," the AI won't spit out random, generic laws.
*   **Dynamic Follow-ups**: The Query Processor actively detects vague questions and dynamically asks to clarify missing facts (e.g., "In which state?", "Did you sign an agreement?").

### 6. 🛡️ Strict Compliance & Safety Guardrails
*   A dedicated **Compliance Agent** stands at the gate to completely block illegal advice.
*   **Emergency Banners**: Automatically detects violent/urgent queries and prominently flags National Emergency Helplines (Police, Women's Helpline, NALSA).

---

## 🛠️ Architecture

The system operates on an advanced "Orchestrator-Worker" Agent pattern:

1.  **Query Processor**: Detects intent, language, and asks clarifying questions for vague prompts.
2.  **Retrieval Agent**: FAISS + HuggingFace Embeddings (`all-MiniLM-L6-v2`) performs semantic search over our local corpus of major Indian Acts.
3.  **Reasoning Agent**: Synthesizes the retrieved chunks and user query using strict CoT (Chain of Thought), preventing hallucinations entirely.
4.  **Compliance Agent**: Evaluates the reasoned response against A2J safety guidelines.
5.  **Output Translator**: Smoothly localizes responses and generates Text-to-Speech (gTTS).

---

## ⚙️ Setup & Installation

### 1. Prerequisites
- Python 3.10+
- [Google AI Studio API Key](https://aistudio.google.com/) for Gemini 1.5 Flash

### 2. Clone & Install
```bash
git clone https://github.com/your-team/ai-legal-assistant.git
cd ai-legal-assistant

# Create virtual environment
python -m venv venv
# Windows: .\venv\Scripts\activate
# Mac/Linux: source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Environment Setup
Create a `.env` file in the root directory:
```bash
GOOGLE_API_KEY=your_gemini_api_key_here
```

### 4. Run the Interface
```bash
streamlit run ui/app.py
```

---

## 👥 Hackathon Team
- **Chhavi Gaba** · **Avani Garg** · **Vidushi Gupta** · **Akshay Haldar**

*Developed with ❤️ for ET AI Hackathon 2026. Empowering citizens through code.*
