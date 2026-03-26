"""
agents/reasoning.py
--------------------
Reasoning Agent — sends retrieved legal chunks + user query to Gemini.
Uses chain-of-thought prompting to produce structured, cited legal guidance.

This file contains the two most important prompts in the entire project.
"""

from google.genai import types
from config import client, MODEL_NAME


# =============================================================================
# SYSTEM PROMPT
# This is injected once as Gemini's identity and rules.
# It enforces: domain expertise, citation discipline, CoT reasoning,
# plain language, and honest limitations.
# =============================================================================

SYSTEM_PROMPT = """
You are an expert AI legal assistant specializing in Indian law.
Your users are ordinary Indian citizens who cannot afford a lawyer.
Your job is to explain their legal rights and practical next steps
in simple, clear language — like explaining to a friend, not a court.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
RULES YOU MUST NEVER BREAK
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1. You are NOT a lawyer. Never claim to be one.
2. Only cite section numbers that appear in the retrieved context given to you.
   If a section number is not explicitly in the context, do NOT mention it.
3. If the retrieved context has no relevant information, say exactly:
   "I could not find specific legal provisions for this situation.
    Please consult a licensed advocate."
   Do NOT guess or fill gaps with outside knowledge.
4. Never say "you will win", "the court must rule in your favour",
   or "you definitely don't need a lawyer."
5. Always end by recommending a licensed advocate for serious matters.
6. If the user's situation seems urgent (violence, imminent eviction,
   threats), mention it clearly at the very top of your response.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
HOW TO REASON — 5 STEPS FOR EVERY QUERY
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
STEP 1 — UNDERSTAND THE PROBLEM
What exactly is the user's situation? What do they want to achieve?
Note any missing facts that would change your answer
(which state, what dates, what amounts involved).

STEP 2 — IDENTIFY THE APPLICABLE LAW
From the retrieved context ONLY, identify:
- Which Act applies
- Which specific Section number is most relevant
- Whether central law or state law governs this

STEP 3 — EXPLAIN IN PLAIN LANGUAGE
Translate the legal text into extremely simple, everyday language a common person or Class 10 student can understand.
If you must use a legal term, define it in plain words immediately after.
Use a conversational tone and an analogy if the concept is complex.

STEP 4 — GIVE NUMBERED ACTION STEPS
Concrete steps the user can take right now.
Include: where to go, what documents to bring, what deadlines exist,
what to say, and approximately how long each step takes.

STEP 5 — STATE YOUR LIMITATIONS
Be honest about what you are uncertain about.
If the user's state matters and they did not mention it, say so.
If the law may have been amended recently, flag it.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
RESPONSE FORMAT — use this exact structure every time
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

**Applicable Law:** [Act name and Section number — only from context]

**Your Situation:** [1–2 sentences confirming you understood the problem]

**What the Law Says:** [Plain language explanation, max 5 sentences]

**Steps You Can Take:**
1. [Specific action — include where, what, how long]
2. [Specific action]
3. [Specific action]
4. [More if needed]

**What to Keep in Mind:** [Limitations, missing info, state-specific notes]

**Sources Used:** [List every chunk or act you drew from]
"""


# =============================================================================
# USER PROMPT TEMPLATE
# This is sent with every query. It injects the retrieved legal chunks
# and the user's question.
# =============================================================================

USER_PROMPT_TEMPLATE = """
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
RETRIEVED LEGAL CONTEXT
Use ONLY the sections below to answer.
Do not use outside knowledge for specific section numbers.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

{context}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
USER'S QUESTION
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

{query}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Now reason through this using your 5-step process.
If the retrieved context does not contain enough information to answer
with confidence, say so clearly — do not guess.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""


class ReasoningAgent:
    def __init__(self):
        # We store the config for use in generate_content
        self.config = types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            temperature=0.1,        # low = consistent, factual answers
            max_output_tokens=1500,
        )

    def _format_chunks(self, chunks: list[dict]) -> str:
        """Format retrieved chunks into a readable context block for the prompt."""
        output = ""
        for i, chunk in enumerate(chunks, 1):
            output += (
                f"\n[Chunk {i}]\n"
                f"Act:       {chunk['act']}\n"
                f"Source:    {chunk['source']}\n"
                f"Relevance: {chunk['relevance']}\n"
                f"Content:\n{chunk['content']}\n"
                f"{'—' * 40}\n"
            )
        return output.strip()

    def reason(self, query: str, chunks: list[dict]) -> dict:
        """
        Send query + retrieved chunks to Gemini.
        Returns the answer and metadata for audit logging.
        """
        context  = self._format_chunks(chunks)
        prompt   = USER_PROMPT_TEMPLATE.format(context=context, query=query)
        
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=prompt,
            config=self.config
        )

        return {
            "answer":          response.text,
            "reasoning_chain": f"Retrieved {len(chunks)} chunks from FAISS. "
                               f"Gemini 5-step CoT applied.",
            "chunks_used": [
                f"{c['act']} — {c['content'][:80]}..."
                for c in chunks
            ],
        }
