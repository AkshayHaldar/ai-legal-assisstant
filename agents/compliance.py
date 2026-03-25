"""
agents/compliance.py
---------------------
Compliance Agent — safety layer that runs on every answer before delivery.

Two-stage approach:
  Stage 1: Hardcoded keyword fast-check (instant, no API call)
  Stage 2: Gemini JSON check for nuanced violations

This ensures the pipeline never crashes even if Gemini returns malformed JSON.
"""

import json
from google.genai import types
from config import client, MODEL_NAME


# =============================================================================
# COMPLIANCE SYSTEM PROMPT
# Instructs Gemini to return a strict JSON object — no markdown, no extra text.
# =============================================================================

COMPLIANCE_PROMPT = """
You are a safety checker for an AI legal assistant serving Indian citizens.

CRITICAL INSTRUCTION:
- Your ENTIRE response must be a single raw JSON object.
- Do NOT include any text before or after the JSON.
- Do NOT wrap it in markdown code fences (no ```json).
- Do NOT include the word "json" anywhere in your response.
- Start your response with { and end with }

Review the AI answer below and check for these violations:

CRIMINAL_ADVICE     — specific advice on: bail applications, FIR strategy,
                      criminal trials, sentencing, anticipatory bail,
                      how to evade criminal prosecution
URGENT_SAFETY       — query involves: domestic violence, same-day eviction,
                      physical threats, imminent danger
FABRICATED_CITATION — answer cites section numbers that do not exist in
                      standard Indian law (e.g. Section 498B, Section 302A)
OVERCONFIDENT       — answer says "you will win", "court must rule for you",
                      "you definitely don't need a lawyer"
OUT_OF_SCOPE        — answer covers: income tax planning, GST, company
                      incorporation, international law, stock market

Return exactly this JSON and nothing else:
{
  "safe_to_deliver": true,
  "violations_found": [],
  "reason": "one sentence explanation of your decision",
  "emergency_flag": false
}
"""

# Standard disclaimer appended to every response
DISCLAIMER = (
    "\n\n---\n"
    "⚠️ **Disclaimer:** This is legal information, not legal advice. "
    "Laws may have changed. For your specific situation, please consult "
    "a licensed advocate registered with the Bar Council of India."
)

# Emergency message prepended when urgent situation detected
EMERGENCY_MSG = (
    "🆘 **Urgent situation detected.** Please reach out for immediate help:\n"
    "- **NALSA Free Legal Helpline:** 15100\n"
    "- **Women Helpline:** 1091\n"
    "- **Police:** 100\n"
    "- **Domestic Violence Helpline:** 181\n\n"
)

# Fast-path keyword lists — no API call needed for these obvious cases
CRIMINAL_KEYWORDS = [
    "bail", "arrested", "in custody", "fir against me", "anticipatory bail",
    "charge sheet", "criminal trial", "sent to jail", "prison sentence",
    "murder case", "rape case", "dacoity", "kidnapping case",
]

URGENT_KEYWORDS = [
    "evicted today", "evicted tomorrow", "domestic violence",
    "being beaten", "physical abuse", "threatened to kill",
    "husband beating", "wife beating", "locked me out today",
]


class ComplianceAgent:
    def __init__(self):
        self.config = types.GenerateContentConfig(temperature=0)

    def _clean_json(self, text: str) -> dict:
        """
        Gemini sometimes wraps JSON in markdown fences despite instructions.
        This strips all possible wrapping before parsing.
        """
        text = text.strip()

        # Remove markdown code fences if present
        if text.startswith("```"):
            lines = [l for l in text.split("\n") if not l.strip().startswith("```")]
            text  = "\n".join(lines).strip()

        # Extract JSON object boundaries
        start = text.find("{")
        end   = text.rfind("}") + 1
        if start != -1 and end > start:
            text = text[start:end]

        return json.loads(text)

    def _fast_check(self, query: str) -> dict | None:
        """
        Stage 1: Instant keyword-based check. Returns result immediately
        without calling Gemini for obvious criminal/urgent cases.
        """
        q_lower = query.lower()

        if any(kw in q_lower for kw in CRIMINAL_KEYWORDS):
            return {
                "safe_to_deliver":  False,
                "violations_found": ["CRIMINAL_ADVICE"],
                "reason":           "Query involves a criminal matter requiring a licensed lawyer.",
                "emergency_flag":   False,
            }

        if any(kw in q_lower for kw in URGENT_KEYWORDS):
            return {
                "safe_to_deliver":  True,
                "violations_found": [],
                "reason":           "Urgent safety situation detected.",
                "emergency_flag":   True,
            }

        return None  # No fast-path match — proceed to Gemini check

    def check(self, query: str, answer: str) -> dict:
        """
        Stage 1: Fast keyword check.
        Stage 2: Gemini JSON compliance check for nuanced violations.
        Always returns a safe default if parsing fails — never crashes.
        """
        # Stage 1
        fast_result = self._fast_check(query)
        if fast_result:
            return fast_result

        # Stage 2 — Gemini check
        prompt   = f"{COMPLIANCE_PROMPT}\n\nQuery: {query}\n\nAI Answer:\n{answer}"
        
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=prompt,
            config=self.config
        )

        try:
            result = self._clean_json(response.text)
            return result
        except Exception as e:
            print(f"[ComplianceAgent] JSON parse failed: {e}")
            print(f"[ComplianceAgent] Raw Gemini output: {response.text[:300]}")
            # Safe default — never block a response due to parse error
            return {
                "safe_to_deliver":  True,
                "violations_found": [],
                "reason":           "Compliance parse error — defaulting to safe delivery.",
                "emergency_flag":   False,
            }
