from agents.retrieval      import RetrievalAgent
from agents.reasoning      import ReasoningAgent
from agents.compliance     import ComplianceAgent, DISCLAIMER, EMERGENCY_MSG
from agents.query_processor import QueryProcessor
from audit import log_event
from config import client, MODEL_NAME


class Orchestrator:
    def __init__(self):
        print("[Orchestrator] Initialising agents...")
        self.query_processor = QueryProcessor()
        self.retrieval       = RetrievalAgent()
        self.reasoning       = ReasoningAgent()
        self.compliance      = ComplianceAgent()
        print("[Orchestrator] All agents ready.\n")

    def run(self, query: str, language: str = "en") -> dict:

        # ── STEP 1: Query processing (Layer 2) ──────────────────────────────────
        print("[Orchestrator] → QueryProcessor")
        processed = self.query_processor.process(query)
        search_query = processed.get("reformulated_query", query)
        intent       = processed.get("intent", "general")

        # If query is vague, ask for more details using dynamically generated clarifying questions
        if processed.get("is_vague"):
            questions = processed.get("clarifying_questions", [])
            if questions:
                q_bullets = "\n".join([f"- {q}" for q in questions])
            else:
                q_bullets = "- Could you please provide a bit more context or details?"

            clarification = (
                "To give you the most accurate legal guidance, I need a little more context. "
                "Could you help me with these details?\n\n"
                f"{q_bullets}\n\n"
                "The more details you share, the better I can assist you."
                + DISCLAIMER
            )

            # Translate clarification if needed
            if language != "en":
                clarification = self._translate(clarification, language)

            return {
                "answer": clarification, "sources": [],
                "audit_id": log_event(query, "query_processor", [], "vague query detected",
                                      False, "", clarification, language=language),
                "safe": True, "emergency": False,
                "compliance_detail": {}, "intent": intent,
                "review_flag": False,
            }

        # ── STEP 2: Retrieve ──────────────────────────────────────────────────
        print(f"[Orchestrator] → RetrievalAgent (intent: {intent})")
        chunks = self.retrieval.retrieve(search_query)

        if not chunks:
            no_result = (
                "I could not find relevant Indian law for your query. "
                "Please add more details or consult a licensed advocate."
                + DISCLAIMER
            )
            if language != "en":
                no_result = self._translate(no_result, language)

            return {
                "answer": no_result, "sources": [], "audit_id":
                log_event(query, "retrieval", [], "", False, "", no_result, language=language),
                "safe": True, "emergency": False,
                "compliance_detail": {}, "intent": intent, "review_flag": False,
            }

        # ── STEP 3: Reason ────────────────────────────────────────────────────
        print("[Orchestrator] → ReasoningAgent")
        reasoning = self.reasoning.reason(search_query, chunks)
        raw_answer = reasoning["answer"]

        # ── STEP 4: Compliance ────────────────────────────────────────────────
        print("[Orchestrator] → ComplianceAgent")
        comp = self.compliance.check(query, raw_answer)

        # ── STEP 5: Multilingual translation ──────────────────────────────────
        if language != "en" and comp.get("safe_to_deliver", True):
            raw_answer = self._translate(raw_answer, language)

        # ── STEP 6: Build final answer ────────────────────────────────────────
        # Human review flag — triggered for family/property/complex cases
        review_flag = intent in ["family", "property"] or len(chunks) < 2

        if not comp.get("safe_to_deliver", True):
            final_answer = (
                "This query requires a licensed lawyer.\n\n"
                f"Reason: {comp.get('reason', 'Safety policy')}\n\n"
                "Free help: NALSA helpline 15100" + DISCLAIMER
            )
            if language != "en":
                final_answer = self._translate(final_answer, language)
        else:
            final_answer = raw_answer + DISCLAIMER
            if comp.get("emergency_flag"):
                final_answer = EMERGENCY_MSG + final_answer

        # ── STEP 7: Audit log ─────────────────────────────────────────────────
        audit_id = log_event(
            query=query,
            agent_name="orchestrator",
            retrieved_chunks=[{"act": c["act"], "relevance": c["relevance"],
                               "preview": c["content"][:120]} for c in chunks],
            reasoning_chain=reasoning.get("reasoning_chain", ""),
            compliance_triggered=not comp.get("safe_to_deliver", True),
            compliance_reason=comp.get("reason", ""),
            final_answer=final_answer,
            language=language,
        )

        return {
            "answer":            final_answer,
            "sources":           reasoning.get("chunks_used", []),
            "audit_id":          audit_id,
            "safe":              comp.get("safe_to_deliver", True),
            "emergency":         comp.get("emergency_flag", False),
            "compliance_detail": comp,
            "intent":            intent,
            "review_flag":       review_flag,   # ← Human review gate
        }

    def _translate(self, text: str, lang_code: str) -> str:
        """Basic Gemini translation — mapping language code."""
        try:
            prompt = (
                f"You are speaking to a common person in a conversational way. Translate the following legal guidance into the language code '{lang_code}'.\n"
                f"- Use EXTREMELY SIMPLE, everyday language.\n"
                f"- Do NOT use heavy, formal or complex vocabulary.\n"
                f"- Explain legal terms simply in the target language (keep the technical English term in brackets if helpful).\n"
                f"- Format the response clearly with simple headings and bullet points.\n\n"
                f"{text}"
            )
            resp = client.models.generate_content(
                model=MODEL_NAME,
                contents=prompt
            )
            return resp.text
        except Exception:
            return text  # fallback: return English if translation fails
