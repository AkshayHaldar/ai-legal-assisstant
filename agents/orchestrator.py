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

        # ── STEP 1: Query processing (Layer 2) ───────────────────────────────
        print("[Orchestrator] → QueryProcessor")
        processed = self.query_processor.process(query)
        search_query = processed["reformulated_query"]
        intent       = processed["intent"]

        # If query is vague, ask for more details instead of guessing
        if processed["is_vague"]:
            clarification = (
                f"Your question is a bit general. To give you accurate legal guidance, "
                f"could you share more details?\n\n"
                f"For example:\n"
                f"- What specifically happened?\n"
                f"- Which state are you in?\n"
                f"- What outcome are you looking for?\n\n"
                f"The more details you share, the better I can help."
                + DISCLAIMER
            )
            return {
                "answer": clarification, "sources": [],
                "audit_id": log_event(query, "query_processor", [], "vague query detected",
                                      False, "", clarification),
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
            return {
                "answer": no_result, "sources": [], "audit_id":
                log_event(query, "retrieval", [], "", False, "", no_result),
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

        # ── STEP 5: Multilingual translation ─────────────────────────────────
        if language == "hi" and comp["safe_to_deliver"]:
            raw_answer = self._translate_to_hindi(raw_answer)

        # ── STEP 6: Build final answer ────────────────────────────────────────
        # Human review flag — triggered for family/property/complex cases
        review_flag = intent in ["family", "property"] or len(chunks) < 2

        if not comp["safe_to_deliver"]:
            final_answer = (
                "This query requires a licensed lawyer.\n\n"
                f"Reason: {comp['reason']}\n\n"
                "Free help: NALSA helpline 15100" + DISCLAIMER
            )
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
            reasoning_chain=reasoning["reasoning_chain"],
            compliance_triggered=not comp["safe_to_deliver"],
            compliance_reason=comp["reason"],
            final_answer=final_answer,
            language=language,
        )

        return {
            "answer":            final_answer,
            "sources":           reasoning["chunks_used"],
            "audit_id":          audit_id,
            "safe":              comp["safe_to_deliver"],
            "emergency":         comp.get("emergency_flag", False),
            "compliance_detail": comp,
            "intent":            intent,
            "review_flag":       review_flag,   # ← Human review gate
        }

    def _translate_to_hindi(self, text: str) -> str:
        """Basic Gemini translation — replace with IndicTrans2 for production."""
        try:
            resp = client.models.generate_content(
                model=MODEL_NAME,
                contents=(
                    f"Translate this legal guidance to simple Hindi. "
                    f"Keep legal terms in English with Hindi explanation in brackets.\n\n{text}"
                )
            )
            return resp.text
        except Exception:
            return text  # fallback: return English if translation fails