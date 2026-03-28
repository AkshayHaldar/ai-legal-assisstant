"""
agents/query_processor.py
--------------------------
Query Processing Agent — Layer 2 of the architecture.
Runs before the orchestrator to:
  1. Detect the language of the query
  2. Classify the intent (consumer, tenancy, labour, criminal, general)
  3. Reformulate vague queries into specific, searchable ones
"""

from google.genai import types
from config import client, MODEL_NAME

QUERY_PROCESSOR_PROMPT = """
You are a query processing agent for an Indian legal assistant.
Given a user's raw query, return ONLY a JSON object — no markdown, no extra text.

Your tasks:
1. Detect language: "en", "hi", "ta", "te", "bn", or "mixed"
2. Classify intent into one of:
   - "consumer"     → consumer rights, product complaints, refunds
   - "tenancy"      → landlord, rent, eviction, deposit
   - "labour"       → salary, employer, workplace, termination
   - "criminal"     → FIR, police, arrest, bail (needs lawyer flag)
   - "family"       → divorce, custody, marriage, inheritance
   - "property"     → land, ownership, registration, dispute
   - "cyber"        → cyber attacks, data privacy, it act, online frauds
   - "general"      → does not fit above categories
3. Assess if the query is too vague to answer (missing key facts like dates, specific actions, state, relationship).
4. Reformulate into a clear, specific, searchable version.
5. If vague, provide an array of 2-3 specific "clarifying_questions" to ask the user to get the missing facts.

Return exactly this JSON:
{
  "detected_language": "en",
  "intent": "consumer",
  "is_vague": false,
  "vague_reason": "",
  "clarifying_questions": [],
  "reformulated_query": "the improved version of the query",
  "needs_lawyer": false
}
"""

# Fast-path intent detection by keywords (no API call needed)
INTENT_KEYWORDS = {
    "consumer":  ["consumer", "product", "defective", "refund", "complaint",
                  "warranty", "online order", "ecommerce", "seller"],
    "tenancy":   ["landlord", "tenant", "rent", "eviction", "deposit",
                  "lease", "flat", "house rent", "notice to vacate"],
    "labour":    ["salary", "employer", "fired", "terminated", "workplace",
                  "employee", "wages", "payslip", "pf", "gratuity"],
    "criminal":  ["fir", "police", "arrested", "bail", "custody", "crime",
                  "complaint against", "murder", "assault"],
    "family":    ["divorce", "custody", "marriage", "husband", "wife",
                  "inheritance", "will", "property dispute family"],
    "property":  ["land", "plot", "registration", "property", "ownership",
                  "mutation", "patta", "encroachment"],
    "cyber":     ["cyber", "data breach", "privacy", "gdpr", "hacking",
                  "phishing", "online fraud", "ransomware", "it act"],}
class QueryProcessor:
    def __init__(self):
        self.config = types.GenerateContentConfig(temperature=0)

    def _fast_intent(self, query: str) -> str:
        """Keyword-based intent detection — instant, no API call."""
        q = query.lower()
        for intent, keywords in INTENT_KEYWORDS.items():
            if any(kw in q for kw in keywords):
                return intent
        return "general"

    def _clean_json(self, text: str) -> dict:
        import json
        text = text.strip()
        if text.startswith("```"):
            lines = [l for l in text.split("\n") if not l.strip().startswith("```")]
            text  = "\n".join(lines).strip()
        start = text.find("{")
        end   = text.rfind("}") + 1
        return json.loads(text[start:end])

    def process(self, query: str) -> dict:
        """
        Process the raw user query.
        Returns enriched query info used by the orchestrator.
        """
        # Fast path first
        fast_intent = self._fast_intent(query)

        # For very short or clearly clear queries, skip the LLM call
        if len(query.split()) >= 4 and fast_intent != "general":
            return {
                "original_query":     query,
                "detected_language":  "en",
                "intent":             fast_intent,
                "is_vague":           False,
                "vague_reason":       "",
                "clarifying_questions": [],
                "reformulated_query": query,
                "needs_lawyer":       fast_intent == "criminal",
            }

        # Full LLM processing for ambiguous/short queries
        try:
            prompt = f"{QUERY_PROCESSOR_PROMPT}\n\nUser query: {query}"

            response = client.models.generate_content(
                model=MODEL_NAME,
                contents=prompt,
                config=self.config
            )

            result   = self._clean_json(response.text)
            result["original_query"] = query
            return result
        except Exception as e:
            print(f"[QueryProcessor] LLM parse failed: {e}. Using fast-path.")
            return {
                "original_query":     query,
                "detected_language":  "en",
                "intent":             fast_intent,
                "is_vague":           len(query.split()) < 5,
                "vague_reason":       "Query too short" if len(query.split()) < 5 else "",
                "clarifying_questions": ["Could you provide a few more details about your situation?"] if len(query.split()) < 5 else [],
                "reformulated_query": query,
                "needs_lawyer":       fast_intent == "criminal",
            }
