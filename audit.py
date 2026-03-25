import json
import uuid
import os
from datetime import datetime
from config import AUDIT_LOG_PATH


def log_event(
    query: str,
    agent_name: str,
    retrieved_chunks: list = None,
    reasoning_chain: str  = "",
    compliance_triggered: bool = False,
    compliance_reason: str     = "",
    final_answer: str          = "",
    language: str              = "en",
) -> str:
    os.makedirs("logs", exist_ok=True)

    entry = {
        "event_id":             str(uuid.uuid4()),
        "timestamp":            datetime.utcnow().isoformat() + "Z",
        "query":                query,
        "agent_called":         agent_name,
        "retrieved_chunks":     retrieved_chunks or [],
        "reasoning_chain":      reasoning_chain,
        "compliance_triggered": compliance_triggered,
        "compliance_reason":    compliance_reason,
        "final_answer_preview": final_answer[:300],
        "language":             language,
    }

    with open(AUDIT_LOG_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    return entry["event_id"]
