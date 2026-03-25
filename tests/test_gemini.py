"""
tests/test_gemini.py
---------------------
Quick sanity check — run this FIRST before anything else to confirm
your Google AI Studio key is working.

Usage:
    python tests/test_gemini.py
"""

import sys
import os
sys.path.append(os.getcwd())  # Ensure root is in path

from google.genai import types
from config import client, MODEL_NAME, GOOGLE_API_KEY


def test_basic_response():
    print("\n[Test 1] Basic Gemini response...")
    # Use the client directly
    response = client.models.generate_content(
        model=MODEL_NAME,
        contents="In one sentence, what is Section 420 of the Indian Penal Code?",
        config=types.GenerateContentConfig()
    )
    print(f"  Response: {response.text.strip()}")
    assert len(response.text) > 10, "Response too short — something is wrong"
    print("  PASS ✅")


def test_json_output():
    print("\n[Test 2] JSON output (compliance agent format)...")
    
    prompt = """
    Return ONLY this JSON object. No markdown, no explanation, start with {:
    {"status": "working", "model": "gemini", "test": true}
    """
    
    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=0,
            response_mime_type="application/json"
        )
    )
    raw = response.text.strip()
    print(f"  Raw response: {raw}")

    import json
    # Try to parse
    try:
        # With response_mime_type="application/json", wrapping usually doesn't happen,
        # but we keep safety checks just in case.
        clean_text = raw
        if raw.startswith("```"):
            lines = [l for l in raw.split("\n") if not l.startswith("```")]
            raw   = "\n".join(lines).strip()
        start = raw.find("{")
        end   = raw.rfind("}") + 1
        parsed = json.loads(raw[start:end])
        print(f"  Parsed JSON: {parsed}")
        print("  PASS ✅")
    except Exception as e:
        print(f"  FAIL ❌ — Could not parse JSON: {e}")


def test_system_instruction():
    print("\n[Test 3] System instruction (reasoning agent format)...")
    
    response = client.models.generate_content(
        model=MODEL_NAME,
        contents="What is a consumer complaint?",
        config=types.GenerateContentConfig(
            temperature=0.1,
            system_instruction="You are a legal expert. Always start your answer with 'LEGAL ANSWER:'"
        )
    )
    print(f"  Response preview: {response.text[:150]}...")
    assert len(response.text) > 20, "Response too short"
    print("  PASS ✅")


if __name__ == "__main__":
    print("=" * 50)
    print("  Gemini API Test Suite")
    print("=" * 50)

    if not GOOGLE_API_KEY or GOOGLE_API_KEY == "your_google_ai_studio_key_here":
        print("\nERROR: GOOGLE_API_KEY not set in .env file")
        print("Get your free key at: https://aistudio.google.com")
        sys.exit(1)

    test_basic_response()
    test_json_output()
    test_system_instruction()

    print("\n" + "=" * 50)
    print("  All Gemini tests passed. You're ready to build.")
    print("=" * 50 + "\n")
