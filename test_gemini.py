#!/usr/bin/env python3
"""Quick test of Gemini API via BLADE's Gemini_LLM."""
import os, sys, traceback
sys.path.insert(0, "BLADE")
sys.path.insert(0, "LLaMEA")

from google import genai
print(f"google-genai: {genai.__version__}")

api_key = os.environ.get("GOOGLE_API_KEY")
if not api_key:
    print("ERROR: GOOGLE_API_KEY not set")
    sys.exit(1)

# 1. Direct SDK test
print("\n--- Direct SDK test ---")
client = genai.Client(api_key=api_key)
try:
    r = client.models.generate_content(model="gemini-3-flash-preview", contents="Say OK")
    print(f"OK: {r.text.strip()}")
except Exception as e:
    print(f"ERROR: {type(e).__name__}: {e}")

# 2. BLADE Gemini_LLM test
print("\n--- BLADE Gemini_LLM test ---")
from iohblade.llm import Gemini_LLM
llm = Gemini_LLM(api_key=api_key, model="gemini-3-flash-preview")
try:
    r = llm._query([{"role": "user", "content": "Say OK"}])
    print(f"OK: {r.strip()[:100]}")
except Exception as e:
    print(f"ERROR: {type(e).__name__}: {e}")
    traceback.print_exc()
