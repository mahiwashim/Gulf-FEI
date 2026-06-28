"""Quick Groq connectivity test — isolates whether the LLM call is what hangs.

Run on the Mac:  python test_groq.py
"""
import os
import time
from dotenv import load_dotenv

load_dotenv()

key = os.getenv("GROQ_API_KEY")
model = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

print(f"GROQ_API_KEY present: {bool(key)}  (length={len(key) if key else 0})")
print(f"GROQ_MODEL: {model}")

if not key:
    raise SystemExit("❌ GROQ_API_KEY is missing from .env — add it and retry.")

from langchain_groq import ChatGroq
from pydantic import SecretStr

llm = ChatGroq(model=model, api_key=SecretStr(key), timeout=30, max_retries=1)

print("→ Calling Groq (max 30s)...")
t0 = time.time()
try:
    resp = llm.invoke("Reply with exactly: OK")
    print(f"✅ Groq responded in {time.time()-t0:.1f}s: {resp.content!r}")
except Exception as exc:
    print(f"❌ Groq call failed after {time.time()-t0:.1f}s: {type(exc).__name__}: {exc}")
