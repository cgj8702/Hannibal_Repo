import json
import time
from fastapi.testclient import TestClient

from hannibal_server import app

# Use TestClient as a context manager so FastAPI startup events run
tests = []

tests.append(
    {
        "name": "Completion - summary",
        "path": "/v1/completions",
        "json": {
            "model": "Hannibal-RAG",
            "prompt": "Summarize Hannibal season 1 episode 1 in two sentences.",
        },
    }
)

tests.append(
    {
        "name": "Chat - who is",
        "path": "/v1/chat/completions",
        "json": {
            "model": "Hannibal-RAG",
            "messages": [{"role": "user", "content": "Who is Hannibal Lecter?"}],
        },
    }
)

tests.append(
    {
        "name": "RAG-specific - quote check",
        "path": "/v1/completions",
        "json": {
            "model": "Hannibal-RAG",
            "prompt": "Find a short quote from episode 1 that mentions food or dining.",
        },
    }
)

results = []

with TestClient(app) as client:
    for t in tests:
        start = time.time()
        try:
            r = client.post(t["path"], json=t["json"], timeout=60)
            dur = time.time() - start
            try:
                data = r.json()
            except Exception:
                data = {"text": r.text}
            results.append(
                {
                    "name": t["name"],
                    "status_code": r.status_code,
                    "duration": dur,
                    "response": data,
                }
            )
        except Exception as e:
            results.append({"name": t["name"], "error": str(e)})

with open("repro_inprocess_output.json", "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

print("Done. Results written to repro_inprocess_output.json")
