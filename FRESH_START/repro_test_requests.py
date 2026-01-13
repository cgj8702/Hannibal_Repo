import json
import sys
import time
import requests

BASE = "http://127.0.0.1:8001"

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

out = []
for t in tests:
    url = BASE + t["path"]
    try:
        start = time.time()
        r = requests.post(url, json=t["json"], timeout=30)
        duration = time.time() - start
        try:
            data = r.json()
        except Exception:
            data = {"text": r.text}
        out.append(
            {
                "name": t["name"],
                "status_code": r.status_code,
                "duration": duration,
                "response": data,
            }
        )
    except Exception as e:
        out.append({"name": t["name"], "error": str(e)})

with open("repro_requests_output.json", "w", encoding="utf-8") as f:
    json.dump(out, f, indent=2, ensure_ascii=False)

print("Done. Results written to repro_requests_output.json")
