import argparse
import json
import time
from typing import Callable

import requests

from fastapi.testclient import TestClient

try:
    # Importing app only required for in-process mode
    from hannibal_server import app
except Exception:
    app = None

tests = []

tests.append(
    {
        "name": "Exact-match quote (episode 1)",
        "path": "/v1/completions",
        "json": {
            "model": "Hannibal-RAG",
            "prompt": "Find and return the short quote that mentions 'vegetarian' from episode 1. If you can't find it, say 'NOT FOUND'.",
        },
    }
)

tests.append(
    {
        "name": "Case-insensitive match",
        "path": "/v1/completions",
        "json": {
            "model": "Hannibal-RAG",
            "prompt": "Search for the word 'vegetarian' (lowercase) in the retrieved context and return the sentence that contains it.",
        },
    }
)

tests.append(
    {
        "name": "Missing-term fallback",
        "path": "/v1/completions",
        "json": {
            "model": "Hannibal-RAG",
            "prompt": "Search for the word 'spaceship' in the context. If the word is not present in the retrieved context, respond exactly with NOT FOUND on the first line and then list the two scene headers returned by the retriever on subsequent lines.",
        },
    }
)

tests.append(
    {
        "name": "List scene headers",
        "path": "/v1/completions",
        "json": {
            "model": "Hannibal-RAG",
            "prompt": "From the retrieved context, list each scene header and its episode metadata as 'episode: scene_header' (one per line).",
        },
    }
)

results = []


def run_tests(client_post: Callable[[str, dict], dict], timeout=60):
    results_local = []
    for t in tests:
        start = time.time()
        try:
            resp = client_post(t["path"], t["json"])
            dur = time.time() - start
            results_local.append(
                {
                    "name": t["name"],
                    "status_code": resp.get("status_code", 500),
                    "duration": dur,
                    "response": resp.get("json", resp.get("text", {})),
                }
            )
        except Exception as e:
            results_local.append({"name": t["name"], "error": str(e)})
    return results_local


def inprocess_post(path: str, payload: dict):
    with TestClient(app) as client:
        r = client.post(path, json=payload, timeout=60)
        try:
            j = r.json()
        except Exception:
            j = {"text": r.text}
        return {"status_code": r.status_code, "json": j}


def http_post_factory(base_url: str):
    def post(path: str, payload: dict):
        url = base_url.rstrip("/") + path
        r = requests.post(url, json=payload, timeout=60)
        try:
            j = r.json()
        except Exception:
            j = {"text": r.text}
        return {"status_code": r.status_code, "json": j}

    return post


def main():
    parser = argparse.ArgumentParser(
        description="Run RAG tests in-process or over HTTP"
    )
    parser.add_argument("--mode", choices=("inprocess", "http"), default="inprocess")
    parser.add_argument(
        "--base-url", default="http://127.0.0.1:8001", help="Base URL for HTTP mode"
    )
    parser.add_argument(
        "--output", default="repro_rag_tests_output.json", help="Output JSON file"
    )
    args = parser.parse_args()

    if args.mode == "inprocess":
        if app is None:
            print("ERROR: app not importable; cannot run inprocess mode")
            raise SystemExit(3)
        client_post = inprocess_post
    else:
        client_post = http_post_factory(args.base_url)

    results = run_tests(client_post)

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    # --- Assertions / checks ---
    failures = []
    for res in results:
        name = res.get("name")
        if res.get("status_code") != 200:
            failures.append(f"{name}: HTTP {res.get('status_code')}")
            continue
        resp = res.get("response", {})
        text = ""
        if isinstance(resp, dict) and resp.get("choices"):
            try:
                # support both completion and chat shapes
                choice0 = resp["choices"][0]
                text = (
                    choice0.get("text")
                    or (choice0.get("message") or {}).get("content")
                    or ""
                )
            except Exception:
                text = str(resp)
        else:
            text = json.dumps(resp)

        tl = text.lower()
        if name == "Exact-match quote (episode 1)":
            if "vegetarian" not in tl:
                failures.append(f"{name}: expected 'vegetarian' in response")
        elif name == "Case-insensitive match":
            if "vegetarian" not in tl:
                failures.append(f"{name}: expected 'vegetarian' in response")
        elif name == "Missing-term fallback":
            absence_indicators = (
                "no mention",
                "not present",
                "not found",
                "find no",
                "can't find",
                "i find no",
            )
            # If the model echoes the search term that's acceptable only if it
            # also states that the term is not present. Otherwise require scene
            # headers or an explicit absence indicator.
            if "spaceship" in tl and not any(k in tl for k in absence_indicators):
                failures.append(
                    f"{name}: unexpected 'spaceship' found in response without absence indicator"
                )
            if not (
                "scene" in tl
                or "int." in tl
                or "ext." in tl
                or any(k in tl for k in absence_indicators)
            ):
                failures.append(
                    f"{name}: expected scene headers or absence fallback, got: {text[:120]!r}"
                )
        elif name == "List scene headers":
            if not ("episode" in tl or "int." in tl or "ext." in tl):
                failures.append(
                    f"{name}: expected scene headers or episode metadata, got: {text[:120]!r}"
                )

    if failures:
        print("TEST FAILURES:")
        for f in failures:
            print(" -", f)
        print(f"Full results saved to {args.output}")
        raise SystemExit(2)

    print(f"All checks passed. Results written to {args.output}")


if __name__ == "__main__":
    main()
