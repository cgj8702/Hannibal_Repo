import requests
import json

def test_streaming(port, endpoint="/v1/chat/completions"):
    url = f"http://localhost:{port}{endpoint}"
    payload = {
        "messages": [{"role": "user", "content": "Tell me a very short sentence about wine."}],
        "stream": True
    }
    
    print(f"Testing streaming on {url}...")
    try:
        response = requests.post(url, json=payload, stream=True, timeout=10)
        response.raise_for_status()
        
        for line in response.iter_lines():
            if line:
                decoded_line = line.decode('utf-8')
                if decoded_line.startswith("data: "):
                    content = decoded_line[6:]
                    if content == "[DONE]":
                        print("\n[STREAM COMPLETE]")
                        break
                    try:
                        data = json.loads(content)
                        delta = data['choices'][0].get('delta', {})
                        text = delta.get('content', '')
                        print(text, end="", flush=True)
                    except Exception as e:
                        print(f"\nError parsing JSON: {e}")
    except Exception as e:
        print(f"\nError: {e}")

if __name__ == "__main__":
    print("This script helps verify streaming. Make sure servers are running!")
    print("\n--- Testing Port 8002 (No-RAG) ---")
    test_streaming(8002)
    print("\n--- Testing Port 8001 (RAG) ---")
    test_streaming(8001)
