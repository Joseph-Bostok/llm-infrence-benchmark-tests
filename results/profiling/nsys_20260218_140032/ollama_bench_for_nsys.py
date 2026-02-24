import json
import sys
import time
import requests

model = sys.argv[1]
num_requests = int(sys.argv[2])
max_tokens = int(sys.argv[3])

prompts = [
    "Explain the concept of recursion in programming in 3 sentences.",
    "What are the main differences between TCP and UDP protocols?",
    "Describe the process of photosynthesis in simple terms.",
]

for i in range(num_requests):
    prompt = prompts[i % len(prompts)]
    t0 = time.monotonic()
    resp = requests.post("http://localhost:11434/api/generate", json={
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {"num_predict": max_tokens}
    })
    t1 = time.monotonic()
    data = resp.json()
    tokens = data.get("eval_count", 0)
    print(f"Request {i}: {tokens} tokens in {t1-t0:.2f}s ({tokens/(t1-t0):.1f} tok/s)")
