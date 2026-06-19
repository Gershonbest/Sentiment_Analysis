"""
test_client.py — smoke-test the running Ray Serve deployment
Run AFTER sentiment_serve.py is up.
"""

import requests

BASE = "http://localhost:8000"
# BASE = "https://lku8dloagkpd3v-8000.proxy.runpod.net"  # change if your deployment is on a different host/port

# ── health check ──────────────────────────────────────────────────────────────
r = requests.get(f"{BASE}/sentiment/health")
print("Health:", r.json())

# ── single text ───────────────────────────────────────────────────────────────
r = requests.post(f"{BASE}/sentiment/predict", json={"text": "Ray Serve is incredibly fast!"})
# print("\nSingle:", r.json())
print("\nSingle:")
for item in r.json()["results"]:
    print(f"  [{item['label']} {item['score']:.2%}] {item['text']}")

# ── batch ─────────────────────────────────────────────────────────────────────
texts = [
    "I absolutely love this product.",
    "The documentation is confusing and poorly written.",
    "It works, but barely meets expectations.",
]
r = requests.post(f"{BASE}/predict", json={"text": texts})
print("\nBatch:")
for item in r.json()["results"]:
    print(f"  [{item['label']} {item['score']:.2%}] {item['text']}")
