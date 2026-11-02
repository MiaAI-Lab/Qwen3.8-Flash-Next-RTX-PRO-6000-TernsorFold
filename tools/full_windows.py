#!/usr/bin/env python3
"""Every stream near a full window at once: each of N prompts holds its own passphrase, and each reply must name its own. Prints the card's peak memory.

Usage: tools/full_windows.py [N] [TOKENS]      (default 3 and 230000; GPU picks the nvidia-smi index, default 0; API_URL / PORT / MODEL as in client.py)
"""
import json
import os
import subprocess
import sys
import threading
import time
import urllib.request

sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bench as B  # noqa: E402


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    size = int(sys.argv[2]) if len(sys.argv) > 2 else 230000
    gpu = os.environ.get("GPU", "0")
    peak, stop = [0], []

    def watch():
        while not stop:
            o = subprocess.run(["nvidia-smi", "-i", gpu, "--query-gpu=memory.used", "--format=csv,noheader,nounits"], capture_output=True, text=True).stdout.strip()
            if o.isdigit():
                peak[0] = max(peak[0], int(o))
            time.sleep(0.5)

    threading.Thread(target=watch, daemon=True).start()
    res = [None] * n

    def one(i):
        code = f"KESTREL-{7000 + i * 137}-{int(time.time()) % 100000}"
        text = B.prose(int(size / 0.8), int(time.time()) + i * 7919)
        half = len(text) // 2
        prompt = text[:half] + f"\n\nThe passphrase is {code}.\n\n" + text[half:] + "\n\nWhat is the passphrase given in the text above? Reply with it alone."
        body = {"model": os.environ.get("MODEL", "Qwen3.8-Flash-Next"), "messages": [{"role": "user", "content": prompt}], "max_tokens": 48, "temperature": 0, "chat_template_kwargs": {"enable_thinking": False}}
        t = time.time()
        with urllib.request.urlopen(urllib.request.Request(B.URL, json.dumps(body).encode(), {"Content-Type": "application/json"}), timeout=1800) as r:
            j = json.load(r)
        res[i] = (code, j["choices"][0]["message"].get("content") or "", j["usage"]["prompt_tokens"], time.time() - t)

    ts = [threading.Thread(target=one, args=(i,)) for i in range(n)]
    t0 = time.time()
    [t.start() for t in ts]
    [t.join() for t in ts]
    stop.append(1)
    ok = all(r and r[0] in r[1] for r in res)
    for r in res:
        print(f"{'ok  ' if r and r[0] in r[1] else 'FAIL'} {r[2] if r else '?':>7} tokens, {r[3] if r else 0:6.1f} s: {(r[1] if r else '')[:60]!r}")
    print(f"full_windows: {'PASS' if ok else 'FAIL'}: {n} prompts in {time.time() - t0:.0f} s, card peak {peak[0]} MiB")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
