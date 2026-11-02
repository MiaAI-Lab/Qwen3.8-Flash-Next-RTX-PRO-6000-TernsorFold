#!/usr/bin/env python3
"""Compare settings on the card: for each arm, restart the server with its settings, measure decode and prefill, and write one JSON line.

Usage: tools/sweep.py ARMS OUT      (run from anywhere; restarts the server through ./start.sh restart, so it takes the server down between arms)
ARMS: one arm a line, its name then KEY=value settings ("base" alone is the defaults); # starts a comment.
OUT: results are appended, and an arm already in OUT is skipped, so a stopped sweep can go on.
Each arm: greedy prose and code at 1 and 3 streams, sampled prose at 1 and 3 (fixed seeds, so every arm decodes the same tokens), fresh prompts of about 16k and 65k tokens,
and decode at the end of a 65k prompt. Before each restart it waits while a client other than this machine has sent a request in the last QUIET seconds (default 600).
"""
import json
import os
import re
import statistics
import subprocess
import sys
import threading
import time
import urllib.request

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import bench as B  # noqa: E402

ROOT = os.path.dirname(HERE)
CONTAINER = os.environ.get("CONTAINER_NAME", "qwen38-flash-next-tf")
QUIET = int(os.environ.get("QUIET", "600"))
PROSE = "Explain step by step how a hash map works, with the reasoning behind each design choice."
CODE = "Write a Python LRU cache class with type hints. Output only Python source. No comments, no docstrings, no markdown fences. Then add tests and the helpers this needs. Keep writing code."


def client_busy():
    """Whether a client other than this machine posted a request in the last QUIET seconds."""
    o = subprocess.run(["docker", "logs", "--since", f"{QUIET}s", CONTAINER], capture_output=True, text=True, errors="replace")
    return any(re.search(r'\] (?!127\.0\.0\.1)\S+ "POST /v1/', line) for line in (o.stdout + o.stderr).splitlines())


def one(prompt, tokens, temperature, seed, thinking):
    body = {"model": os.environ.get("MODEL", "Qwen3.8-Flash-Next"), "stream": True, "max_tokens": tokens, "stream_options": {"include_usage": True},
            "messages": [{"role": "user", "content": prompt}], "temperature": temperature, "seed": seed, "chat_template_kwargs": {"enable_thinking": thinking}}
    req = urllib.request.Request(B.URL, json.dumps(body).encode(), {"Content-Type": "application/json"})
    start, first, tf, usage = time.time(), None, {}, {}
    for line in urllib.request.urlopen(req, timeout=1800):
        line = line.decode().strip()
        if not line.startswith("data:") or line.endswith("[DONE]"):
            continue
        chunk = json.loads(line[5:])
        delta = (chunk.get("choices") or [{}])[0].get("delta", {})
        if first is None and (delta.get("content") or delta.get("reasoning_content")):
            first = time.time()
        tf = chunk.get("tensorfold") or tf
        usage = chunk.get("usage") or usage
    return B.normalize(tf, usage, (first or time.time()) - start)


def decode(prompt, streams, tokens, temperature, thinking, reps=3):
    totals = []
    for k in range(reps):
        out = [None] * streams

        def run(i, k=k):
            # a different opening a stream, so concurrent streams are not one prompt; the same one in every arm
            out[i] = one(f"(Task {i + 1}.) " + prompt, tokens, temperature, 1000 + i, thinking)

        ts = [threading.Thread(target=run, args=(i,)) for i in range(streams)]
        [t.start() for t in ts]
        [t.join() for t in ts]
        totals.append(sum(r["decode_tps"] or 0 for r in out))
    return round(statistics.median(totals), 1)


def prefill(tokens, reps, seed):
    rates = []
    for k in range(reps):
        r = B.run(B.prose(tokens, seed + tokens + k * 7919) + "\n\nSummarize the text above in one sentence.", 16)
        rates.append(B.prefilled(r) / r["prefill_s"])
    return round(statistics.median(rates))


def measure():
    out = {}
    # warm the kernels and the n-gram rows these prompts use
    for prompt in (PROSE, CODE):
        one(prompt, 32, 0, 1, False)
    out["greedy_prose_1"] = decode(PROSE, 1, 400, 0, False)
    out["greedy_prose_3"] = decode(PROSE, 3, 400, 0, False)
    out["greedy_code_1"] = decode(CODE, 1, 400, 0, False)
    out["greedy_code_3"] = decode(CODE, 3, 400, 0, False)
    out["sampled_prose_1"] = decode(PROSE, 1, 400, 0.7, True)
    out["sampled_prose_3"] = decode(PROSE, 3, 400, 0.7, True)
    seed = int(time.time())
    out["prefill_16k"] = prefill(20000, 3, seed)
    out["prefill_65k"] = prefill(80000, 2, seed)
    long = B.prose(80000, seed + 5) + "\n\nWrite a long story that uses these words."
    B.run(long, 8)
    r = B.run(long, 300, 0)
    out["decode_at_65k"] = round(r["decode_tps"] or 0, 1)
    return out


def main():
    arms_path, out_path = sys.argv[1], sys.argv[2]
    done = set()
    if os.path.exists(out_path):
        done = {json.loads(line)["arm"] for line in open(out_path) if line.strip()}
    for line in open(arms_path):
        line = line.split("#", 1)[0].strip()
        if not line:
            continue
        name, *settings = line.split()
        if name in done:
            continue
        while client_busy():
            print(f"{name}: a client is using the server; waiting", flush=True)
            time.sleep(60)
        env = {**os.environ, **dict(s.split("=", 1) for s in settings)}
        t0 = time.time()
        r = subprocess.run([os.path.join(ROOT, "start.sh"), "restart"], env=env, capture_output=True, text=True, errors="replace")
        rec = {"arm": name, "settings": settings, "start_rc": r.returncode, "load_s": round(time.time() - t0)}
        if r.returncode == 0:
            try:
                rec.update(measure())
            except Exception as e:
                rec["error"] = repr(e)[:300]
        else:
            rec["error"] = (r.stdout + r.stderr)[-300:]
        with open(out_path, "a") as f:
            f.write(json.dumps(rec) + "\n")
        print(json.dumps(rec), flush=True)


if __name__ == "__main__":
    main()
