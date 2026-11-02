#!/usr/bin/env python3
"""Replies resumed from the prompt cache must equal the cold reply byte for byte, wherever the state comes back from.

Usage: tools/resume_exact.py [sizes]      (default 2000,30000,90000 words; API_URL / PORT / MODEL as in client.py; thinking off, greedy)
Each size: a fresh prompt cold, then the same prompt resumed alone, resumed after three other prompts took the streams (so its state comes back from the cache),
resumed beside two running requests, and a follow-up turn against that same turn resumed. Exit 0 pass, 1 a reply differs.
"""
import json
import os
import random
import sys
import threading
import time
import urllib.request

API = os.environ.get("API_URL", "http://127.0.0.1:" + os.environ.get("PORT", "8888")).rstrip("/")
MODEL = os.environ.get("MODEL", "Qwen3.8-Flash-Next")
WORDS = "river stone cedar lantern copper meadow harbor quiet orbit velvet thunder saddle marble pepper willow anchor".split()
fails = 0


def prose(n, seed):
    r = random.Random(seed)
    return " ".join(r.choice(WORDS) for _ in range(n))


def chat(messages, tokens=200):
    body = {"model": MODEL, "messages": messages, "max_tokens": tokens, "temperature": 0, "chat_template_kwargs": {"enable_thinking": False}}
    req = urllib.request.Request(API + "/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=900) as f:
        o = json.load(f)
    u = o["usage"]
    return o["choices"][0]["message"].get("content") or "", u["prompt_tokens"], (u.get("prompt_tokens_details") or {}).get("cached_tokens", 0)


def same(name, ref, got):
    global fails
    ok = ref[0] == got[0] and got[2] > got[1] * 0.9
    fails += not ok
    print(f"{'PASS' if ok else 'FAIL'} {name}: prompt {got[1]}, cached {got[2]} ({100 * got[2] / got[1]:.1f}%), {len(got[0])} chars" + ("" if ref[0] == got[0] else f"\n  cold    {ref[0][:160]!r}\n  resumed {got[0][:160]!r}"), flush=True)


def beside(prompts, tokens):
    ts = [threading.Thread(target=chat, args=([{"role": "user", "content": p}], tokens)) for p in prompts]
    [t.start() for t in ts]
    return ts


def main():
    sizes = [int(x) for x in (sys.argv[1] if len(sys.argv) > 1 else "2000,30000,90000").split(",")]
    for size in sizes:
        seed = int(time.time()) + size
        m = [{"role": "user", "content": prose(size, seed) + "\n\nList eight of the words above in a numbered list, then write a short poem using them."}]
        cold = chat(m)
        print(f"     {size} words: cold prompt {cold[1]}, cached {cold[2]}", flush=True)
        if cold[2] >= 100:
            sys.exit(f"the prompt was not cold: {cold[2]} tokens cached")
        same(f"{size}: resumed alone", cold, chat(m))
        [t.join() for t in beside([prose(6000, seed + 1 + i) + "\n\nOne word." for i in range(3)], 8)]
        same(f"{size}: resumed after three other prompts", cold, chat(m))
        ts = beside([prose(4000, seed + 9 + i) + "\n\nWrite a limerick." for i in range(2)], 120)
        time.sleep(0.1)
        got = chat(m)
        [t.join() for t in ts]
        same(f"{size}: resumed beside two running requests", cold, got)
        m2 = m + [{"role": "assistant", "content": cold[0]}, {"role": "user", "content": "Now write it again, reversed line by line."}]
        a = chat(m2)
        same(f"{size}: a follow-up turn resumed", a, chat(m2))
    print("resume_exact:", "PASS" if not fails else f"{fails} FAILED")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
