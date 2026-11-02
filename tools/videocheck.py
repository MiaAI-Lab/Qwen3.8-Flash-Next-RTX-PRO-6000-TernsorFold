#!/usr/bin/env python3
"""Video input on the running server: each container answers, the prompt has the processor's layout, a chat resumes past its video, the limits hold.

Usage: tools/videocheck.py        (API_URL / PORT / MODEL as in client.py; thinking off, greedy)
The clips in tools/fixtures are 4 seconds of a red ball moving left to right, the same frames in MP4 (H.264), WebM (VP9), MKV and MOV. Exit 0 pass, 1 a check failed.
"""
import base64
import json
import os
import sys
import urllib.error
import urllib.request

API = os.environ.get("API_URL", "http://127.0.0.1:" + os.environ.get("PORT", "8888")).rstrip("/")
MODEL = os.environ.get("MODEL", "Qwen3.8-Flash-Next")
HERE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures")
MIME = {"mp4": "video/mp4", "webm": "video/webm", "mkv": "video/x-matroska", "mov": "video/quicktime"}
QUESTION = "What colour is the moving object, and does it move to the left or to the right? One short sentence."
fails = 0


def post(path, body):
    req = urllib.request.Request(API + path, json.dumps(body).encode(), {"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=600) as r:
            return r.status, json.load(r)
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read() or b"{}")


def part(raw, mime="video/mp4"):
    return {"type": "video_url", "video_url": {"url": f"data:{mime};base64," + base64.b64encode(raw).decode()}}


def text(t):
    return {"type": "text", "text": t}


def chat(messages, tokens=60):
    return post("/v1/chat/completions", {"model": MODEL, "messages": messages, "max_tokens": tokens, "temperature": 0, "chat_template_kwargs": {"enable_thinking": False}})


def user(*parts):
    return {"role": "user", "content": list(parts)}


def check(name, ok, detail=""):
    global fails
    fails += not ok
    print(f"{'PASS' if ok else 'FAIL'} {name}{': ' + str(detail)[:150] if detail else ''}", flush=True)


def reply(r):
    return r["choices"][0]["message"].get("content") or ""


def main():
    clip = {k: open(os.path.join(HERE, "ball." + k), "rb").read() for k in MIME}
    for kind, mime in MIME.items():
        code, r = chat([user(part(clip[kind], mime), text(QUESTION))])
        low = reply(r).lower() if code == 200 else ""
        check(f"{kind} answers", "red" in low and "right" in low, reply(r) if code == 200 else r)

    code, r = post("/tokenize", {"messages": [user(part(clip["mp4"]), text("What moves?"))]})
    check("/tokenize takes a video", code == 200, r if code != 200 else "")
    if code == 200:
        code, d = post("/detokenize", {"tokens": r["tokens"]})
        s = d.get("prompt") or d.get("text") or json.dumps(d)
        check("each group has its time before its own markers", "seconds><|vision_start|><|video_pad|>" in s)
        check("the template's outer markers are gone", "<|vision_start|><0" not in s and "<|vision_end|><|vision_end|>" not in s)

    long = " ".join(f"Detail {i}: the clip shows a ball, a colour and motion." for i in range(40))
    first = [user(part(clip["mp4"]), text("What is shown?")), {"role": "assistant", "content": long}, user(text("Again, in three words."))]
    chat(first, 8)
    code, r = chat(first + [{"role": "assistant", "content": "A red ball."}, user(text("Once more."))], 8)
    usage = r.get("usage", {}) if code == 200 else {}
    cached = (usage.get("prompt_tokens_details") or {}).get("cached_tokens", 0)
    check("a later turn resumes past its video", code == 200 and cached > usage.get("prompt_tokens", 0) * 0.9, f"cached {cached} of {usage.get('prompt_tokens')}")

    code, r = chat([user(*[part(clip["mp4"])] * 4, text("How many clips?"))], 16)
    check("four videos are the limit, and answer", code == 200, r if code != 200 else reply(r))
    for name, messages, words in (
        ("five videos", [user(*[part(clip["mp4"])] * 5, text("hi"))], "at most 4"),
        ("over 16 MB", [user(part(clip["mp4"] + b"\0" * 17_000_000), text("hi"))], "byte limit"),
        ("bytes that are no video", [user(part(b"not a video at all" * 20), text("hi"))], "invalid or unsupported"),
        ("a video in an assistant message", [user(text("hi")), {"role": "assistant", "content": [part(clip["mp4"])]}, user(text("ok"))], "only in user messages"),
    ):
        code, r = chat(messages)
        message = r.get("error", {}).get("message", "")
        check(f"{name} is refused", code == 400 and words in message, message or r)
    print("videocheck:", "PASS" if not fails else f"{fails} FAILED")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
