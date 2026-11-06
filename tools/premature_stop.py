#!/usr/bin/env python3
"""Premature end of turn: a reply that quotes the template's own tokens in backticks stops there.

Usage: tools/premature_stop.py [--verbose]      (API_URL / PORT / MODEL as in client.py; greedy, one trial a case)

The trap is the tokenizer, not the parser: `<|im_end|>` is the turn's end token, so a model that writes it as
text inside a sentence ends the turn instead. The server reports the tokens it drew (`finish_reason` stop, no
tool call), so nothing downstream can repair the reply. It bites hardest on a conversation *about* tool-call
syntax; the same request without `tools` is unaffected.

A reply's own rendering decides it, so one phrasing can come back clean on a build where another stops: the
cases vary the tool schema and the wording, and each runs once (greedy, so a build gives the same answer twice).

Checks, each printed PASS or FAIL:
  1. tools:    a conversation that already holds a tool call and its result, then a question that quotes the
               template's markers. FAIL lists every case that finished `stop` inside an unterminated span.
  2. control:  the same questions with no tools offered. Those must never stop mid-span.
Exit 0 all pass, 1 any fails, 2 could not run.
"""
import argparse
import sys

sys.dont_write_bytecode = True           # no tools/__pycache__ from importing client
import client  # noqa: E402

# a finished tool turn is what puts the model in token-quoting distance of the template's own markers
HISTORY = [
    {"role": "user", "content": "Ping example.com for me."},
    {"role": "assistant", "content": None, "tool_calls": [
        {"id": "call_1", "type": "function",
         "function": {"name": "ping", "arguments": "{\"host\":\"example.com\"}"}}]},
    {"role": "tool", "tool_call_id": "call_1", "content": "{\"ok\": true, \"rtt_ms\": 12}"},
]

ASKS = [
    "Explain engine patch 0052 in 5 bullets. You must quote these three strings exactly, in backticks, at least "
    "once each: `<tool_call>` `</tool_call>` `<|tool_call_begin|>`. Keep each bullet short.",
    "Explain engine patch 0052 in 4 bullets. Quote `<tool_call>`, `</tool_call>` and `<|im_end|>` in backticks, "
    "once each. Keep it short.",
]

SCHEMAS = [
    ("bare", "Ping a host.", {"host": {"type": "string"}}, []),
    ("required", "Ping a host.", {"host": {"type": "string"}}, ["host"]),
    ("described", "Ping a host and report its round-trip time.",
     {"host": {"type": "string", "description": "The host to ping."}}, ["host"]),
]


def tools_for(description, properties, required):
    parameters = {"type": "object", "properties": properties}
    if required:
        parameters["required"] = required
    return [{"type": "function", "function": {"name": "ping", "description": description, "parameters": parameters}}]


def unfinished(content: str) -> bool:
    """True when the reply ends inside an inline-code span or a half-written tag."""
    tail = content.rstrip()
    if not tail:
        return True
    return tail.endswith("<") or tail.count("`") % 2 == 1


def trial(messages, tools, verbose):
    """(stopped inside a span, reply) for one greedy trial."""
    extra = {"tools": tools} if tools else {}
    r = client.chat(messages, max_tokens=600, temperature=0.0, **extra)
    early = r.finish_reason == "stop" and not r.tool_calls and unfinished(r.content)
    if verbose or early:
        print(f"  finish={r.finish_reason} tokens={r.usage.get('completion_tokens')} "
              f"calls={len(r.tool_calls)} tail={r.content[-70:]!r}")
    return early, r


def main() -> int:
    ap = argparse.ArgumentParser(description="Replies that end inside an open inline-code span.")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()
    try:
        client.get_json("/health")
    except Exception as e:                       # noqa: BLE001 - any failure to reach it is "could not run"
        print(f"could not reach the server: {e}")
        return 2
    bad = 0
    for how in ("tools", "control"):
        stopped = []
        for ask in ASKS:
            for name, description, properties, required in SCHEMAS:
                tools = tools_for(description, properties, required) if how == "tools" else None
                messages = (HISTORY if tools else []) + [{"role": "user", "content": ask}]
                if trial(messages, tools, args.verbose)[0]:
                    stopped.append(f"{name}/ask{ASKS.index(ask) + 1}")
        ok = not stopped
        print(f"{how}: {'PASS' if ok else 'FAIL'} - {len(stopped)}/{len(ASKS) * len(SCHEMAS)} replies stopped "
              f"inside a span" + (f" ({', '.join(stopped)})" if stopped else ""))
        bad += 0 if ok else 1
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
