#!/usr/bin/env python3
"""Premature end of turn: a reply that quotes the template's own tokens in backticks stops there.

Usage: tools/premature_stop.py [--verbose] [--record FILE] [--extra JSON]
       (API_URL / PORT / MODEL as in client.py; greedy, one trial a case)

--record FILE writes every case as JSON: the exact request body, finish_reason, completion tokens, the reply's tail
and, where the server returns them (`"return_token_ids": true` on TensorFold 1.2+), the reply's last token ids.
--extra JSON merges request fields into every case, e.g. '{"return_token_ids": true}' or '{"ignore_eos": true}'.

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
import json
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


def trial(messages, tools, verbose, fields=None):
    """(stopped inside a span, reply) for one greedy trial."""
    extra = dict(fields or {})
    if tools:
        extra["tools"] = tools
    r = client.chat(messages, max_tokens=600, temperature=0.0, **extra)
    early = r.finish_reason == "stop" and not r.tool_calls and unfinished(r.content)
    if verbose or early:
        print(f"  finish={r.finish_reason} tokens={r.usage.get('completion_tokens')} "
              f"calls={len(r.tool_calls)} tail={r.content[-70:]!r}")
    return early, r


def reachable():
    """True when the endpoint answers, on an engine or on a gateway in front of one."""
    for path in ("/health", "/health/readiness", "/v1/models"):
        try:
            client.get_json(path, timeout=10)
            return True
        except Exception:                    # noqa: BLE001 - try the next spelling
            continue
    return False


def main() -> int:
    ap = argparse.ArgumentParser(description="Replies that end inside an open inline-code span.")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--record", metavar="FILE", help="write every case's request body and reply tail as JSON")
    ap.add_argument("--extra", metavar="JSON", default="{}", help="request fields merged into every case")
    args = ap.parse_args()
    fields = json.loads(args.extra)
    records = []
    if not reachable():
        print(f"could not reach the server at {client.API_URL}")
        return 2
    bad = 0
    for how in ("tools", "control"):
        stopped = []
        for ask in ASKS:
            for name, description, properties, required in SCHEMAS:
                tools = tools_for(description, properties, required) if how == "tools" else None
                messages = (HISTORY if tools else []) + [{"role": "user", "content": ask}]
                case = f"{name}/ask{ASKS.index(ask) + 1}"
                early, reply = trial(messages, tools, args.verbose, fields)
                if early:
                    stopped.append(case)
                if args.record:
                    extra = dict(fields, **({"tools": tools} if tools else {}))
                    ids = reply.tensorfold.get("token_ids") or []
                    records.append({
                        "check": how, "case": case, "stopped_in_span": early,
                        "request": client.chat_body(messages, 600, False, 0.0, False, **extra),
                        "finish_reason": reply.finish_reason,
                        "completion_tokens": reply.usage.get("completion_tokens"),
                        "content_tail": reply.content[-120:],
                        "last_token_ids": ids[-8:],
                        "token_sha": reply.token_sha,
                    })
        ok = not stopped
        print(f"{how}: {'PASS' if ok else 'FAIL'} - {len(stopped)}/{len(ASKS) * len(SCHEMAS)} replies stopped "
              f"inside a span" + (f" ({', '.join(stopped)})" if stopped else ""))
        bad += 0 if ok else 1
    if args.record:
        with open(args.record, "w") as f:
            json.dump({"server": client.API_URL, "model": client.model_id(), "extra": fields, "cases": records}, f, indent=1)
        print(f"recorded {len(records)} cases to {args.record}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
