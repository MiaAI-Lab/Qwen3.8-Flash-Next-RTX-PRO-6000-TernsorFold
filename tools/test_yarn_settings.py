#!/usr/bin/env python3
"""YARN=1's settings: what scripts/config.sh gives each way of asking for a long window, and that it fits the card.

Runs scripts/config.sh with no server, no GPU and no Docker. Checks:
  - by default nothing changes: 262,144 tokens, 3 streams, a 3 GiB reserve, a 16 GiB prompt cache;
  - YARN=1 is one stream at 1,000,000 tokens, the engine's smallest reserve, and a prompt cache that holds that state whole;
  - a CONTEXT past 262,144 asks for the same on its own, and explicit settings win;
  - YARN=1 with a CONTEXT that is not past 262,144 stops, as does any other YARN value;
  - the start gate passes on an empty card at 1,000,000 tokens and stops a second long stream.

Usage: tools/test_yarn_settings.py
"""
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
# What an empty RTX PRO 6000 reports free on this machine (docs/details.md, KV and memory: 95.04 GiB).
EMPTY_CARD_MIB = 97321
NAMES = ("YARN", "CONTEXT", "PARALLEL", "TENSORFOLD_MEMORY_RESERVE_GIB", "PROMPT_CACHE_GIB")


def settings(**env):
    """Run config.sh with only `env` set of the names under test; return (rc, values or stderr)."""
    clean = {k: v for k, v in os.environ.items() if k not in NAMES and k not in ("AZ", "ABLIT", "CHECKPOINT")}
    clean.update(env)
    show = " ".join(f'"${n}"' for n in NAMES)
    p = subprocess.run(["bash", "-c", f'source scripts/config.sh; echo {show} "$(gpu_need_mib)"'], cwd=ROOT,
                       env=clean, capture_output=True, text=True)
    if p.returncode != 0:
        return p.returncode, p.stderr.strip()
    yarn, context, parallel, reserve, cache, need = p.stdout.split()
    return 0, {"YARN": yarn, "CONTEXT": int(context), "PARALLEL": int(parallel), "RESERVE": reserve,
               "CACHE": cache, "NEED": int(need)}


def main() -> int:
    if (ROOT / ".env").exists():
        print("note: .env is read too, and may change what this test sees")
    bad = []

    def check(name, got, want):
        if got != want:
            bad.append(f"{name}: got {got!r}, want {want!r}")

    rc, v = settings()
    check("default rc", rc, 0)
    check("default", (v["YARN"], v["CONTEXT"], v["PARALLEL"], v["RESERVE"], v["CACHE"]), ("0", 262144, 3, "3", "16"))

    rc, v = settings(YARN="1")
    check("YARN=1 rc", rc, 0)
    check("YARN=1", (v["CONTEXT"], v["PARALLEL"], v["RESERVE"], v["CACHE"]), (1000000, 1, "0.5", "20"))
    if v["NEED"] > EMPTY_CARD_MIB:
        bad.append(f"YARN=1 needs {v['NEED']} MiB free, and an empty card has {EMPTY_CARD_MIB}")

    rc, v = settings(CONTEXT="500000")
    check("CONTEXT=500000 rc", rc, 0)
    check("CONTEXT=500000", (v["YARN"], v["CONTEXT"], v["PARALLEL"], v["RESERVE"]), ("1", 500000, 1, "0.5"))

    rc, v = settings(YARN="1", CONTEXT="800000", PARALLEL="2", TENSORFOLD_MEMORY_RESERVE_GIB="1", PROMPT_CACHE_GIB="8")
    check("explicit rc", rc, 0)
    check("explicit settings win", (v["CONTEXT"], v["PARALLEL"], v["RESERVE"], v["CACHE"]), (800000, 2, "1", "8"))

    rc, v = settings(CONTEXT="262144")
    check("CONTEXT=262144 stays plain", (rc, v["YARN"], v["PARALLEL"]), (0, "0", 3))

    rc, why = settings(YARN="1", CONTEXT="200000")
    check("YARN=1 CONTEXT=200000 stops", rc != 0 and "past 262,144" in why, True)
    rc, why = settings(YARN="2")
    check("YARN=2 stops", rc != 0 and "YARN must be 0 or 1" in why, True)

    rc, v = settings(YARN="1", PARALLEL="2")
    check("two long streams rc", rc, 0)
    if v["NEED"] <= EMPTY_CARD_MIB:
        bad.append(f"two 1,000,000-token streams need only {v['NEED']} MiB: the gate would let them load")

    for line in bad:
        print("FAIL:", line)
    if not bad:
        print("PASS: YARN settings")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
