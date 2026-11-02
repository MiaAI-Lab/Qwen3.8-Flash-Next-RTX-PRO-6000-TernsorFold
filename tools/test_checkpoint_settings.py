#!/usr/bin/env python3
"""Which checkpoint scripts/config.sh picks for each way of asking.

Runs scripts/config.sh in an empty folder (so this repo's own .env does not count), with no server, no GPU and no Docker. Checks:
  - by default, and with ABLIT=1, it is NVIDIA's FP4 in its original form, at its pinned revision;
  - the modified FP4 (AZ=1, CHECKPOINT=az) stops, however it is asked for, as do CHECKPOINT=int4 and unknown values;
  - a MODEL_ID of your own stays yours, with no pin;
  - .env is read, and the environment wins over it.

Usage: tools/test_checkpoint_settings.py
"""
import os
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CONFIG = ROOT / "scripts" / "config.sh"
NAMES = ("AZ", "CHECKPOINT", "ABLIT", "MODEL_ID", "MODEL_REVISION", "ABLIT_REVISION")
NVIDIA = ("nvfp4", "nvidia/Qwen3.8-Flash-Next-FP4-Local", "b9ce18a6887e18c089fdea8509a33e63429a3300")


def pick(dotenv="", **env):
    """Source config.sh with `env` set and `dotenv` as .env; return (rc, (checkpoint, model, revision) or stderr)."""
    clean = {k: v for k, v in os.environ.items() if k not in NAMES}
    clean.update(env)
    with tempfile.TemporaryDirectory() as d:
        if dotenv:
            (Path(d) / ".env").write_text(dotenv)
        p = subprocess.run(["bash", "-c", f'source "{CONFIG}"; printf "%s\\n" "$CHECKPOINT" "$MODEL_ID" "$MODEL_REVISION"'],
                           cwd=d, env=clean, capture_output=True, text=True)
    if p.returncode != 0:
        return p.returncode, p.stderr.strip()
    return 0, tuple(p.stdout.split("\n")[:3])


def main() -> int:
    bad = []

    def serves(name, want, **kw):
        got = pick(**kw)
        if got != (0, want):
            bad.append(f"{name}: got {got!r}, want {want!r}")

    def stops(name, why, **kw):
        rc, got = pick(**kw)
        if rc == 0 or why not in got:
            bad.append(f"{name}: rc {rc}, {got!r}: want it to stop with {why!r}")

    serves("the default", NVIDIA)
    serves("CHECKPOINT=nvfp4", NVIDIA, CHECKPOINT="nvfp4")
    serves("AZ=0 changes nothing", NVIDIA, AZ="0")
    serves("an empty AZ changes nothing", NVIDIA, AZ="")
    serves("ABLIT=1 takes the original", NVIDIA, ABLIT="1")
    serves("ABLIT=0", NVIDIA, ABLIT="0")

    serves("a MODEL_ID of your own takes no pin", ("nvfp4", "someone/else", ""), MODEL_ID="someone/else")
    serves("a MODEL_ID and a MODEL_REVISION", ("nvfp4", "someone/else", "abc123"), MODEL_ID="someone/else", MODEL_REVISION="abc123")
    serves("a MODEL_REVISION alone moves the pin", ("nvfp4", NVIDIA[1], "abc123"), MODEL_REVISION="abc123")

    stops("AZ=1", "original form", AZ="1")
    stops("AZ=2", "original form", AZ="2")
    stops("CHECKPOINT=az", "original form", CHECKPOINT="az")
    stops("AZ=0 CHECKPOINT=az", "original form", AZ="0", CHECKPOINT="az")
    stops("ABLIT=1 AZ=1", "original form", ABLIT="1", AZ="1")
    stops("CHECKPOINT=int4", "not served", CHECKPOINT="int4")
    stops("CHECKPOINT=other", "CHECKPOINT is nvfp4", CHECKPOINT="other")
    stops("ABLIT=1 with another model", "ABLIT=1 grafts", ABLIT="1", MODEL_ID="someone/else")
    stops("ABLIT=1 with another revision", "ABLIT=1 grafts", ABLIT="1", MODEL_REVISION="abc123")

    serves(".env CHECKPOINT=nvfp4", NVIDIA, dotenv="CHECKPOINT=nvfp4\n")
    stops(".env AZ=1", "original form", dotenv="AZ=1\n")
    stops(".env CHECKPOINT=az", "original form", dotenv="CHECKPOINT=az\n")
    serves("the environment wins over .env", NVIDIA, dotenv="AZ=1\n", AZ="0")

    for line in bad:
        print("FAIL:", line)
    if not bad:
        print("PASS: checkpoint settings")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
