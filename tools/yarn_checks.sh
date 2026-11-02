#!/bin/bash
# Every check of a long window, against a server started with YARN=1 (one stream at 1,000,000 tokens), one after another.
# About 20 minutes: the longest needles read most of a million tokens each, about 5 minutes apiece. Results go to LOG and LOG.<check>; exit 1 if any check failed.
# Usage: tools/yarn_checks.sh LOG [CONTEXT]        (API_URL / PORT / MODEL as in tools/client.py)
set -u
cd "$(dirname "$(readlink -f "$0")")/.."
L="${1:?usage: tools/yarn_checks.sh LOG [CONTEXT]}"
C="${2:-1000000}"
: > "$L"
fails=0
run() {
  local n="$1" o="$L.$1" s rc
  shift
  s=$(date +%s)
  timeout 3000 "$@" > "$o" 2>&1
  rc=$?
  (( rc == 0 )) || fails=$((fails + 1))
  echo "$n rc=$rc $(( $(date +%s) - s ))s | $(tail -1 "$o" | cut -c1-170)" | tee -a "$L"
}
# Answers must not depend on drafting, on what ran before, or on a resume, at short lengths and past the trained window.
run exact python3 tools/exact.py --pad 3000
run exact-long python3 tools/exact.py --pad 300k --prompts 2
run resume python3 tools/resume_exact.py
run reuse python3 tools/prompt_reuse.py 300000
# The window's edges, with drafts on and off.
run boundary python3 tools/context_boundary.py --context "$C"
# Tools, images and video still work with the rotary changed.
run toolcheck python3 tools/toolcheck.py
run visioncheck python3 tools/visioncheck.py
run videocheck python3 tools/videocheck.py
# A passphrase has to be found past the trained window: at the plain window's length, then a quarter past it and near the end.
run needle-200k python3 tools/needle.py needle-200k 200000
run needle-550k python3 tools/needle.py needle-550k 550000
run needle-990k python3 tools/needle.py needle-990k "$(( C * 99 / 100 ))"
NEEDLE_DEPTH=0.05 run needle-early python3 tools/needle.py needle-early "$(( C * 99 / 100 ))"
echo "DONE, $fails failed" | tee -a "$L"
(( fails == 0 ))
