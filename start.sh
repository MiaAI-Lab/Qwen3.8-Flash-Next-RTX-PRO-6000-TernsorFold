#!/usr/bin/env bash
# Serve Qwen3.8 Flash Next (NVIDIA's QAD FP4 checkpoint) with TensorFold's Zig engine on one RTX PRO 6000 (TP=1):
# runs scripts/prepare.sh when the image or the checkpoint is not ready, launches tensorfold-native serve on port
# 8888, waits until the OpenAI API answers, then runs a smoke test. Stop it with ./stop.sh.
#
# Usage: ./start.sh [restart] [extra tensorfold-native serve args]
#   ./start.sh                         # up to PARALLEL requests at once, a 262,144-token window, fp8 KV, image and video, MTP drafts
#   ./start.sh restart                 # stop, then start again; the new arguments are checked before stopping
#   ./start.sh restart --parallel 2 --context 131072
#   YARN=1 ./start.sh restart          # one stream, a 1,000,000-token window with static YaRN x4 (patch 0025)
#   GPU=1 ./start.sh restart           # another card, by its nvidia-smi index
#   DRAFTS=0 ./start.sh restart        # no MTP drafts (--no-drafts)
#   DRY_RUN=1 ./start.sh               # print the docker command and exit
# Extra arguments go to tensorfold-native serve after the defaults, so they win.
# Settings, from the environment or ./.env (scripts/config.sh):
#   serving  CONTEXT, PARALLEL, MAX_TOKENS, THINKING, DRAFTS, TEMPERATURE, TOP_P, TOP_K, SERVED_NAME, HOST, PORT
#   files    CHECKPOINT (nvfp4), MODEL_ID, MODEL_REVISION, HF_CACHE, KERNEL_CACHE, STATE_DIR
#   window   YARN=1 (or a CONTEXT past 262,144, up to 1,048,576) rotates every prompt with static YaRN x4; YARN=0 (default) is the plain rotary at 262,144
#   ablit    ABLIT=1 serves NVIDIA's FP4 with the Keys house-qsa abliteration overlay (gated; needs an HF token); 0 is stock (default)
#   image    IMAGE, TF_REPO, TF_REF, ZIG_VERSION, BASE_IMAGE, GHCR_IMAGE, CONTAINER_NAME
#   card     GPU (nvidia-smi index), CUDA_SM, TENSORFOLD_MEMORY_RESERVE_GIB
#   setup    PREPARE (auto | 1 | 0), PULL, FOREGROUND=1, WAIT_TIMEOUT, DRY_RUN=1, GPU_NEED_MIB, MEM_FLOOR_GIB, MEM_CHECK=0
#   engine   every TENSORFOLD_* and TF_FLASHNEXT_* variable goes into the container
#            (TENSORFOLD_API_KEY by name only). KV_DTYPE is fp8 or bf16. VISION=1 passes --vision (image and video).
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")"
source ./scripts/config.sh

usage() { awk 'NR > 1 && /^#/ { sub(/^# ?/, ""); print; next } NR > 1 { exit }' "$0"; }
WAIT_TIMEOUT="${WAIT_TIMEOUT:-1800}"
MODE=start
case "${1:-}" in
  restart) MODE=restart; shift ;;
  help) usage; exit 0 ;;
esac
for arg in "$@"; do [[ "$arg" == -h || "$arg" == --help ]] && { usage; exit 0; }; done

[[ "$CONTEXT" =~ ^[1-9][0-9]*$ && "$CONTEXT" -le 1048576 ]] || die "CONTEXT is a token count up to 1048576, not $CONTEXT"
[[ "$PARALLEL" =~ ^([1-9]|1[0-6])$ ]] || die "PARALLEL is 1 to 16, not $PARALLEL"
[[ "$MAX_TOKENS" =~ ^[1-9][0-9]*$ ]] || die "MAX_TOKENS is a token count, not $MAX_TOKENS"
for v in THINKING DRAFTS VISION VISION_URLS; do [[ "${!v}" =~ ^[01]$ ]] || die "$v is 0 or 1, not ${!v}"; done
if (( CONTEXT > 262144 )); then
  [[ "$KV_DTYPE" == fp8 ]] || die "CONTEXT=$CONTEXT with KV_DTYPE=$KV_DTYPE: a long window fits this card only in fp8"
  log "A $CONTEXT-token window: every prompt rotates with static YaRN x4, which can cost short texts"
fi
DRY=0; [[ "${DRY_RUN:-0}" == 1 ]] && DRY=1

SERVE_ARGS=(--context "$CONTEXT" --parallel "$PARALLEL" --max-tokens "$MAX_TOKENS"
            --temperature "$TEMPERATURE" --top-p "$TOP_P" --top-k "$TOP_K")
if [[ "$THINKING" == 1 ]]; then SERVE_ARGS+=(--thinking); else SERVE_ARGS+=(--no-thinking); fi
[[ "$DRAFTS" == 1 ]] || SERVE_ARGS+=(--no-drafts)
[[ "$KV_DTYPE" =~ ^(bf16|fp8)$ ]] || die "KV_DTYPE is bf16 or fp8, not $KV_DTYPE"
SERVE_ARGS+=(--kv-cache "$KV_DTYPE" --prompt-cache-gib "$PROMPT_CACHE_GIB")
[[ "$MAX_VIDEOS" =~ ^[0-9]+$ ]] || die "MAX_VIDEOS is a count, not $MAX_VIDEOS"
if [[ "$VISION" == 1 ]]; then
  SERVE_ARGS+=(--vision --vision-max-images "$TENSORFOLD_MAX_IMAGES" --vision-max-videos "$MAX_VIDEOS"
               --vision-image-tokens "$TENSORFOLD_IMAGE_TOKENS" --vision-video-tokens "$TENSORFOLD_VIDEO_TOKENS")
  [[ "$VISION_URLS" == 1 ]] && SERVE_ARGS+=(--vision-urls)
fi
USER_ARGS=("$@")
arg_value() {
  local flag=$1 value="" i
  local -a all=("${SERVE_ARGS[@]}" "${USER_ARGS[@]}")
  for (( i = 0; i < ${#all[@]}; i++ )); do
    case "${all[i]}" in
      "$flag") value="${all[i + 1]:-}" ;;
      "$flag="*) value="${all[i]#*=}" ;;
    esac
  done
  echo "$value"
}
API_HOST="$HOST"; [[ "$HOST" == 0.0.0.0 || "$HOST" == "::" ]] && API_HOST=127.0.0.1
[[ "$API_HOST" == *:* ]] && API_HOST="[$API_HOST]"
URL="http://$API_HOST:$PORT"

running() { [[ "$(docker inspect -f '{{.State.Running}}' "$CONTAINER_NAME" 2>/dev/null)" == true ]]; }
served_name() {
  curl -s --max-time 5 "$URL/v1/models" 2>/dev/null |
    python3 -c 'import json,sys; print(json.load(sys.stdin)["data"][0]["id"])' 2>/dev/null
}

B=$'\033[1m'; M=$'\033[1;35m'; G=$'\033[1;32m'; D=$'\033[2m'; R=$'\033[0m'
[[ -t 1 ]] || { B=; M=; G=; D=; R=; }
source ./scripts/banner.sh
echo
banner
printf '\n%s  Mia'"'"'s TensorFold Start Script%s\n' "$M" "$R"
printf '%s  %s · GPU %s · %s at once · %s-token window · %s KV · vision %s · drafts %s · port %s%s\n\n' "$D" "$MODEL_ID$( [[ "$ABLIT" == 1 ]] && echo ' + ablit')" "$GPU" \
  "$(arg_value --parallel)" "$(arg_value --context)" "$KV_DTYPE" \
  "$( [[ "$VISION" == 1 ]] && echo on || echo off)" \
  "$( [[ "$DRAFTS" == 1 ]] && echo MTP || echo off)" "$PORT" "$R"
STEPS=5
step() { printf '%s[%s/%s]%s %s%s%s\n' "$M" "$1" "$STEPS" "$R" "$B" "$2" "$R"; }

[[ "$(uname -m)" == x86_64 ]] || die "this recipe builds for x86_64 and sm_120 (RTX PRO 6000); this host is $(uname -m)"
[[ "$GPU" =~ ^[0-9]+$ ]] || die "GPU is one nvidia-smi index, not $GPU (TP=1: one card)"
command -v docker >/dev/null || die "docker is not installed"
mkdir -p "$KERNEL_CACHE" "$STATE_DIR"
exec 8>"$STATE_DIR/start.lock"
flock -n 8 || die "another ./start.sh is already running; wait for it to finish"
(( DRY )) && log "DRY_RUN=1: printing the docker command; nothing is stopped or started"

if (( ! DRY )) && [[ "$MODE" == start ]] && running; then
  log "$CONTAINER_NAME is already running (model: $(served_name || echo "not answering yet"), port $PORT): nothing to do."
  log "Use ./start.sh restart to restart it, or ./stop.sh to stop it."
  exit 0
fi

step 1 "Setup: image and checkpoint"
if [[ "${PREPARE:-auto}" == 1 || ( "${PREPARE:-auto}" != 0 && "$(prepared_state 2>/dev/null)" != "$(cat "$PREPARED_MARKER" 2>/dev/null)" ) ]]; then
  if (( DRY )); then log "DRY_RUN: scripts/prepare.sh would run now"
  else
    log "Not ready yet: running scripts/prepare.sh"
    ./scripts/prepare.sh
  fi
else
  log "Ready: $IMAGE (patches $(image_hash)) and $MODEL_ID"
fi
why="scripts/prepare.sh did not"; [[ "${PREPARE:-auto}" == 0 ]] && why="PREPARE=0 skipped scripts/prepare.sh, which would"
(( DRY )) && why="DRY_RUN: scripts/prepare.sh would"
missing() { if (( DRY )); then warn "$1"; else die "$1"; fi; }
have_image=1
docker image inspect "$IMAGE" >/dev/null 2>&1 || { have_image=0; missing "image $IMAGE missing: $why build it"; }
if (( have_image )); then
  kernels=$(docker image inspect -f '{{index .Config.Labels "tf.kernels"}}' "$IMAGE" 2>/dev/null || true)
  [[ "$kernels" == embedded ]] || missing "$IMAGE is not an image of this recipe (tf.kernels=${kernels:-?}): scripts/prepare.sh --rebuild"
fi
KCACHE=$(docker image inspect -f '{{index .Config.Labels "tf.patches"}}' "$IMAGE" 2>/dev/null || true)
[[ "$KCACHE" =~ ^[0-9a-f]{12}$ ]] || KCACHE=$(image_hash)
rev=$(snapshot_rev)
sub=$(model_sub)
if [[ ! -f "$HF_CACHE/$sub/config.json" ]]; then missing "$MODEL_ID @ ${rev:0:8} not in $HF_CACHE ($sub): $why download it"; fi
MODEL_ARG="/root/.cache/huggingface/$sub"

step 2 "Checks: arguments, previous server, port, the card, host memory"
if (( have_image )); then
  code=0
  docker run --rm --network none --entrypoint /opt/tensorfold/native/bin/tensorfold-native "$IMAGE" \
    serve /nonexistent/argument-check --name "$SERVED_NAME" --host "$HOST" --port "$PORT" "${SERVE_ARGS[@]}" "${USER_ARGS[@]}" \
    >/dev/null 2>"$STATE_DIR/args.err" || code=$?
  if (( code == 2 )) || ! grep -q 'neither a directory' "$STATE_DIR/args.err"; then
    if (( DRY )); then warn "DRY_RUN: tensorfold-native rejects these arguments: $(grep -v '^usage:' "$STATE_DIR/args.err" | tail -1)"
    else cat "$STATE_DIR/args.err" >&2; die "tensorfold-native serve rejects these arguments; nothing was changed"; fi
  fi
fi
if (( DRY )); then
  running && log "DRY_RUN: $CONTAINER_NAME is running; a real start would stop it first"
elif [[ "$MODE" == restart ]] && running; then
  ./stop.sh
elif [[ "$MODE" == start ]] && running; then
  log "A server was left running: stopping it, then starting"
  ./stop.sh
fi
if (( ! DRY )) && docker ps -a --format '{{.Names}}' | grep -qx "$CONTAINER_NAME"; then
  log "Removing the previous container $CONTAINER_NAME"
  mkdir -p "$LOG_DIR"
  docker logs "$CONTAINER_NAME" > "$LOG_DIR/$(date +%Y%m%d-%H%M%S).log" 2>&1 || true
  docker rm -f "$CONTAINER_NAME" >/dev/null
fi
if ss -ltn "sport = :$PORT" 2>/dev/null | grep -q LISTEN; then
  (( DRY )) && log "DRY_RUN: port $PORT is in use" ||
    die "port $PORT is already in use: $(ss -ltnp "sport = :$PORT" 2>/dev/null | tail -n +2)"
fi
avail_gb=$(awk '/^MemAvailable:/ {print int($2 / 1048576)}' /proc/meminfo)   # not free(1): its row labels are translated
memory_short() { if (( DRY )) || [[ "${MEM_CHECK:-1}" == 0 ]]; then warn "$1"; else die "$1 (MEM_CHECK=0 starts anyway)"; fi; }
(( avail_gb >= MEM_FLOOR_GIB )) || memory_short "only ${avail_gb} GiB of host memory available (MEM_FLOOR_GIB $MEM_FLOOR_GIB): the weights stage through it while they load"
streams=$(arg_value --parallel); [[ "$streams" =~ ^[0-9]+$ ]] || streams=$PARALLEL
need_mib=${GPU_NEED_MIB:-$(gpu_need_mib "$(arg_value --context)" "$streams")}
free_mib=$(gpu_free_mib)
card=$(gpu_name)
[[ -n "$free_mib" && -n "$card" ]] || die "no answer from GPU $GPU: set GPU to an index nvidia-smi lists, and check the driver"
if (( free_mib >= need_mib )); then
  log "Arguments OK, port $PORT free, GPU $GPU ($card) has $free_mib MiB free ($streams streams of $(arg_value --context) tokens need ~$need_mib), ${avail_gb} GiB host memory available"
else
  busy=$(nvidia-smi -i "$GPU" --query-compute-apps=pid,process_name,used_memory --format=csv,noheader 2>/dev/null | awk -F', *' '$3 + 0 >= 1024 { sub(/.*\//, "", $2); printf " %s (pid %s, %s)", $2, $1, $3 }')
  memory_short "GPU $GPU has $free_mib MiB free; $streams streams of $(arg_value --context) tokens need ~$need_mib MiB.${busy:+ On it:$busy.} Stop what holds the card, or lower PARALLEL or CONTEXT"
fi

ENV_ARGS=(-e HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}")
while IFS='=' read -r name _; do
  [[ "$name" == TENSORFOLD_API_KEY ]] && continue
  ENV_ARGS+=(-e "$name=${!name}")
done < <(env | grep -E '^(TENSORFOLD|TF_FLASHNEXT)_[A-Z0-9_]+=' || true)
KEY_ARGS=(); [[ -z "${TENSORFOLD_API_KEY:-}" ]] || { export TENSORFOLD_API_KEY; KEY_ARGS=(-e TENSORFOLD_API_KEY); }

step 3 "Launch: container $CONTAINER_NAME"
log "tensorfold-native serve $MODEL_ARG --name $SERVED_NAME --host $HOST --port $PORT ${SERVE_ARGS[*]} ${USER_ARGS[*]}"
here_cmd=(docker run -d --name "$CONTAINER_NAME"
          --gpus "device=$GPU" --ipc=host --network host
          --ulimit memlock=-1 --ulimit stack=67108864
          "${ENV_ARGS[@]}" "${KEY_ARGS[@]}"
          -v "$HF_CACHE":/root/.cache/huggingface:ro
          -v "$KERNEL_CACHE/$KCACHE":/cache
          "$IMAGE"
          tensorfold-native serve "$MODEL_ARG" --name "$SERVED_NAME" --host "$HOST" --port "$PORT"
          "${SERVE_ARGS[@]}" "${USER_ARGS[@]}")
if (( DRY )); then
  printf '[dry-run]\n  %s\n' "$(printf '%q ' "${here_cmd[@]}" | sed 's/ $//')"
  exit 0
fi
"${here_cmd[@]}" >/dev/null

if [[ "${FOREGROUND:-0}" == 1 ]]; then
  trap './stop.sh; exit 130' INT TERM
  docker logs -f "$CONTAINER_NAME" || true
  exit "$(docker inspect -f '{{.State.ExitCode}}' "$CONTAINER_NAME" 2>/dev/null || echo 1)"
fi

step 4 "Loading: ~$WEIGHTS_GIB GiB onto GPU $GPU (the n-gram table stays on disk)"
NOISE='^\s*$|^=+$|^== PyTorch ==|^NVIDIA Release|Copyright|All rights reserved|PyTorch Version|Various files include|NOTE: CUDA Forward|Using CUDA|cuda-compatibility|Container image|^\[tensorfold\] [^ ]+ "GET /(health|v1/models|models|metrics|stats|slots)[ ?]'
docker logs -f "$CONTAINER_NAME" > >(grep --line-buffered -v -E "$NOISE" | sed -u "s/^/  ${D}│${R} /") 2>&1 &
LOGS_PID=$!
trap 'kill $LOGS_PID 2>/dev/null || true' EXIT
loaded_gib() {
  local pids
  pids=$(docker top "$CONTAINER_NAME" -eo pid 2>/dev/null | tail -n +2 | paste -sd'|')
  [[ -n "$pids" ]] || { echo 0; return; }
  nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader,nounits 2>/dev/null |
    awk -F', *' -v re="^($pids)$" '$1 ~ re { s += $2 } END { printf "%.1f", s / 1024 }'
}
fail() {
  kill $LOGS_PID 2>/dev/null || true
  sleep 0.5
  printf '\n%s── last server log lines ──%s\n' "$D" "$R"
  docker logs --tail 40 "$CONTAINER_NAME" 2>&1 | sed 's/^/  │ /'
  die "$1"
}
start=$SECONDS
next_beat=15
until curl -sf --max-time 5 "$URL/health" >/dev/null 2>&1; do
  running || fail "the server exited (code $(docker inspect -f '{{.State.ExitCode}}' "$CONTAINER_NAME")) before it was ready"
  (( SECONDS - start < WAIT_TIMEOUT )) || fail "not ready after ${WAIT_TIMEOUT}s; it is still running: docker logs -f $CONTAINER_NAME"
  a=$(awk '/MemAvailable/ {print int($2 / 1048576)}' /proc/meminfo)
  if (( a < MEM_FLOOR_GIB )); then
    docker stop -t 5 "$CONTAINER_NAME" >/dev/null 2>&1 || true
    fail "only $a GiB host memory left while loading (MEM_FLOOR_GIB $MEM_FLOOR_GIB): stopped"
  fi
  if (( SECONDS - start >= next_beat )); then
    printf '  %s⋯ %ss elapsed, %s GiB on the GPU (%s GiB host memory free)%s\n' "$D" "$((SECONDS - start))" "$(loaded_gib)" "$a" "$R"
    next_beat=$((next_beat + 15))
  fi
  sleep 3
done
kill $LOGS_PID 2>/dev/null || true
sleep 0.3
LOADED_S=$((SECONDS - start))
log "Server answered after ${LOADED_S}s"

step 5 "Smoke test: one chat completion"
SERVED=$(served_name || echo "$SERVED_NAME")
body="{\"model\": \"$SERVED\", \"max_tokens\": 32, \"temperature\": 0, \"chat_template_kwargs\": {\"enable_thinking\": false}, \"messages\": [{\"role\": \"user\", \"content\": \"Reply with OK.\"}]}"
if smoke=$(curl -s --max-time 180 "$URL/v1/chat/completions" -H 'Content-Type: application/json' -d "$body" |
           python3 -c 'import json,sys; r = json.load(sys.stdin); c = r["choices"][0]["message"].get("content") or ""; assert c.strip(); print(repr(c.strip()[:40]) + ",", r["usage"]["completion_tokens"], "tokens")' 2>/dev/null); then
  log "OK: $smoke"
else
  fail "the smoke test request failed; the server is still running"
fi
IP=$(hostname -I 2>/dev/null | awk '{print $1}')
[[ "$HOST" == 0.0.0.0 || "$HOST" == "::" ]] || IP="$HOST"
printf '\n%s  ✔ %s is now LIVE! on port %s%s\n\n' "$G" "$SERVED" "$PORT" "$R"
cat <<EOF
    API      http://${IP:-<host>}:$PORT/v1   (model: $SERVED)
    Card     GPU $GPU ($card)
    Loaded   in ${LOADED_S}s
    Model    $MODEL_ID @ ${MODEL_REVISION:0:8} ($QUANT_LABEL)
    Window   $(arg_value --context) tokens$( (( $(arg_value --context) > 262144 )) && echo ' (YaRN x4)') · $(arg_value --parallel) at once · $KV_DTYPE KV · drafts $( [[ "$DRAFTS" == 1 ]] && echo MTP || echo off)
    Logs     docker logs -f $CONTAINER_NAME
    Restart  ./start.sh restart
    Stop     ./stop.sh

EOF
