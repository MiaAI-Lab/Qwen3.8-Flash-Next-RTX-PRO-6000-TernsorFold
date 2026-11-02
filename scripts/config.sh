# Shared settings for start.sh, stop.sh and scripts/*.sh. A setting's value comes from the first of these that sets it:
#   1. the environment: `PORT=9000 ./start.sh`, `PULL=0 scripts/prepare.sh`
#   2. ./.env (KEY=value lines, read, never run)
# Settings marked PROVISIONAL follow TensorFold's Zig engine (tensorfold-native) and change when a gated commit lands.
if [[ -f .env ]]; then
  while IFS= read -r _line || [[ -n "$_line" ]]; do
    [[ "$_line" =~ ^[[:space:]]*(export[[:space:]]+)?([A-Za-z_][A-Za-z0-9_]*)=(.*)$ ]] || continue
    _key=${BASH_REMATCH[2]}; _value=${BASH_REMATCH[3]}
    if [[ "$_value" =~ ^\"([^\"]*)\"[[:space:]]*(#.*)?$ || "$_value" =~ ^\'([^\']*)\'[[:space:]]*(#.*)?$ ]]; then
      _value=${BASH_REMATCH[1]}
    else
      _value=${_value%%#*}; _value=${_value%"${_value##*[![:space:]]}"}
    fi
    [[ -n "${!_key+set}" ]] || export "$_key=$_value"
  done < .env
fi
unset _line _key _value

# Checkpoint. NVIDIA's quantization-aware-distilled FP4 (NVFP4 routed experts, MXFP8 and bf16 elsewhere, FP8 n-gram table, MTP head), nvidia/Qwen3.8-Flash-Next-FP4-Local, ~96 GiB,
# as NVIDIA published it: FP4 is only served in its original form, so the 5-expert AZ cut stops here. ABLIT=1 below adds an overlay to this same checkpoint. azampatti's GPTQ INT4-AutoRound is a
# different format with no kernels in this engine: CHECKPOINT=int4 stops rather than start an engine that would refuse it.
# The engine reads the format from the checkpoint. WEIGHTS_GIB is what loading takes of the card before any stream: the engine's startup line calls 75.92 GiB the model (weights,
# vision tower, scratch), and its budget leaves about 1.1 GiB less again (measured, vision on). It only decides whether start.sh tries a load at all; the engine does the real sizing.
if [[ -n "${AZ:-}" && "$AZ" != 0 ]]; then
  echo "AZ=$AZ is not served: FP4 is only served in its original form, NVIDIA's. Leave AZ out." >&2; exit 1
fi
CHECKPOINT="${CHECKPOINT:-nvfp4}"
_pin=1; [[ -z "${MODEL_ID:-}" ]] || _pin=0       # a MODEL_ID set by hand takes no preset revision pin
case "$CHECKPOINT" in
  nvfp4)
    : "${MODEL_ID:=nvidia/Qwen3.8-Flash-Next-FP4-Local}"
    (( ! _pin )) || : "${MODEL_REVISION:=b9ce18a6887e18c089fdea8509a33e63429a3300}"
    CKPT_GIB=96; WEIGHTS_GIB="${WEIGHTS_GIB:-77}"
    QUANT_LABEL="NVIDIA FP4 QAD: NVFP4 experts, MXFP8" ;;
  az)
    echo "CHECKPOINT=az is not served: FP4 is only served in its original form, CHECKPOINT=nvfp4." >&2
    exit 1 ;;
  int4)
    echo "CHECKPOINT=int4 is not served: azampatti's INT4-AutoRound is GPTQ int4, and this engine has no kernels for it." >&2
    exit 1 ;;
  *) echo "CHECKPOINT is nvfp4, not $CHECKPOINT" >&2; exit 1 ;;
esac
MODEL_REVISION="${MODEL_REVISION:-}"

# ABLIT=1 serves NVIDIA's FP4-Local with drowzeys's Keys "house-qsa" abliteration overlay grafted in: the 12 QSA layers' self_attn.o_proj
# (L3, 7, ... 47; MXFP8 g32, requantized after a rank-1 refusal-direction projection) come from the overlay's ~190 MB sidecar. GDN out_proj,
# every expert, the MTP head, the n-gram table, vision and the chat template stay NVIDIA's. The overlay is gated, free, with automatic approval after you accept
# its Responsible Use Agreement at https://huggingface.co/drowzeys/keys-Qwen3.8-Flash-Next-NVFP4-FP4-QAD-house-qsa-L3-47 while logged in;
# export HF_TOKEN or `hf auth login` with that account. It only fits the nvfp4 checkpoint at the revision it was made from. The model has had safety
# refusals removed. Its author's own checks (refusal32 32/32) ran on the vLLM/TensorFold recipe, not on this build: validate it here (tools/exact.py,
# tools/needle.py) before relying on it.
ABLIT="${ABLIT:-0}"
[[ "$ABLIT" =~ ^[01]$ ]] || { echo "ABLIT must be 0 or 1, not $ABLIT" >&2; exit 1; }
ABLIT_ID=drowzeys/keys-Qwen3.8-Flash-Next-NVFP4-FP4-QAD-house-qsa-L3-47
ABLIT_REVISION="${ABLIT_REVISION:-c44f5b92ee3d307b82aab1c55f4cc4ae1d440344}"
ABLIT_SIDECAR=fp4-local/house-qsa-oproj-mxfp8-L3-47.safetensors
ABLIT_STOCK_REVISION=b9ce18a6887e18c089fdea8509a33e63429a3300     # the nvidia/Qwen3.8-Flash-Next-FP4-Local snapshot the sidecar was made from
if [[ "$ABLIT" == 1 ]]; then
  if [[ "$CHECKPOINT" != nvfp4 || "$_pin" != 1 || "$MODEL_REVISION" != "$ABLIT_STOCK_REVISION" ]]; then
    echo "ABLIT=1 grafts onto nvidia/Qwen3.8-Flash-Next-FP4-Local @ ${ABLIT_STOCK_REVISION:0:8} (CHECKPOINT=nvfp4, no MODEL_ID or MODEL_REVISION of your own)" >&2
    exit 1
  fi
  QUANT_LABEL+=" + Keys house-qsa ablit"
fi
unset _pin

# The engine: TensorFold's Zig engine (tensorfold-native), built from TF_REPO at TF_REF (the v1.1.4 release) with patches/*.patch
# applied (git apply in the checkout's root), with Zig ZIG_VERSION, for x86_64 and the card's SASS target (CUDA_SM). Only the Zig build goes into the
# image, plus the two image decoders the engine wants (libjpeg-turbo for JPEG, libwebp's decoder for WebP), built from pinned sources.
TF_REPO="${TF_REPO:-https://github.com/ashhart/TensorFold.git}"
TF_REF="${TF_REF:-eab8582819688325e371907e4706e1c602b8a273}"   # v1.1.4
ZIG_VERSION="${ZIG_VERSION:-0.17.0}"
ZIG_SHA256="${ZIG_SHA256:-1cbe9df9f27e6b78d14ccbca43b6703a404ef79ef1c463de901d7f088d4e2026}"   # zig-x86_64-linux-0.17.0.tar.xz
CUDA_SM="${CUDA_SM:-120}"                                # RTX PRO 6000 Blackwell is sm_120; TensorFold's build defaults to sm_121, whose kernels do not load here
ZIG_CPU="${ZIG_CPU:-native}"                             # -Dcpu for the host code (draws, n-gram gather). native is the build machine; set x86_64_v3 for an image that moves
BASE_IMAGE="${BASE_IMAGE:-nvcr.io/nvidia/pytorch:26.07-py3}"
LIBJPEG_TURBO_VERSION=3.1.4.1
LIBJPEG_TURBO_SHA256=ecae8008e2cc9ade2f2c1bb9d5e6d4fb73e7c433866a056bd82980741571a022
LIBWEBP_VERSION=1.6.0
LIBWEBP_SHA256=e4ab7009bf0629fd11982d4c2aa83964cf244cffba7347ecd39019a9e38c4564
# libtfgrammar.so (structured output, tool_choice "required" and named functions) comes prebuilt from TensorFold's own release archive, checked against its SHA-256;
# the archive ships it with xgrammar's Apache-2.0 notices. TF_RELEASE is the release that matches TF_REF.
TF_RELEASE_VERSION=1.1.4
TF_RELEASE_X86_64_SHA256=878a23f0eddc1e0750e49c5e279e57479c48be3d7c3de0a1d1eed81ae18ce2b0
TF_TAG="v1.1.4-${TF_REF:0:7}"
IMAGE="${IMAGE:-tensorfold-qwen38:${TF_TAG}-sm${CUDA_SM}}"
image_hash() {
  ( cd "$(dirname "${BASH_SOURCE[0]}")/.." || exit 1
    cat patches/*.patch 2>/dev/null
    sed -n "/<<'DOCKERFILE'/,/^DOCKERFILE\$/p" scripts/prepare.sh
    echo "$TF_REPO $TF_REF zig-$ZIG_VERSION $ZIG_SHA256 sm-$CUDA_SM cpu-$ZIG_CPU $BASE_IMAGE jpeg-$LIBJPEG_TURBO_VERSION-$LIBJPEG_TURBO_SHA256 webp-$LIBWEBP_VERSION-$LIBWEBP_SHA256 grammar-$TF_RELEASE_VERSION-$TF_RELEASE_X86_64_SHA256"
  ) | sha256sum | cut -c1-12
}
# No image of this port is published: prepare.sh builds it. Set GHCR_IMAGE to a registry path of your own to publish one (scripts/publish-image.sh) and to pull it
# (PULL=1).
GHCR_IMAGE="${GHCR_IMAGE:-}"
prebuilt_image() { echo "$GHCR_IMAGE:${TF_TAG}-sm${CUDA_SM}-$(image_hash)"; }
CONTAINER_NAME="${CONTAINER_NAME:-qwen38-flash-next-tf}"
# The card this server takes, by its nvidia-smi index. One rank, one GPU (TP=1): the other card stays free. On a machine with a desktop, take the card without
# the monitors: the desktop's memory comes off what the streams can use.
GPU="${GPU:-0}"

SERVED_NAME="${SERVED_NAME:-Qwen3.8-Flash-Next}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8888}"
# Native window 262,144. YARN=1 (or any CONTEXT past it, up to 1,048,576) serves a long window: patch 0025 rotates every prompt with static YaRN x4, as TensorFold's Metal engine does.
# Qwen recommends YaRN only where long prompts need it, since static scaling can cost short texts, so it is never the default. On this card it is one stream: 1,000,000 tokens
# fit beside the weights only with the engine's smallest reserve, 0.5 GiB (the engine reports 17.17 GiB for the stream of 17.2 GiB left). 1,048,576 needs DRAFTS=0 as well (calculated, not run).
YARN="${YARN:-0}"
[[ "$YARN" =~ ^[01]$ ]] || { echo "YARN must be 0 or 1, not $YARN" >&2; exit 1; }
if [[ "${CONTEXT:-}" =~ ^[0-9]+$ ]] && (( CONTEXT > 262144 )); then YARN=1; fi
if [[ "$YARN" == 1 ]]; then
  [[ -z "${CONTEXT:-}" || ! "$CONTEXT" =~ ^[0-9]+$ ]] || (( CONTEXT > 262144 )) || { echo "YARN=1 is a window past 262,144 tokens, not CONTEXT=$CONTEXT: leave YARN out for the plain rotary" >&2; exit 1; }
  CONTEXT="${CONTEXT:-1000000}"
  PARALLEL="${PARALLEL:-1}"
  export TENSORFOLD_MEMORY_RESERVE_GIB="${TENSORFOLD_MEMORY_RESERVE_GIB:-0.5}"
  # a 1,000,000-token state takes up to about 18 GiB, and a chat resumes from one only if it is kept whole
  PROMPT_CACHE_GIB="${PROMPT_CACHE_GIB:-20}"
fi
CONTEXT="${CONTEXT:-262144}"
# Requests decoded together. At start the engine counts a full window for every stream and refuses a PARALLEL that does not fit; the memory is taken per request (prompt, max_tokens, draft rows). Measured on an empty 96 GB card with vision on: 15.0 GiB left after the
# 3 GiB reserve, 4.86 GiB a stream at 262,144 tokens in fp8, so 3 fit and 4 do not. A shorter CONTEXT fits more (PARALLEL=4 CONTEXT=192000, calculated, not run).
PARALLEL="${PARALLEL:-3}"
# MTP: up to this many drafts a round (the engine's default is 6; drafts are checked, so replies match serial decoding).
export TF_FLASHNEXT_DRAFTS="${TF_FLASHNEXT_DRAFTS:-6}"
# Sampled requests (temperature > 0, what chat clients send) in the batched draft pass, with the top candidates picked on the GPU
# (patches 0007, 0008, 0011), all lanes' n-gram rows gathered in one pass (0010) and the host draws run on worker threads (0011).
# None of them changes a reply's tokens (tools/exact.py, pDS-sampled-exact: drafted == serial == concurrent). 0 turns each off.
export TF_FLASHNEXT_SAMPLED_HEADS="${TF_FLASHNEXT_SAMPLED_HEADS:-1}"
export TF_FLASHNEXT_SAMPLE_CAND="${TF_FLASHNEXT_SAMPLE_CAND:-1}"
export TF_FLASHNEXT_GATHER_EARLY="${TF_FLASHNEXT_GATHER_EARLY:-1}"
export TF_FLASHNEXT_HOST_DRAW_THREADS="${TF_FLASHNEXT_HOST_DRAW_THREADS:-7}"
# A partly rejected draft window no longer replays the DeltaNet state: the kept rows wait in a log and the next verify pass applies them (patch 0012).
export TF_FLASHNEXT_KEEP_LATE="${TF_FLASHNEXT_KEEP_LATE:-1}"
# The n-gram table is read with pread by a thread pool and read ahead (draft tokens' rows during the head step, the next prompt chunk's rows during
# the current one) instead of through an mmap that page-faults (patch 0013). The last two prompt chunks are evened out when the last would be under 512 rows.
export TF_FLASHNEXT_GATHER_READ="${TF_FLASHNEXT_GATHER_READ:-pread}"
export TF_FLASHNEXT_DRAFT_AHEAD="${TF_FLASHNEXT_DRAFT_AHEAD:-1}"
export TF_FLASHNEXT_WARM_AHEAD="${TF_FLASHNEXT_WARM_AHEAD:-1}"
export TF_FLASHNEXT_PREFILL_TAIL="${TF_FLASHNEXT_PREFILL_TAIL:-512}"
# Prompt passes: the hyper-connection normalize, down GEMM, up GEMM and mix run as fused kernels, the shared expert's rows go straight into the y slot,
# the routed experts use the shared-memory row tiles from 256 rows, and the MTP head takes prompt pieces of 1536 rows (patch 0014; bit-identical to the
# unfused path: tf-cuda-test hc-fuse, exact.py, pDS-sampled-exact).
export TF_FLASHNEXT_PROMPT_HC2="${TF_FLASHNEXT_PROMPT_HC2:-1}"
export TF_FLASHNEXT_PROMPT_SHARED_DIRECT="${TF_FLASHNEXT_PROMPT_SHARED_DIRECT:-1}"
export TF_FLASHNEXT_PROMPT_EXPERT_TILE_ROWS="${TF_FLASHNEXT_PROMPT_EXPERT_TILE_ROWS:-256}"
export TF_FLASHNEXT_HEAD_PIECE="${TF_FLASHNEXT_HEAD_PIECE:-1536}"
# The MX128 prompt tile with two pipeline stages and two blocks an SM (patch 0015): the same arithmetic and bits as the one-stage tile.
export TF_FLASHNEXT_PROMPT_GEMM2="${TF_FLASHNEXT_PROMPT_GEMM2:-1}"
# Decode rounds (patch 0016): the shared expert runs on a second stream beside the routed experts (decode rounds, windows and head passes only), and a shared
# round takes every lane's DeltaNet rows in three launches a layer instead of five a lane (needs KEEP_LATE=1). Both bit-identical (exact.py, pDS-sampled-exact).
export TF_FLASHNEXT_SHARED_SIDE="${TF_FLASHNEXT_SHARED_SIDE:-1}"
export TF_FLASHNEXT_GDN_LANES="${TF_FLASHNEXT_GDN_LANES:-1}"
# The draft chain stops on the running product of its draft probabilities (keep draft j while p1*...*pj >= 0.4; the db281878 build's rule) up to 5 streams
# drafting together, the per-draft floor 0.5 above (patch 0017). v1.1.4's default is the per-draft floor 0.6. Only the number of draft rows changes, never a reply.
# TF_FLASHNEXT_CONFIDENCE=0.6 restores the stock rule.
export TF_FLASHNEXT_CONFIDENCE="${TF_FLASHNEXT_CONFIDENCE:--0.4}"
export TF_FLASHNEXT_PRODUCT_STREAMS="${TF_FLASHNEXT_PRODUCT_STREAMS:-5}"
# Kept prompt states (a chat's next turn resumes from them; --prompt-cache-gib): pinned host memory taken as states are kept (patch 0019), at most half of what the host has
# available at start. A state costs about 18 KB a token, so 16 GiB holds about 900,000 tokens of states. On the card under 0.5 GiB is left beside three full windows, less than one 30,000-token
# chat. 0 turns the cache off; TF_FLASHNEXT_HOST_SNAPS=0 keeps states on the card.
PROMPT_CACHE_GIB="${PROMPT_CACHE_GIB:-16}"
export TF_FLASHNEXT_HOST_SNAPS="${TF_FLASHNEXT_HOST_SNAPS:-1}"
# KV cache (--kv-cache). fp8 is the default (patch 0003): about 1.84x the pool of bf16, and lossy (~98.8% top-1 agreement with a
# bf16 cache, so a free-running reply can differ). bf16 is exact: KV_DTYPE=bf16. FP8 works together with --vision.
KV_DTYPE="${KV_DTYPE:-fp8}"
MAX_TOKENS="${MAX_TOKENS:-32768}"
THINKING="${THINKING:-1}"
DRAFTS="${DRAFTS:-1}"
# Sampling defaults (a request's own values win). The Zig server honours these flags.
TEMPERATURE="${TEMPERATURE:-1.0}"
TOP_P="${TOP_P:-0.95}"
TOP_K="${TOP_K:-20}"
# Image and video. VISION=1 (default) passes --vision: the checkpoint's 27-layer tower runs in the engine, on the card; images are decoded with libjpeg-turbo and libwebp, video with
# libtfvideo over FFmpeg. VISION=0 is text only and gives the tower's memory back to the streams. VISION_URLS=1 also accepts public https:// URLs; the default is data URLs only.
VISION="${VISION:-1}"
VISION_URLS="${VISION_URLS:-0}"
export TENSORFOLD_MAX_IMAGES="${TENSORFOLD_MAX_IMAGES:-50}"
MAX_VIDEOS="${MAX_VIDEOS:-4}"                                          # videos a request (--vision-max-videos)
export TENSORFOLD_IMAGE_TOKENS="${TENSORFOLD_IMAGE_TOKENS:-16384}"
export TENSORFOLD_VIDEO_TOKENS="${TENSORFOLD_VIDEO_TOKENS:-16384}"
# What the engine keeps free on the card when it sizes the streams and the prompt cache: its own default for a 96 GB card (1/32, least 0.5).
# Raise it if the card's desktop grows under long sessions (GPU=<the monitors' card>).
export TENSORFOLD_MEMORY_RESERVE_GIB="${TENSORFOLD_MEMORY_RESERVE_GIB:-3}"
export TENSORFOLD_NO_UPDATE_CHECK="${TENSORFOLD_NO_UPDATE_CHECK:-1}"

# There is no int8 cache. The n-gram table stays on disk and the host reads its rows (pread, O_DIRECT where the file system allows it), so it takes neither VRAM nor much RAM.
# The card and the host are separate memories. The fp8 cache's 17,888 bytes a token and a state's 18 KB are the same on both. GPU_NEED_MIB is what the card must have free at start (default: gpu_need_mib below); MEM_FLOOR_GIB is the host's MemAvailable
# that start.sh stops a load at and prepare.sh refuses to start below, because the weights stage through host memory on their way to the card. MEM_CHECK=0 turns both into a warning.
GPU_NEED_MIB="${GPU_NEED_MIB:-}"
MEM_FLOOR_GIB="${MEM_FLOOR_GIB:-4}"
# stream_mib <tokens>: one stream on the card: its cache plus the fixed part beside it: draft rows and rings, and the request's image and video rows (patch 0020 adds 80 MiB for 16,384
# video tokens to the 4.78 GiB the engine reported at 262,144 tokens in fp8 before it).
stream_mib() {
  local ctx=${1:-$CONTEXT} bytes=30784
  [[ "$ctx" =~ ^[0-9]+$ ]] || ctx=$CONTEXT
  [[ "$KV_DTYPE" == fp8 ]] && bytes=17888
  echo $(( bytes * ctx / 1048576 + 503 ))
}
# gpu_need_mib <context> <streams>: free card memory a start needs: the model, every stream (each allocates its own), the reserve. Past the trained window the block scores and a stream's
# row bookkeeping grow with it: 256 + 16 bytes a token.
gpu_need_mib() {
  local ctx=${1:-$CONTEXT} n=${2:-$PARALLEL} reserve
  [[ "$n" =~ ^[0-9]+$ ]] || n=$PARALLEL
  reserve=$(awk -v g="$TENSORFOLD_MEMORY_RESERVE_GIB" 'BEGIN { printf "%d", g * 1024 }')
  local long=0; (( ctx > 262144 )) && long=$(( (ctx - 262144) * 272 / 1048576 ))
  echo $(( WEIGHTS_GIB * 1024 + n * $(stream_mib "$ctx") + long + reserve ))
}
# The card's free MiB and name (nvidia-smi); nothing when the index is wrong or the driver does not answer.
gpu_free_mib() { nvidia-smi -i "$GPU" --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | tr -dc '0-9'; }
gpu_name() { nvidia-smi -i "$GPU" --query-gpu=name --format=csv,noheader 2>/dev/null; }

HF_CACHE="${HF_CACHE:-${HF_HOME:-$HOME/.cache/huggingface}}"
KERNEL_CACHE="${KERNEL_CACHE:-$HOME/.cache/tensorfold-qwen38}"
STATE_DIR="${STATE_DIR:-$HOME/.local/state/qwen38-flash-next-tf}"
LOG_DIR="${LOG_DIR:-$HOME/.cache/tensorfold-qwen38/logs}"
LOG_KEEP="${LOG_KEEP:-10}"
MIN_FREE_GB="${MIN_FREE_GB:-$((CKPT_GIB + 18))}"
IMAGE_FREE_GB="${IMAGE_FREE_GB:-40}"

_c() { [[ -t "$1" ]] && printf '\033[%sm' "$2" || true; }
log()  { printf '%s[%s]%s %s\n' "$(_c 1 '1;36')" "$(basename "$0")" "$(_c 1 0)" "$*"; }
warn() { printf '%s[%s] WARN:%s %s\n' "$(_c 2 '1;33')" "$(basename "$0")" "$(_c 2 0)" "$*" >&2; }
die()  { printf '%s[%s] ERROR:%s %s\n' "$(_c 2 '1;31')" "$(basename "$0")" "$(_c 2 0)" "$*" >&2; exit 1; }

[[ -z "${HF_TOKEN:-}" ]] || export HF_TOKEN
model_cache_dir() { echo "$HF_CACHE/hub/models--${MODEL_ID//\//--}"; }
snapshot_rev() {
  if [[ -n "$MODEL_REVISION" ]]; then echo "$MODEL_REVISION"
  else cat "$(model_cache_dir)/refs/main" 2>/dev/null; fi
}

# The folder under HF_CACHE that holds the checkpoint the engine reads: the snapshot, or with ABLIT=1 the view scripts/ablit_view.py builds
# (the snapshot's files linked, plus the overlay's sidecar). It stays under HF_CACHE because that is all the container mounts.
ABLIT_VIEW_SUB="tensorfold-qwen38/ablit-${MODEL_REVISION:0:8}-${ABLIT_REVISION:0:8}"
ablit_cache_dir() { echo "$HF_CACHE/hub/models--${ABLIT_ID//\//--}"; }
model_sub() {
  if [[ "$ABLIT" == 1 ]]; then echo "$ABLIT_VIEW_SUB"
  else echo "hub/models--${MODEL_ID//\//--}/snapshots/$(snapshot_rev)"; fi
}

PREPARED_MARKER="$STATE_DIR/prepared"
prepared_state() {
  local hash label rev extra=""
  hash=$(image_hash)
  label=$(docker image inspect -f '{{.Id}}' "$IMAGE" 2>/dev/null || echo missing)
  rev=$(snapshot_rev)
  [[ "$ABLIT" != 1 ]] || extra=" ablit=$ABLIT_ID@$ABLIT_REVISION"
  echo "model=$MODEL_ID@$rev image=$label patches=$hash$extra"
}
