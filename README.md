<h1 align="center">Qwen3.8 Flash Next on one RTX PRO 6000 (TensorFold)</h1>

<p align="center">
  <sub>by <a href="https://x.com/MiaAI_lab">Mia's AI Lab</a></sub>
  <br><br>
  <a href="https://github.com/sponsors/MiaAI-Lab" target="_blank" rel="noopener noreferrer" style="display:inline-block;margin:0 8px;vertical-align:middle;"><img src="https://img.shields.io/badge/Sponsor%20me%20on%20GitHub-181717?style=for-the-badge&logo=githubsponsors&logoColor=white" alt="Sponsor me on GitHub" height="28" style="height:28px;width:auto;vertical-align:middle;border:0;" /></a>
  <a href="https://x.com/MiaAI_lab" target="_blank" rel="noopener noreferrer" style="display:inline-block;margin:0 8px;vertical-align:middle;"><img src="https://img.shields.io/badge/Follow%20me%20on%20X-000000?style=for-the-badge&logo=x&logoColor=white" alt="Follow Mia on X" height="28" style="height:28px;width:auto;vertical-align:middle;border:0;" /></a>
</p>

> **NVIDIA's quantization-aware-distilled (QAD) FP4 checkpoint, close to bf16, in its original form.** The model is
> [`nvidia/Qwen3.8-Flash-Next-FP4-Local`](https://huggingface.co/nvidia/Qwen3.8-Flash-Next-FP4-Local): NVIDIA trained the
> NVFP4 and MXFP8 weights against the bf16 model as teacher, so the 4-bit weights are tuned to match it. On NVIDIA's own
> numbers it scores 91.5% on GPQA Diamond and 88.7% on ToolEvalBench against 92.2% and 89.4% for bf16 (model card, not
> measured here). It is what `./start.sh` serves; the [performance](#performance) numbers below are for it.

Serve **Qwen3.8 Flash Next** from one NVIDIA RTX PRO 6000 Blackwell (96 GB, sm_120) through an OpenAI-compatible API, at tensor parallel 1: one rank on one GPU. A second card in
the machine stays free. It runs [TensorFold](https://github.com/ashhart/TensorFold)'s Zig engine, `tensorfold-native` (release
[v1.1.4](https://github.com/ashhart/TensorFold/releases/tag/v1.1.4), commit `eab85828`, with this recipe's patches on top), in NVIDIA's PyTorch container, with image and video input
(`VISION=1`). `VISION=0` serves text only.

The checkpoint is NVIDIA's FP4 QAD export: routed experts in NVFP4 (group size 16, CSF-compressed scale factors),
attention and shared-expert linears in MXFP8, the MTP experts and the vision tower quantized too (MXFP8 attention, W4A16
NVFP4 MLP), an NVFP4 n-gram table, and all 512 routed experts with 10 per token.

- Checkpoint: [`nvidia/Qwen3.8-Flash-Next-FP4-Local`](https://huggingface.co/nvidia/Qwen3.8-Flash-Next-FP4-Local)
  at revision `b9ce18a6887e18c089fdea8509a33e63429a3300` (~96 GiB on disk, 75.92 GiB on the card; the 26.8 GiB n-gram table stays on disk). `ABLIT=1` serves it with an abliteration overlay ([Choosing the checkpoint](#choosing-the-checkpoint)).
- API model id: `Qwen3.8-Flash-Next`
- Context: **262,144 tokens** a request (the native window). **3 requests at once** (`PARALLEL`), each with its own full window. KV is 14.58 GiB of the card's 15.0 GiB left after the
  weights and a 3 GiB reserve, **786,432 tokens in all** ([KV and memory](#kv-and-memory)).
- **YaRN is supported, and off by default.** `YARN=1 ./start.sh restart` trades the three streams for **one request at a time with a window of up to 1,000,000 tokens**, using static YaRN x4 on CUDA (patch `0025`). On this card, on azampatti's 5-expert FP4 cut, a 988,983-token prompt was read in 275 s and a passphrase in it found; it has not run on NVIDIA's checkpoint yet ([Long window](#long-window-yarn1)).
- Prompt cache: **16 GiB of pinned host memory**, about 900,000 tokens of kept states, so a long chat's next turn resumes instead of reading its prompt again (patch `0019`).
- One command: `./start.sh` sets everything up on the first run and starts the server; `./stop.sh` stops it

## Performance

One RTX PRO 6000 with its **power limit at 300 W** (the card's own is 600 W, so these are not its best), on PCIe Gen5 x8 with a Ryzen 9 9900X, nothing else on the GPU. `nvidia/Qwen3.8-Flash-Next-FP4-Local`, 262,144-token
window, 3 requests at once, fp8 KV, vision on, MTP drafts, patches `0001` to `0017` (`0018` to `0025`, added after this run, add video input, move prompt states to host memory, choose one prompt tile by the card's SM count, which can change prefill speed, read the routing widths from the checkpoint, and add the long window). Measured with [sparkDash](https://github.com/MiaAI-Lab/sparkDash)'s own `DecodeBench` and `PrefillBench` code on the same machine, one run each. Prose: temperature 0.7, thinking on. Code: temperature 0,
thinking off. Each reply is 400 tokens. Aggregate is all the streams' tokens over the time from the first token to the last; per request is the median stream; TTFT is the time to the first token.

A decode figure is `(completion_tokens - 1) / (time of the last streamed token - time of the first)`, so TTFT is not in it. The prose prompt asks for a step-by-step explanation of how a hash map works; the code prompts each name a
small Python function or class and end "Output only Python source. No comments, no docstrings, no markdown fences. Then add tests and the helpers this needs. Keep writing code.", and every concurrent stream gets a different task.
MTP drafts pay only when they are accepted, so decode speed depends on the text, and a figure from another harness is comparable only with the same prompts, sampling and thinking setting.

**Decode, prose**

| Concurrent requests | Aggregate | Per request | TTFT |
| ---: | ---: | ---: | ---: |
| 1 | 171.4 tok/s | 171.4 tok/s | 30 ms |
| 2 | 298.5 tok/s | 161.4 tok/s | 104 ms |
| 3 | 406.4 tok/s | 139.9 tok/s | 104 ms |

**Decode, code**

| Concurrent requests | Aggregate | Per request | TTFT |
| ---: | ---: | ---: | ---: |
| 1 | 364.9 tok/s | 364.9 tok/s | 62 ms |
| 2 | 539.5 tok/s | 280.0 tok/s | 73 ms |
| 3 | 647.8 tok/s | 222.8 tok/s | 69 ms |

**Prefill** (cold prompts, no prompt-cache hit)

| Prompt tokens | Prefill | TTFT |
| ---: | ---: | ---: |
| 4,130 | 6,044 tok/s | 0.75 s |
| 8,231 | 5,910 tok/s | 1.46 s |
| 16,427 | 5,762 tok/s | 2.92 s |
| 32,804 | 5,637 tok/s | 5.88 s |
| 65,575 | 5,437 tok/s | 12.13 s |
| 131,110 | 5,184 tok/s | 25.36 s |

**The speed settings have not been measured on this card yet.** The tile, thread and draft defaults in `scripts/config.sh` were carried over from a different GPU; they change speed only, never a reply. `tools/sweep.py` compares them here. Patch `0023` already chooses the fused read-out's down tile from this card's 188 SMs.

### Checks on this card

With patches `0001` to `0017` on GPU 0, defaults, all passing:

- `tools/exact.py --pad 3000`: six drafted greedy replies (538 drafts accepted) equal the `"draft": false` ones, and six concurrent replies equal the one-at-a-time ones.
- `tools/toolcheck.py` 3/3, `tools/visioncheck.py`, `tools/context_boundary.py --context 262144` (16 checks), `tools/prompt_reuse.py 30000` (worst 99.3% resumed).
- A 199,730-token needle in a haystack is found, with a prefill of 39.05 s.
- Three concurrent prompts of about 231,700 tokens each name their own passphrase. The card's memory peaked at 92,976 MiB of 97,887.

With patches `0001` to `0019`:

- Images: PNG, JPEG, WebP (lossy and lossless), 12 images in one request (12 of 12 colours), a 4,122-token image read correctly (3.2 s warm), `"detail": "low"`, 50 images in one request (16,340 tokens; first and last told apart; the card peaked at 79,677 MiB), and an image chat's second turn resuming.
- Prompt states in host memory, in about 3 hours of real use: 2,462 requests, prompts up to 156,754 tokens, 95.5% of all prompt tokens resumed (2,415 hits, 50 misses, 0 refused, 0 failed in the store's counters). Before patch `0019`, `tools/prompt_reuse.py 60000` resumed 2.5% of its prompt and read it again each turn (11.6 s).

With patches `0001` to `0020`, all passing:

- `tools/exact.py --pad 3000` (538 drafts accepted), `tools/toolcheck.py` 3/3, `tools/visioncheck.py`, `tools/context_boundary.py --context 262144` (16 checks), and every image check above again.
- `tools/prompt_reuse.py 60000`: worst 99.6% resumed, 0.5 to 0.6 s a follow-up turn against 11.9 s cold.
- Resumed replies equal cold ones, byte for byte (a one-off greedy check at 2,271, 33,643 and 101,324 tokens): resumed alone, after three other prompts took the streams, and beside two running requests, plus a repeated follow-up turn.
- Three concurrent prompts of about 225,900 tokens each name their own passphrase. The card's memory peaked at 92,487 MiB of 97,887.
- Video: MP4, WebM, MKV and MOV clips answered correctly. A 60-second 1080p clip (7,754 tokens) alone and three at once, and four clips in one request (15,910 tokens, near the 16,384 budget); the card peaked at 79,703 MiB. Five videos and a file over 16 MB are refused.
- TensorFold's `tools/zig/video_served.py` against transformers 5.20.2: prompt ids equal on 12 of 12 clips, a later turn resumes past its video, drafted equals `"draft": false`, concurrent equals alone. Two of its checks fail here because they expect GLM's limits: a refusal of three videos (`MAX_VIDEOS` is 4 here) and of a 0.4-second clip (Qwen's processor samples it, and the server's prompt ids equal the processor's).

With all 25 patches, on NVIDIA's checkpoint, all passing: the six greedy replies of `tools/exact.py --pad 3000` (537 drafts accepted) are identical, token for token, to those on patches `0001` to `0020`; `tools/context_boundary.py --context 262144` (16 checks), `tools/toolcheck.py` 3/3, `tools/visioncheck.py`, `tools/videocheck.py`, `tools/prompt_reuse.py 60000` (worst 99.6% resumed), and a 199,730-token needle in a haystack (prefill 42.56 s).

These ran only on azampatti's 5-expert FP4 cut (AZ), which this recipe no longer serves, not on NVIDIA's checkpoint with these patches: forced and named tool calls, a JSON schema and JSON mode, `tools/resume_exact.py`, and `tools/full_windows.py` (three prompts of about 225,900 tokens, the card's peak 92,647 MiB). The checks with YaRN on are under [Long window](#long-window-yarn1). The engine's 28 host test steps pass with these patches, as they do on stock v1.1.4; on 13 of 13 clips the prompt `0020` serves, without the outer pair `0022` removes, equals transformers 5.8.0's ids; and the engine's config check reads 10 experts and a 640-wide shared expert from NVIDIA's real `config.json`.

## Requirements

- Linux x86_64 with one RTX PRO 6000 Blackwell (96 GB, sm_120) and nothing else large on it: 3 full windows need 96,845 MiB free at start, which an empty card has (97,322 MiB here) and a card driving a desktop may not. `start.sh` stops before
  launching below that (`GPU_NEED_MIB`). `GPU` picks the card by its `nvidia-smi` index. On a machine with two cards, take the one without the monitors: on the other one use `PARALLEL=2`.
- A driver for CUDA 13, Docker with the NVIDIA container runtime, and your user in the `docker` group.
- Host memory: the weights stage through it while they load (`MemAvailable` stayed near 71 GiB of 89 during a load here), and the prompt cache pins up to 16 GiB (`PROMPT_CACHE_GIB`, at most half of what is available at start) as states are kept. Developed on 89 GB of RAM.
- ~160 GB free disk on a fresh machine: ~96 GiB for the checkpoint under `~/.cache/huggingface` and the image under
  Docker's root. `scripts/prepare.sh` checks both. An NVMe drive for the checkpoint: the host reads n-gram rows from it while serving.
- Optional: the `hf` CLI on the host and a Hugging Face token in `~/.cache/huggingface/token` or `HF_TOKEN`. The
  checkpoint is public. `ABLIT=1` (below) needs a token, because its overlay is gated.

## Quick start

From this checkout:

```bash
./start.sh
```

The first run builds the engine (about 5 minutes: TensorFold v1.1.4 plus `patches/`, with FFmpeg for video; the kernels are embedded in the binary) and
downloads the checkpoint if it is not already in the cache. Later starts load the weights in about 2.5 minutes. `start.sh` shows each
step, runs a smoke test, prints `Qwen3.8-Flash-Next is now LIVE! on port 8888`, and returns you to the shell.

```bash
curl -s http://<host>:8888/v1/models

curl -s http://<host>:8888/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "Qwen3.8-Flash-Next",
  "messages": [{"role": "user", "content": "Write a Python fibonacci function."}],
  "max_tokens": 1000
}'
```

Any OpenAI client works with `base_url = "http://<host>:8888/v1"` and the model `Qwen3.8-Flash-Next`.
Streaming, tool calls and reasoning content are supported. The model thinks before it answers, so give replies
enough `max_tokens`. `"draft": false` on a request serves one token at a time; a drafted reply matches that.

```bash
./start.sh restart
./stop.sh
docker logs -f qwen38-flash-next-tf
curl -s http://<host>:8888/health
```

## Choosing the checkpoint

This recipe serves two things:

| Start with | It serves |
| --- | --- |
| `./start.sh` | NVIDIA's FP4 QAD checkpoint, `nvidia/Qwen3.8-Flash-Next-FP4-Local` at revision `b9ce18a6887e18c089fdea8509a33e63429a3300`, in its original form: 10 routed experts a token and a 640-wide shared expert |
| `ABLIT=1 ./start.sh` | the same checkpoint with the Keys abliteration overlay grafted in ([ABLIT](#ablit-opt-in)) |

NVIDIA's checkpoint runs on the current image ([checks](#checks-on-this-card)). **`ABLIT=1` has never started on this card.**

FP4 is only served in its original form: `AZ=1` and `CHECKPOINT=az`, azampatti's cut of it to 5 experts a token, stop with a message. `CHECKPOINT=int4` stops too: [`azampatti/Qwen3.8-Flash-Next-125B-A5B-INT4-AutoRound`](https://huggingface.co/azampatti/Qwen3.8-Flash-Next-125B-A5B-INT4-AutoRound) stores its routed experts and head as GPTQ int4 and its attention as block FP8, and TensorFold v1.1.4 has no kernels for GPTQ.

Check which checkpoint is running with `docker inspect -f '{{.Config.Cmd}}' qwen38-flash-next-tf`: the path after `serve` names it. `DRY_RUN=1 ./start.sh`
prints what a start would do (the model, the memory it needs and the docker command) without stopping or starting anything. To serve a different model directory of the same
format, set `MODEL_ID` (and `MODEL_REVISION`, or leave it empty for the latest) instead of `CHECKPOINT`. A `MODEL_ID` set by hand takes no preset revision pin.

## ABLIT (opt-in)

`ABLIT=1` serves NVIDIA's FP4 checkpoint with [drowzeys's Keys "house-qsa" abliteration overlay](https://huggingface.co/drowzeys/keys-Qwen3.8-Flash-Next-NVFP4-FP4-QAD-house-qsa-L3-47)
grafted in. The overlay replaces 24 tensors: `self_attn.o_proj` (weight and scale, MXFP8) of the 12 attention (QSA) layers (L3, 7, 11, ... 47), edited by a
rank-1 refusal-direction projection and requantized. DeltaNet `out_proj`, every expert, the MTP head, the n-gram table, the vision tower and the chat template stay NVIDIA's
(see the overlay's own `ABLIT_META.json`). The model has had its safety refusals removed; read the overlay's Responsible Use Agreement, which you accept to download it.

The overlay is gated, free and approved automatically. Accept its terms at its Hugging Face page while logged in, then `export HF_TOKEN=hf_...` or `hf auth login`
with that account, before:

```bash
ABLIT=1 ./start.sh restart
```

Set `ABLIT=0` (the default) to return to the stock checkpoint. The first `ABLIT=1` start downloads the overlay's ~190 MB sidecar; the checkpoint itself is not downloaded
again and is not modified. The replaced tensors have the stock tensors' dtypes and shapes, so the memory a start needs is the same; speed has not been measured with the overlay.

How it is applied: `scripts/prepare.sh` runs [`scripts/ablit_view.py`](scripts/ablit_view.py), which builds a folder `$HF_CACHE/tensorfold-qwen38/ablit-b9ce18a6-c44f5b92` holding
symlinks to every file of the stock snapshot, the sidecar as `tensors/house-qsa-oproj-mxfp8-L3-47.safetensors`, and a `model.safetensors.index.json` that points the 24 tensor
names at it. TensorFold's engine reads the shards its index names, in path order, and takes a tensor from the first shard that holds it; the sidecar's path sorts before the
stock `tensors/model-*` shards. The script refuses to build the folder if the sidecar does not hold exactly those 24 names, if a stock tensor has another dtype or shape, or if the
sidecar would not sort first. It only works for `CHECKPOINT=nvfp4` at the pinned revision (`ABLIT=1` with another `MODEL_ID` or `MODEL_REVISION` stops with a message). Check it
is active with `docker inspect -f '{{.Config.Cmd}}' qwen38-flash-next-tf`: the path after `serve` ends in `ablit-b9ce18a6-c44f5b92`.

**What has been checked.** The folder builder against the real sidecar and the real stock index and shard headers (24 of 24 names, same dtypes and shapes), and offline tests
(`tools/test_ablit_view.py`). `ABLIT=1` has **not** been started on this card, and `tools/exact.py`, `tools/needle.py` and a refusal check on it have not been run. The overlay author's own
numbers (refusal32 32/32 on a thinking-off run of their TensorFold recipe) are theirs, not measured here. The edit changes the weights, so replies differ from the stock checkpoint's: the byte-identical
checks in this repository compare a build against itself, not `ABLIT=1` against `ABLIT=0`.

## Images and video

The model's own vision tower (27 layers, from the same checkpoint) runs in the engine, on the card, and turns images and video frames into
tokens. Images are decoded with libjpeg-turbo and libwebp, videos with `libtfvideo` over FFmpeg 9.0.2. Send them as OpenAI-style content parts in a user message:

```bash
IMG=$(base64 -w0 photo.jpg)
curl -s http://<host>:8888/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "Qwen3.8-Flash-Next",
  "messages": [{"role": "user", "content": [
    {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,'"$IMG"'"}},
    {"type": "text", "text": "What is in this picture?"}]}],
  "max_tokens": 2000
}'
```

A video is a `video_url` part (`{"type": "video_url", "video_url": {"url": "data:video/mp4;base64,..."}}`).

| | Images | Videos |
| --- | --- | --- |
| Formats | JPEG, PNG, WebP | MP4, WebM, MOV, MKV holding H.264, HEVC, VP8, VP9 or MPEG-4 |
| Per request | up to 50 (all of a chat's turns count), 10 MB each, 20 MB in all | up to 4 (`MAX_VIDEOS`), 16 MB each, 20 MB in all, up to an hour of footage, 8192 pixels a side |
| Tokens | up to 16,384 for all images, at most 4,096 an image (`"detail": "low"`: 256 an image) | 2 frames a second, up to 16,384 tokens a request (`TENSORFOLD_VIDEO_TOKENS`) |

By default only data URLs are accepted. `VISION_URLS=1` also lets the server fetch public `https://` URLs. Image prompts are kept for prefix reuse
(each image matched by a hash of its bytes), so a chat's next turn after an image prefills only its new rows, but the vision tower still runs for every image each turn.  `VISION=0 ./start.sh restart` serves text only and gives the tower's memory back to the streams.

Measured on this card: a 153-token image prompt 0.13 s to its reply, 12 images (819 tokens) 0.37 s, a 4,122-token image 3.2 s (4.0 s on first use), 50 images (16,340 tokens) 5.1 s.

**Video on this engine.** Stock TensorFold v1.1.4 refuses a `video_url` part on CUDA. Patches `0018`, `0020` and `0022` and the image's `libtfvideo` add it: each pair of frames goes through the tower as one group, and the prompt is laid out as transformers 5.8.0's processor does it, the release nearest the `5.8.0.dev0` the checkpoint names: `<t seconds><|vision_start|>` the group's rows `<|vision_end|>` a group, without the pair of markers the template writes around the video. (transformers 5.20.2 keeps that outer pair, so its prompt ids are longer by two tokens a video.) A video's rows share the 84 MiB a stream's vision rows take. The results are under [Checks on this card](#checks-on-this-card); `tools/videocheck.py` repeats them.

## What `start.sh` and `scripts/prepare.sh` do

1. **Setup:** `scripts/prepare.sh` when the image or the checkpoint is not the one these settings name.
2. **Checks:** the arguments (tensorfold-native's own parser, in a throwaway container), the port, the card's free memory (it names what holds the card) and the host's.
   A typo leaves a running server alone. `./start.sh restart` stops only after those checks pass.
3. **Launch:** `tensorfold-native serve` with the settings from `scripts/config.sh`.
4. **Loading:** the log, and a stop if the host's `MemAvailable` falls below `MEM_FLOOR_GIB` (4 GiB).
5. **Smoke test**, then the LIVE message.

`FOREGROUND=1 ./start.sh` stays attached and exits with the container's exit code. Extra arguments go to
`tensorfold-native serve` after the defaults (`./start.sh restart --context 131072`). `DRY_RUN=1 ./start.sh` prints
the docker command and changes nothing.

**`scripts/prepare.sh`:**

1. Preflight: x86_64, Docker, the GPU, disk.
2. The image `tensorfold-qwen38:v1.1.4-eab8582-sm120`: TensorFold v1.1.4 (`TF_REF`) with every `patches/*.patch` applied, Zig 0.17.0 for x86_64, built for sm_120 (`CUDA_SM`) and this machine's CPU (`ZIG_CPU`; set `x86_64_v3` for an image that moves), libjpeg-turbo, libwebp's decoder and FFmpeg 9.0.2
   (built from checksummed sources), on `nvcr.io/nvidia/pytorch:26.07-py3`; the build runs `fatbins native` and `video`. `libtfgrammar.so` is copied out of TensorFold's v1.1.4 x86_64 release archive
   (SHA-256 pinned in `config.sh`). No image is published (`GHCR_IMAGE` is empty, so `PULL` does nothing until you set one).
3. The checkpoint, at the pinned revision.
4. A check that it is a `qwen4_exp` ModelOpt NVFP4 checkpoint and every shard present.
5. With `ABLIT=1` only: the overlay's sidecar from Hugging Face, and the checkpoint folder with it grafted in ([ABLIT](#ablit-opt-in)).

```bash
scripts/prepare.sh
scripts/prepare.sh --rebuild
PREPARE=1 ./start.sh restart
```

## KV and memory

The card and the host are separate memories. Every weight the kernels read is on the card; the n-gram table stays on disk, and the host reads its rows.

| On the card, measured (vision on, `PARALLEL=3`) | GiB |
| --- | ---: |
| Free on an empty card (of 95.6) | 95.04 |
| The model: weights, tower, scratch | 75.92 |
| Other engine allocations the budget counts | ~1.1 |
| Reserve (`TENSORFOLD_MEMORY_RESERVE_GIB`) | 3.0 |
| **Left for streams** | **15.0** |
| 3 streams at 262,144 tokens, 4.86 each | 14.58 |
| Unused | 0.42 |

**Total KV: 786,432 tokens, three full windows.** At start the engine counts a full window for every stream and refuses a `PARALLEL` that does not fit. A fourth full window does not (19.4 GiB, and the smallest reserve, 0.5 GiB, leaves 17.5). The memory itself is taken per request: its prompt, its `max_tokens` and a few draft rows, returned afterwards. So `nvidia-smi` shows free memory until the windows fill, and that memory is not spare: three full windows took the card to 92,647 MiB. One request reaches 262,144 tokens at most. Patch `0020` added 80 MiB a stream for video rows: the engine reported 4.78 GiB a stream before it and reports 4.86 GiB now.

KV is FP8 by default (`KV_DTYPE=fp8`). It is lossy: about 98.8% top-1 agreement with a bf16 cache, so a free-running reply can differ from the bf16 one. `KV_DTYPE=bf16` is the exact cache; at about 7.9 GiB a window, one stream fits beside the weights (calculated, not run). The format is adapted from MiaAI-Lab's GLM recipe patch `0038-glm-kv-fp8`.

One token's cache, 13 layers (12 attention plus the MTP head):

| | bytes a token | one 262,144-token sequence |
| --- | ---: | ---: |
| bf16 | 30,784 | 7.52 GiB |
| fp8 | 17,888 | 4.37 GiB |

**Prompt cache.** A chat's next turn resumes from a state kept at the end of its last one. Beside three full windows the card has under 0.5 GiB for those states, less than one 30,000-token chat: kept on the card, `tools/prompt_reuse.py 60000` resumed 2.5% of its prompt and spent 11.6 s reading it again every turn, and 24 s at 120,000 tokens. So patch `0019` keeps them in pinned host memory, the same idea as the host tier of MiaAI-Lab's GLM recipe on RTX PRO 6000. A state costs about 18 KB a token. In real use a 93,338-token turn that resumed 99% of its prompt reached its first token in 0.81 s.
`PROMPT_CACHE_GIB=16` is the most it pins, taken as states are kept and at most half of the host's `MemAvailable` at start. Each turn keeps a state, so a long chat's older ones fill it and are evicted, oldest first. `--prompt-cache-gib 0` (`PROMPT_CACHE_GIB=0`) turns it off, and `TF_FLASHNEXT_HOST_SNAPS=0` keeps states on the card.

<a id="long-window-yarn1"></a>**Long window (`YARN=1`).** `YARN=1 ./start.sh restart` serves one request at a time with a window of 1,000,000 tokens (patch `0025`). Any `CONTEXT` past 262,144, up to 1,048,576, asks for the same. Past the trained window every prompt rotates with static YaRN x4, the method and table TensorFold's Metal engine uses (the table is `transformers`' `_compute_yarn_parameters`, equal to 5.8.0's in all 32 frequencies and the scale, bit for bit). It applies to every prompt, short ones too. Qwen recommends YaRN only where long prompts need it, since static scaling can cost short texts, so it is not the default, and at 262,144 or less the rotary is the plain one with its scale at exactly 1.

Measured on this card at 300 W, in one run with a few requests from another machine mixed in. **All of it is on azampatti's 5-expert FP4 cut (AZ), which this recipe no longer serves; YaRN has not run on NVIDIA's checkpoint.** NVIDIA's model is about 0.09 GiB smaller, so it should fit with a little more room. The server loaded in 134 s.

| On the card with `YARN=1`, on AZ | GiB |
| --- | ---: |
| Free on an empty card | 95.04 |
| The model, with its block scores grown for the longer window | 76.18 |
| The engine's other allocations | ~1.2 |
| Reserve, the engine's smallest | 0.50 |
| **Left for the stream** | **17.2** |
| One stream at 1,000,000 tokens | 17.17 |

The engine's check passes by about 0.03 GiB, and with a prompt of about 990,000 tokens on the card 785 MiB of it was free, so nothing else should use this card. A margin that thin means a longer window does not fit with drafts on: 1,048,576 tokens needs `YARN=1 CONTEXT=1048576 DRAFTS=0`, which drops the MTP head's cache and its speed-up (calculated, not run). `PROMPT_CACHE_GIB` goes to 20 because a state of 1,000,000 tokens takes up to about 18 GiB and a chat resumes only from a state kept whole. It needs 40 GiB of `MemAvailable` at start, since the cache takes at most half of it. `YARN` lasts for that start: a later `./start.sh restart` without it returns to 262,144 tokens and 3 streams.

| Cold prompt, tokens | Prefill | Read in | Result |
| ---: | ---: | ---: | --- |
| 199,730 | 5,379 tok/s | 37.1 s | passphrase found |
| 301,332 | 5,068 tok/s | 59.5 s | follow-up turns resumed 99.9% of it, 1.3 s each |
| 549,405 | 4,440 tok/s | 123.7 s | passphrase found |
| 988,983 | 3,599 tok/s | 274.8 s | passphrase found |

`tools/yarn_checks.sh LOG` runs every check against such a server, in about 20 minutes. Passing with YaRN on: `tools/exact.py --pad 3000` (547 drafts accepted) and `--pad 300k --prompts 2` (200 accepted, replies past the trained window equal with drafts on and off), `tools/resume_exact.py`, `tools/prompt_reuse.py 300000` (worst 99.9% resumed), `tools/context_boundary.py --context 1000000` (16 checks), `tools/toolcheck.py` 3/3, `tools/visioncheck.py`, `tools/videocheck.py`, and the passphrases above, each 60% of the way into its prompt. With one stream, `exact.py`'s concurrent requests queue, so batching is not exercised. Three prompt-cache restores were refused (kept states of 2 and 13 tokens); the engine read those prompts from the start, which costs time only.

Not checked: a passphrase 5% of the way into a 990,000-token prompt (the last check in the script, stopped before it finished), resuming a state past about 302,000 tokens, and what static scaling costs short texts, which Qwen warns of. YaRN does change replies at short lengths: on six 3,000-token prompts, four of `exact.py`'s greedy replies differ from the plain rotary's. That shows they differ, not which is better.

**More streams** take a shorter window: `PARALLEL=4 CONTEXT=192000` fits by the sizes above, and `PARALLEL=4 CONTEXT=230000 TENSORFOLD_MEMORY_RESERVE_GIB=0.5` with the smallest reserve (both calculated, not run). Four windows of 256,000 tokens or more do not fit: 19.02 GiB against 17.5. A request is admitted when its prompt plus its reply budget (`max_tokens`) fits its window; a request past `PARALLEL` waits for a stream.

**The reserve** is what the engine leaves free for what it does not count, such as a desktop on the same card. Three full windows at work left about 4.2 GiB of the card free. The GLM recipe's notes record out-of-memory errors under long sessions with small reserves when the desktop shared the card, so raise it on such a card.


## Configuration

Every setting lives in [`scripts/config.sh`](scripts/config.sh). Override it from the environment
(`PARALLEL=2 ./start.sh`; the memory a `PARALLEL` needs is explained under [KV and memory](#kv-and-memory)) or from `.env` next to `start.sh` (`KEY=value`; the environment wins).

| Variable | Default | Meaning |
| --- | --- | --- |
| `GPU` | `0` | the card, by its `nvidia-smi` index. TP=1: one card |
| `PARALLEL` | `3` (`1` with `YARN=1`) | requests decoded together, each with a full window; a refused start past what fits |
| `CONTEXT` | `262144` (`1000000` with `YARN=1`) | prompt + reply window. Past 262,144, up to 1,048,576, every prompt rotates with static YaRN x4 |
| `YARN` | `0` | `1`: a long window on one stream, with the settings marked here ([Long window](#long-window-yarn1)) |
| `DRAFTS` | `1` | MTP drafts; `0` passes `--no-drafts` |
| `TEMPERATURE` / `TOP_P` / `TOP_K` | `1.0` / `0.95` / `20` | default sampling; a request's own values win |
| `THINKING` | `1` | think before answering; `0` passes `--no-thinking` |
| `MAX_TOKENS` | `32768` | the reply budget when a request sets none |
| `VISION` / `VISION_URLS` | `1` / `0` | image and video (`--vision`); `VISION_URLS=1` also accepts public `https://` URLs |
| `TENSORFOLD_MAX_IMAGES` / `TENSORFOLD_IMAGE_TOKENS` | `50` / `16384` | images a request may carry, and the tokens they share (4,096 at most an image) |
| `MAX_VIDEOS` | `4` | videos a request (`--vision-max-videos`) |
| `TENSORFOLD_VIDEO_TOKENS` | `16384` | a request's video token budget (`--vision-video-tokens`) |
| `SERVED_NAME` | `Qwen3.8-Flash-Next` | the model id in `/v1/models` |
| `PORT` / `HOST` | `8888` / `0.0.0.0` | where the API listens |
| `KV_DTYPE` | `fp8` | lossy (~98.8% top-1 agreement with bf16, about 1.8x the pool). `bf16` is exact. Works with `VISION=1`. Adapted from MiaAI-Lab's GLM recipe patch `0038-glm-kv-fp8` |
| `TENSORFOLD_MEMORY_RESERVE_GIB` | `3` (`0.5` with `YARN=1`) | kept free on the card when the engine sizes the streams (the engine's own default for 96 GB) |
| `PROMPT_CACHE_GIB` | `16` (`20` with `YARN=1`) | the most pinned host memory the kept prompt states take (patch `0019`); `0` is none |
| `TF_FLASHNEXT_HOST_SNAPS` | `1` | `0` keeps prompt states on the card, in what the streams leave of it |
| `CUDA_SM` / `ZIG_CPU` | `120` / `native` | the kernels' SASS target and the host code's CPU target (both in the image hash) |
| `GPU_NEED_MIB` / `MEM_FLOOR_GIB` / `MEM_CHECK` | computed / `4` / `1` | free card memory a start needs; the host memory a load stops at; `MEM_CHECK=0` turns both into warnings |
| `TF_FLASHNEXT_DRAFTS` | `6` | drafts a round at most |
| `TF_FLASHNEXT_CONFIDENCE` / `TF_FLASHNEXT_PRODUCT_STREAMS` | `-0.4` / `5` | the draft chain keeps draft j while the running product of the draft probabilities is at least 0.4, for up to 5 streams drafting together (0.5 per draft above that) (patch 0017). Only the number of drafts changes, never a reply. `TF_FLASHNEXT_CONFIDENCE=0.6` restores v1.1.4's per-draft rule |
| `TF_FLASHNEXT_SAMPLED_HEADS` / `TF_FLASHNEXT_SAMPLE_CAND` | `1` / `1` | sampled requests join the batched MTP head pass, and their top-k candidates are found on the GPU (patches 0007, 0008) |
| `TF_FLASHNEXT_HOST_DRAW_THREADS` | `7` | worker threads for the host-side sampled draws (patch 0011); bit-identical to one thread |
| `TF_FLASHNEXT_KEEP_LATE` | `1` | a partly rejected draft window does not replay the DeltaNet state: the kept rows wait in a log and the next verify pass applies them (patch 0012) |
| `TF_FLASHNEXT_GATHER_EARLY` / `TF_FLASHNEXT_GATHER_READ` | `1` / `pread` | the n-gram rows of all lanes are gathered in one pass, early, and read with a `pread` thread pool instead of an `mmap` that page-faults (patches 0010, 0013). `mmap` restores the stock reader |
| `TF_FLASHNEXT_DRAFT_AHEAD` / `TF_FLASHNEXT_WARM_AHEAD` | `1` / `1` | read ahead the n-gram rows of the draft tokens during the head step, and of the next prompt chunk during the current one |
| `TF_FLASHNEXT_PREFILL_TAIL` | `512` | the last two prompt chunks are evened out when the last would be under this many rows |
| `TF_FLASHNEXT_PROMPT_HC2` / `TF_FLASHNEXT_PROMPT_SHARED_DIRECT` | `1` / `1` | prompt passes run the hyper-connection read-out as two fused kernels instead of five, and the shared expert's rows go straight into the output slot (patch 0014). Bit-identical |
| `TF_FLASHNEXT_PROMPT_EXPERT_TILE_ROWS` / `TF_FLASHNEXT_HEAD_PIECE` | `256` / `1536` | the routed experts of prompt chunks use the shared-memory row tiles from this many rows; the MTP head takes prompt pieces of this many rows (patch 0014) |
| `TF_FLASHNEXT_PROMPT_GEMM2` | `1` | the MX128 prompt tile runs with two pipeline stages and two blocks an SM (patch 0015). Same bits |
| `TF_FLASHNEXT_SHARED_SIDE` / `TF_FLASHNEXT_GDN_LANES` | `1` / `1` | the shared expert runs on a second stream in decode rounds, and a shared round takes every lane's DeltaNet rows in three launches a layer (needs `KEEP_LATE=1`) (patch 0016). Bit-identical |
| `CHECKPOINT` | `nvfp4` | `nvfp4` serves `nvidia/Qwen3.8-Flash-Next-FP4-Local`. `az` and `int4` stop with a message. `MODEL_ID` / `MODEL_REVISION` override the model |
| `MODEL_REVISION` | `b9ce18a6…` | the checkpoint commit this recipe serves |
| `ABLIT` | `0` | `1` serves NVIDIA's checkpoint with drowzeys's Keys house-qsa abliteration overlay ([above](#ablit-opt-in)); gated, needs `HF_TOKEN`. Only with `CHECKPOINT=nvfp4` at the pinned revision |

KV is fp8 (or bf16). The checkpoint's NVFP4 n-gram table (26.8 GiB) is read from disk by the host
(`TF_FLASHNEXT_GATHER_READ=pread` by default, with read-ahead of the next prompt chunk; `TF_FLASHNEXT_WARM_AHEAD=0` turns that off).

### Thinking and sampling

By default the model thinks first. Per request, `temperature`, `top_p`, `top_k` and `seed` override the defaults
(`temperature: 0` is greedy). `"chat_template_kwargs": {"enable_thinking": false}` answers without thinking, and
`"chat_template_kwargs": {"reasoning_effort": "low"}` (or `"medium"`, `"xhigh"`) sets reasoning effort. The reasoning
comes back in `reasoning_content`, the answer in `content`. A small `max_tokens` can return empty `content` because
the model was still thinking.

- **A reply that spends its whole `max_tokens` thinking:** `thinking_budget` (a request field; `--thinking-budget N` as a server default, for example
  `./start.sh restart --thinking-budget 4096`) closes the think block after that many reply tokens and lets the answer start
  (`think_budget` in `zig/src/core/lanes/stream.zig`; the closing is done in the engine's lanes, so drafted and serial replies cut at the same token).
  `reasoning_effort: low` and `enable_thinking: false` are the other levers.
- **Without a `seed`, a sampled request draws a fresh one** (TensorFold v1.1.4's default, as OpenAI's API does); an explicit `seed` still wins. For reproducible
  replies send a `seed`, or start the server with `--prompt-seed` (the engine then derives the seed from the prompt's token ids, so the same prompt gives the same
  sampled reply, and `"draft": false` comparisons are reproducible; `tools/exact.py` and the sampled-exactness checks run that way). `TENSORFOLD_SEED_SALT=<integer>`
  changes every derived seed at once.
- **`presence_penalty`, `frequency_penalty` and `repetition_penalty` are accepted and have no effect** (the engine does not implement them), and out-of-range
  sampling values such as `temperature: -1` are not rejected. Use `top_p`, `top_k`, `min_p` and `temperature` to steer repetition.
- **A control token in the reasoning ends the turn.** The engine stops on the checkpoint's end tokens (`<|endoftext|>` 248044 and `<|im_end|>` 248046 in its
  `eos_token_id`; the report saw `<|im_start|>` 248045 stop a reply too). If the model writes one while it is still thinking, for example when it quotes it in its analysis, the reply stops right there
  with `finish_reason: "stop"`, empty `content` and a cut `reasoning_content`.
  A client can treat a `stop` with empty `content` as a failed turn and retry with `enable_thinking: false`.
- **`tool_choice: "required"`, a named function and structured output** are enforced by the engine's grammar library, `libtfgrammar.so` (xgrammar), which the image takes
  prebuilt from TensorFold's v1.1.4 release archive. With thinking off, `required` still gives the model no room to reason before the call: the grammar forces a call from the
  first token, and the function is then picked from that start. With many tools the pick can be less relevant than under `auto`. Keep thinking on, use `auto` with an
  instruction to call a tool, or name the function in `tool_choice`.

### API notes

- Endpoints: `/v1/chat/completions`, `/v1/completions`, `/v1/responses`, `/v1/messages` (Anthropic) and
  `/v1/messages/count_tokens`, `/v1/models`, `/tokenize` and `/detokenize` (also under `/v1/`), `/health`
  and Prometheus `/metrics` (stage timers, time per output token, prefix-cache and speculative-decoding counters).
- Refused with HTTP 400: `logprobs: true` / `top_logprobs` (no token probabilities) and `n` other than 1.
- `"draft": false` serves the request without drafts, the serial reference.
- Tool calls work, including `tool_choice: "required"`, a named function and Anthropic's `{"type": "any"}`: the reply then always opens a tool call.
- A conversation's next turn resumes from the kept state of its earlier turns, with images too (built into TensorFold v1.1.4: a changed
  image resumes only up to its first row). A video's turns resume the same way.
- Structured outputs (`response_format` with a JSON schema, the `guided_*` fields) are enforced with `libtfgrammar.so`; a reply follows the schema. Without the library (a hand-run
  `tensorfold-native` that lacks it) the server answers HTTP 400.
- A request whose prompt plus `max_tokens` does not fit its window is refused. One past `PARALLEL` waits for a stream. Kept prompt states are in host memory and take no room from the streams. A request whose prompt plus `max_tokens` equals `--context` is admitted: the draft rows sit past that
  window (checked with `tools/context_boundary.py`; a streamed request over the limit is refused with an error inside the stream). That admission change was authored by [321sssrt-bit](https://github.com/321sssrt-bit)
  ([issue #1](https://github.com/MiaAI-Lab/Qwen3.8-Flash-Dual-DGX-Sparks-TensorFold/issues/1),
  [PR #2](https://github.com/MiaAI-Lab/Qwen3.8-Flash-Dual-DGX-Sparks-TensorFold/pull/2)).

## What the patches are

`patches/*.patch` are this recipe's changes to TensorFold v1.1.4 (commit `eab85828`, which brought its own Flash Next CUDA engine), applied with `git apply`
in the checkout root by `scripts/prepare.sh`. The earlier series for TensorFold `db281878` (patches `0001` to `0031`) is retired, since v1.1.4 replaced the engine it
targeted. Drafted replies are still checked: a drafted reply matches `"draft": false`.

| Patch | What it does |
| --- | --- |
| `0001`, `0002` | the vision tower loads from NVIDIA's quantized weights (MXFP8 attention, NVFP4 MLP and merger decoded to bf16 at load); image budget 16,384 tokens |
| `0003` | `--kv-cache fp8` for Flash Next: 17,888 bytes a token against bf16's 30,784, so three full windows fit on the card where bf16 fits one |
| `0004`, `0006` | prompt expert tile and prefill work: no drain between prompt chunks, vectorised hyper-connection kernels, a 128x128 MXFP8 prompt tile, 3,072-row chunks |
| `0005` | the fp8 attention reader keeps the next tile in registers; the MTP head's cache stays bf16 |
| `0007` to `0011` | sampled requests join the batched draft pass, GPU top-k candidates, a round-cost model (off), n-gram rows of all lanes gathered at once, host draws on worker threads |
| `0012`, `0013` | a partly rejected draft window no longer replays the DeltaNet state; the n-gram table is read with `pread` and read ahead instead of through an `mmap` |
| `0014`, `0015` | fused hyper-connection kernels, expert row tiles and MTP head pieces for prompt passes; a two-stage MX128 prompt tile |
| `0016` | the shared expert runs on a second stream in decode rounds; a shared round takes every lane's DeltaNet rows in three launches a layer |
| `0017` | the draft chain stops on the running product of its draft probabilities (the rule of the earlier build) |
| `0018` | `libtfvideo` links on x86_64: only its own API leaves the library, or FFmpeg's x86 assembly is refused by the linker |
| `0019` | kept prompt states go to pinned host memory instead of what the streams leave of the card |
| `0020` | video input on the Flash Next CUDA path: each frame pair one group through the tower |
| `0021` | the engine's host tests set the fp8 patch's KV format, so its whole suite passes |
| `0022` | a video's prompt drops the markers the template writes around it, as the processor the checkpoint was made with does |
| `0023` | the fused read-out's down tile is chosen from the card's SM count, read from the device (it was fixed at 48) |
| `0024` | a ModelOpt checkpoint's routed experts a token and shared-expert width come from `config.json`, written for azampatti's AZ cut, no longer served; NVIDIA's keeps 10 and 640 |
| `0025` | `--context` past 262,144, up to 1,048,576, on the CUDA path: static YaRN x4 in the attention and pooling kernels, a cache sized by the long window (`YARN=1`) |

Every knob is an environment variable with its default in `scripts/config.sh`; the CHANGELOG lists them with the measured effect of each round.

## Checks

`tools/` talks to the running server (`API_URL`, default `http://127.0.0.1:8888`, or `PORT`):

| Script | What it does |
| --- | --- |
| `tools/exact.py` | greedy replies: drafted, `"draft": false`, and concurrent, must match |
| `tools/toolcheck.py` | a tool call whose array argument comes back as a JSON array |
| `tools/needle.py` | a passphrase in a ~200,000-token prompt |
| `tools/prompt_reuse.py` | each follow-up turn resumes most of its prompt from the cache |
| `tools/resume_exact.py` | replies resumed from the cache equal the cold reply, byte for byte, also after other prompts took the streams |
| `tools/full_windows.py` | three prompts near a full window at once, each naming its own passphrase, and the card's peak memory |
| `tools/yarn_checks.sh LOG` | every check of a long window against a `YARN=1` server, with passphrases at about 200,000, 550,000 and 990,000 tokens; about 20 minutes |
| `tools/test_yarn_settings.py` | no server: what `YARN=1` and a long `CONTEXT` set, and that the start check passes on an empty card |
| `tools/test_checkpoint_settings.py` | no server: which checkpoint `CHECKPOINT`, `ABLIT`, `MODEL_ID` and `.env` pick, and that AZ and INT4 stop |
| `tools/videocheck.py` | every video container answers, the prompt's layout, a later turn resuming past its video, the limits |
| `tools/sweep.py` | compares settings: restarts the server for each and measures decode and prefill (it takes the server down between arms) |
| `tools/bench.py` | prefill at 8k-128k and a short decode |
| `tools/bench_streams.py` | decode at 1 to 3 concurrent requests (prose and code, with TTFT) and prefill 4k to 128k, median of 3 |
| `tools/context_boundary.py --context N` | a prompt plus reply that fills `--context` succeeds; one token over is refused (HTTP 400, or an error in the stream). Authored by [321sssrt-bit](https://github.com/321sssrt-bit) |

## Repository layout

```
start.sh      set up (first run) and start the server
stop.sh       stop it
scripts/      prepare.sh, config.sh, publish-image.sh, banner.sh, ablit_view.py
patches/      this recipe's changes, applied onto TensorFold v1.1.4 at TF_REF
tools/        checks, and the video clips they send (tools/fixtures)
.github/      issue and pull request templates, GitHub Sponsors
CREDITS.md    who and what this builds on
CHANGELOG.md  upgrade notes
NOTICE        third-party notices
```

## License

MIT, see [`LICENSE`](LICENSE). The model weights, downloaded from Hugging Face and not part of this repository, are
under the checkpoint's own license (the `ABLIT=1` overlay inherits NVIDIA's and the Qwen license, and has its own Responsible Use Agreement): NVIDIA's FP4 checkpoint under the
[NVIDIA Open Model Agreement](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-agreement/) (its base model, Qwen3.8 Flash Next, is under the
Qwen Community License 1.0).

**Third-party software in the image.** The image `scripts/prepare.sh` builds is based on NVIDIA's PyTorch container
`nvcr.io/nvidia/pytorch:26.07-py3`, redistributed as a value-added runtime image. The NVIDIA software in it is
governed by the [NVIDIA Software License Agreement](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-software-license-agreement/)
and the [Product-Specific Terms for NVIDIA AI Products](https://www.nvidia.com/en-us/agreements/enterprise-software/product-specific-terms-for-ai-products/),
which the container prints at every start. By pulling or running the image you accept them. The MIT license above
covers this repository's scripts only.

## Credits

- **Engine:** [TensorFold](https://github.com/ashhart/TensorFold) and its Zig engine (`tensorfold-native`, release
  [v1.1.4](https://github.com/ashhart/TensorFold/releases/tag/v1.1.4)) by Ash Hart ([ashhart](https://github.com/ashhart)) and the
  [TensorFold contributors](https://github.com/ashhart/TensorFold/graphs/contributors). The Flash Next CUDA engine this recipe builds on is TensorFold's own, written by Ash Hart and the TensorFold contributors.
- **Zig CUDA serving path and CUDA family registry:** Jürgen Schmied ([jschmied](https://github.com/jschmied)), in
  [TensorFold PR #443](https://github.com/ashhart/TensorFold/pull/443)
  (commit [`59e77e8`](https://github.com/ashhart/TensorFold/commit/59e77e8f4b875ce0e863a8c896fc8e424bc539ac)).
- **Patches 0001 to 0017:** written by MiaAI-Lab (Mia's AI Lab), MIT License. They modify
  TensorFold's own Flash Next CUDA engine (code by Ash Hart and the TensorFold contributors, under TensorFold's licenses). The list, patch by patch, with where each
  idea comes from, is in [`CREDITS.md`](CREDITS.md): the vision tower (0001, 0002), fp8 KV (0003, 0005; the format adapted from MiaAI-Lab's GLM recipe patch 0038), prompt
  expert tile and prefill (0004, 0006, 0014, 0015), sampled streams (0007 to 0011; the GPU top-k idea from MiaAI-Lab's DeepSeek recipe), rollback, `pread` n-gram reader and
  decode (0012, 0013, 0016, 0017; 0013, 0016 and 0017 carry forward parts of the earlier db281878 series, whose NVIDIA FP4 port is by MiaAI-Lab from
  the TensorFold Contributors' Python implementation of the same formats).
- **Patch 0025:** written by MiaAI-Lab for the CUDA path, reusing TensorFold's own YaRN table (`rotary_config.zig`) and following its Metal engine's static x4 design.
- **Patches 0018 to 0024:** written by MiaAI-Lab. The host tier of `0019` follows the idea of MiaAI-Lab's GLM recipe on RTX PRO 6000, whose scripts also gave this port its GPU checks and x86_64 build inputs. FFmpeg (LGPL) is in the image through `libtfvideo`.
- **Earlier db281878 series** (on `main`): free after drain and admission reclaim (patch 0029) and image prompts in the prompt cache (patch 0030) by [tempeste](https://github.com/tempeste), now built
  into TensorFold v1.1.4; sliced fills one at a time (patch 0021) by [BadAd84](https://github.com/BadAd84) (PR #53).
- **Model:** [Qwen3.8 Flash Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) by the Qwen team (Alibaba).
- **Checkpoint:** NVIDIA's [FP4 QAD export](https://huggingface.co/nvidia/Qwen3.8-Flash-Next-FP4-Local) (Model Optimizer).
- **AZ checkpoint** (patch `0024` was written for it; no longer served): [`azampatti/Qwen3.8-Flash-Next-FP4-Local-AZ`](https://huggingface.co/azampatti/Qwen3.8-Flash-Next-FP4-Local-AZ) by [azampatti](https://huggingface.co/azampatti), built on NVIDIA's export.
- **INT4 checkpoint** (not served by this engine; the source of AZ's shared expert): [azampatti](https://huggingface.co/azampatti)'s
  [`azampatti/Qwen3.8-Flash-Next-125B-A5B-INT4-AutoRound`](https://huggingface.co/azampatti/Qwen3.8-Flash-Next-125B-A5B-INT4-AutoRound)
  (the top-5 expert cut, the shared-expert healing and the checkpoint), with its AutoRound int4 quantization by
  [Intel](https://huggingface.co/Intel/Qwen3.8-Flash-Next-W4A16-AutoRound) and its hybrid checkpoint and FP8 n-gram table by
  [Saren-Arterius](https://github.com/Saren-Arterius) ([`Saren/Qwen3.8-Flash-Next-ple-table-fp8`](https://huggingface.co/Saren/Qwen3.8-Flash-Next-ple-table-fp8)).
- **Previous checkpoint** (before the Zig engine): [`Vontra/Qwen3.8-Flash-Next-MLX-4bit-MTP`](https://huggingface.co/Vontra/Qwen3.8-Flash-Next-MLX-4bit-MTP)
  by [Vontra](https://huggingface.co/Vontra).
- **Image and video input:** adapted from MiaAI-Lab's earlier patches 0008 and 0009 and from TensorFold's vision code by Ash Hart and the
  TensorFold contributors. Decoding uses libjpeg-turbo, libwebp and [FFmpeg](https://ffmpeg.org/) through TensorFold's `libtfvideo`.
- **FP8 KV format** (`KV_DTYPE=fp8`): adapted from MiaAI-Lab's GLM recipe patch `0038-glm-kv-fp8` in
  [GLM-5.3-Flash-EXL3-2x-DGX-Sparks-TensorFold](https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks-TensorFold).
- **Full-window admission fix and `tools/context_boundary.py`:** [321sssrt-bit](https://github.com/321sssrt-bit)
  ([issue #1](https://github.com/MiaAI-Lab/Qwen3.8-Flash-Dual-DGX-Sparks-TensorFold/issues/1),
  [PR #2](https://github.com/MiaAI-Lab/Qwen3.8-Flash-Dual-DGX-Sparks-TensorFold/pull/2)): a request whose prompt plus `max_tokens` equals the window is
  admitted, because the engine already keeps the draft rows beyond that window.
- **Prompt-chunk change** on the earlier Python engine: [MovieMaker93](https://github.com/MovieMaker93)
  ([TensorFold #40](https://github.com/ashhart/TensorFold/pull/40)).
- **Language draft vocabularies** of the earlier Python recipe: Javier ([jvr0x](https://github.com/jvr0x))
  ([#84](https://github.com/MiaAI-Lab/Qwen3.8-Flash-Next-Single-DGX-Spark/pull/84)); that draft-language image is not built anymore.
- **RoCE all-gather:** the one-shot all-gather in the engine implements the RoCEnante protocol of [b12x](https://github.com/local-inference-lab/b12x) by
  local-inference-lab; this recipe leaves it off.
- **Grammar runtime:** [xgrammar](https://github.com/mlc-ai/xgrammar) by the MLC-AI team (Apache-2.0), shipped as `libtfgrammar.so` in TensorFold's release archive.
- **[Zig](https://ziglang.org)** by the Zig Software Foundation and the Zig contributors; **[FFmpeg](https://ffmpeg.org/)** decodes video; the image is based on NVIDIA's
  [PyTorch container](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch).
- **Benchmarks:** the performance numbers were measured with [sparkDash](https://github.com/MiaAI-Lab/sparkDash) by MiaAI-Lab.

The full list is in [`CREDITS.md`](CREDITS.md).
