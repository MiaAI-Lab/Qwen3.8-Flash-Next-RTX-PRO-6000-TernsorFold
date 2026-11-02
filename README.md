<h1 align="center">Qwen3.8 Flash Next on one RTX PRO 6000 (TensorFold)</h1>

<p align="center">
  <sub>by <a href="https://x.com/MiaAI_lab">Mia's AI Lab</a></sub>
  <br><br>
  <a href="https://github.com/sponsors/MiaAI-Lab" target="_blank" rel="noopener noreferrer" style="display:inline-block;margin:0 8px;vertical-align:middle;"><img src="https://img.shields.io/badge/Sponsor%20me%20on%20GitHub-181717?style=for-the-badge&logo=githubsponsors&logoColor=white" alt="Sponsor me on GitHub" height="28" style="height:28px;width:auto;vertical-align:middle;border:0;" /></a>
  <a href="https://x.com/MiaAI_lab" target="_blank" rel="noopener noreferrer" style="display:inline-block;margin:0 8px;vertical-align:middle;"><img src="https://img.shields.io/badge/Follow%20me%20on%20X-000000?style=for-the-badge&logo=x&logoColor=white" alt="Follow Mia on X" height="28" style="height:28px;width:auto;vertical-align:middle;border:0;" /></a>
</p>

Serve **Qwen3.8 Flash Next** from one NVIDIA RTX PRO 6000 Blackwell (96 GB) through an OpenAI-compatible API, with image and video input. It runs [TensorFold](https://github.com/ashhart/TensorFold)'s Zig engine, v1.1.4 with this recipe's patches, on one GPU (TP=1).

## Modes

| Start with | What it serves | Status on this card |
| --- | --- | --- |
| `./start.sh` | [NVIDIA's FP4 checkpoint](https://huggingface.co/nvidia/Qwen3.8-Flash-Next-FP4-Local) in its original form: 3 requests at once, 262,144 tokens each | Checked |
| `ABLIT=1 ./start.sh` | The same checkpoint with [drowzeys's overlay](https://huggingface.co/drowzeys/keys-Qwen3.8-Flash-Next-NVFP4-FP4-QAD-house-qsa-L3-47), which removes safety refusals. Gated: accept its terms, then set `HF_TOKEN` | Never started |
| `YARN=1 ./start.sh` | YaRN x4: one request at a time, up to 1,000,000 tokens. Off by default, because it changes replies on short prompts too | Not run on NVIDIA's FP4 yet ([results so far](docs/details.md#long-window-yarn1)) |

## Requirements

- Linux x86_64 with one RTX PRO 6000 Blackwell and nothing else large on it: a start needs 96,845 MiB free on the card
- A driver for CUDA 13, Docker with the NVIDIA container runtime, and your user in the `docker` group
- About 160 GB of free disk, with the checkpoint on an NVMe drive
- Developed on 89 GB of RAM; the prompt cache pins up to 16 GiB of it

## Quick start

```bash
./start.sh
```

The first run builds the engine (about 5 minutes) and downloads the checkpoint (about 96 GiB). Later starts take about 2.5 minutes.

```bash
curl -s http://<host>:8888/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "Qwen3.8-Flash-Next",
  "messages": [{"role": "user", "content": "Write a Python fibonacci function."}],
  "max_tokens": 1000
}'
```

Any OpenAI client works with `base_url = "http://<host>:8888/v1"`. Streaming, tool calls, structured output and reasoning content are supported. The model thinks before it answers, so give replies enough `max_tokens`.

```bash
./start.sh restart                      # stop, then start again
./stop.sh                               # stop
docker logs -f qwen38-flash-next-tf     # logs
DRY_RUN=1 ./start.sh                    # show what a start would do, and change nothing
```

## Performance

**Run with the card's power limit set to 300 W** (its default is 600 W, so these are not its best). NVIDIA's FP4 with all 25 patches, 3 requests at once at 262,144 tokens each, fp8 KV, vision and MTP drafts on, measured with [sparkDash](https://github.com/MiaAI-Lab/sparkDash).

**Decode, prose** (temperature 0.7, thinking on)

| Requests at once | Aggregate | Per request | TTFT |
| ---: | ---: | ---: | ---: |
| 1 | 213.4 tok/s | 213.4 tok/s | 35 ms |
| 2 | 371.7 tok/s | 189.9 tok/s | 110 ms |
| 3 | 452.9 tok/s | 157.0 tok/s | 110 ms |

**Prefill**

| Prompt | Tokens | Prefill | TTFT |
| --- | ---: | ---: | ---: |
| 4k | 4,134 | 6,283.3 tok/s | 729 ms |
| 8k | 8,231 | 6,074.2 tok/s | 1.43 s |
| 16k | 16,423 | 5,949.0 tok/s | 2.83 s |
| 32k | 32,805 | 5,691.1 tok/s | 5.83 s |
| 64k | 65,574 | 5,532.5 tok/s | 11.92 s |
| 128k | 131,113 | 5,217.4 tok/s | 25.20 s |

**Quality:** 96.8% on sparkDash's quality bench, averaged over its 8 categories (749 of 797 questions at temperature 0, one run). The breakdown is in [`docs/details.md`](docs/details.md#quality).

## Images and video

Send them as OpenAI-style `image_url` and `video_url` content parts, as data URLs (`VISION_URLS=1` also allows public `https://` URLs). `VISION=0` serves text only.

| | Formats | Per request |
| --- | --- | --- |
| Images | JPEG, PNG, WebP | up to 50, 10 MB each, 16,384 tokens in all |
| Video | MP4, WebM, MOV, MKV | up to 4, 16 MB each, 16,384 tokens in all |

## Settings

Set these in the environment (`PARALLEL=2 ./start.sh`) or in `.env` next to `start.sh`.

| Setting | Default | Meaning |
| --- | --- | --- |
| `GPU` | `0` | the card, by its `nvidia-smi` index |
| `PORT` | `8888` | where the API listens |
| `PARALLEL` | `3` | requests decoded together. Four full windows do not fit the card |
| `CONTEXT` | `262144` | prompt + reply window. Above this, YaRN turns on |
| `MAX_TOKENS` | `32768` | the reply budget when a request sets none |
| `THINKING` | `1` | `0` answers without thinking |
| `DRAFTS` | `1` | MTP drafts, which speed up decoding and never change a reply |
| `KV_DTYPE` | `fp8` | `bf16` is the exact cache, but only one stream fits |
| `PROMPT_CACHE_GIB` | `16` | host memory for kept prompts, so a chat's next turn resumes instead of reading again |

## More

- [`docs/details.md`](docs/details.md): every setting, patch, check and measurement, plus the memory breakdown and API notes
- [`CHANGELOG.md`](CHANGELOG.md): what changed, with results

## License

MIT, see [`LICENSE`](LICENSE); this covers the repository's scripts only. The weights are downloaded from Hugging Face under their own terms: the [NVIDIA Open Model Agreement](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-agreement/) and the Qwen Community License 1.0. The `ABLIT=1` overlay also has its own Responsible Use Agreement.

The image `scripts/prepare.sh` builds is based on NVIDIA's PyTorch container, under the [NVIDIA Software License Agreement](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-software-license-agreement/) and the [Product-Specific Terms for NVIDIA AI Products](https://www.nvidia.com/en-us/agreements/enterprise-software/product-specific-terms-for-ai-products/). By pulling or running the image you accept them. See [`NOTICE`](NOTICE).

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
- **Earlier db281878 series** (in this repository's history): free after drain and admission reclaim (patch 0029) and image prompts in the prompt cache (patch 0030) by [tempeste](https://github.com/tempeste), now built
  into TensorFold v1.1.4; sliced fills one at a time (patch 0021) by [BadAd84](https://github.com/BadAd84) (PR #53).
- **Model:** [Qwen3.8 Flash Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) by the Qwen team (Alibaba).
- **Checkpoint:** NVIDIA's [FP4 QAD export](https://huggingface.co/nvidia/Qwen3.8-Flash-Next-FP4-Local) (Model Optimizer).
- **`ABLIT=1` overlay:** the Keys "house-qsa" abliteration overlay, [`drowzeys/keys-Qwen3.8-Flash-Next-NVFP4-FP4-QAD-house-qsa-L3-47`](https://huggingface.co/drowzeys/keys-Qwen3.8-Flash-Next-NVFP4-FP4-QAD-house-qsa-L3-47), by [drowzeys](https://huggingface.co/drowzeys). `scripts/ablit_view.py`, which grafts it in, is MiaAI-Lab's.
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
