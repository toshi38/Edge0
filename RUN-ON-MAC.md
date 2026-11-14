# Run Edge0-145B on a Mac

This fork's branch `stelau/metal-145b` runs Edge0-145B (a 145B latent-MoE model) on Apple Silicon via a Metal port of
the Strata engine. Upstream ships the 145B for CUDA only.

**Tested on:** an M4 Max, 64 GB. Smaller Macs are untested but should work with the memory knobs at the end turned down.

## What you need

- An **Apple Silicon** Mac (M1 or newer). More RAM is faster, not required; see "Smaller Macs" below.
- **~120 GB free on the internal SSD.** The pack is 99 GB and streams from disk per token, so it must be on a fast
  local drive, not an external or network disk.
- **Xcode**, **CMake**, **Ninja** (`brew install cmake ninja`), and **uv** (`brew install uv`) for the Python server.

## 1. Build the engine

```sh
git clone -b stelau/metal-145b https://github.com/toshi38/Edge0 && cd Edge0/edge0-strata
# two downloads into third_party/ (both git-ignored): see metal/setup_mac.py for the metal-cpp URL+SHA,
# and third_party/ggml/VERSION.txt for the llama.cpp commit to check out into third_party/llama.cpp
cmake -S metal -B build-metal -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DMETAL_CPP_DIR=$PWD/third_party/metal-cpp -DSTRATA_GGML_DIR=$PWD/third_party/llama.cpp
ninja -C build-metal strata
uv venv .venv && uv pip install --python .venv/bin/python -r requirements.txt
```

Shaders compile at run time, so a shader error (if any) shows on the first run, not in the build.

## 2. Get the pack

Download the `Edge0/Edge0-145B-q3k` pack (99 GB) to a folder; call it `$MODEL_DIR`.

## 3. Build the draft-head GGUF (one time)

The pack ships the speculative-decoding drafter as a runtime directory; the Metal engine reads it as a GGUF:

```sh
.venv/bin/python tools/mtp_unrt.py --rt $MODEL_DIR/mtp --out $MODEL_DIR/mtp-q2_0.gguf
```

## 4. Launch

Put this `model-mac.json` in `$MODEL_DIR` (replace `<repo>` with the absolute path to the cloned `Edge0`):

```json
{
  "model_name": "edge0-145b-q3k",
  "exe": "<repo>/edge0-strata/build-metal/strata",
  "args": ["--pack", ".", "--native", "./model-skeleton.gguf", "--native-dense-gguf", "./model-skeleton.gguf",
           "--ple-gguf", "./ple-fp8.gguf", "--expert-profile", "./profile.bin", "--expert-cache", "64",
           "--prefill", "auto", "--spec", "4", "--spec-min-p", "0.5", "--mtp", "./mtp",
           "--max-context", "32768", "--mmap-experts"],
  "cwd": ".", "tokenizer": "./tokenizer", "log": "./serve-mac.log", "port": 8123, "lib_dirs": [],
  "env": {"STRATA_LATENT_DENSE": "./latent_dense_q8.bin", "STRATA_LATENT_DENSE_FMT": "8",
          "STRATA_LATENT_A": "./experts.bin", "STRATA_LATENT_LABELS": "./labels.bin",
          "STRATA_METAL_MTP_GGUF": "./mtp-q2_0.gguf", "STRATA_METAL_DRAFT_VOCAB": "./mtp/draft_vocab.bin"}
}
```

```sh
cd $MODEL_DIR
<repo>/edge0-strata/.venv/bin/python <repo>/edge0-strata/serve/server.py \
  --engine strata --config ./model-mac.json --port 8123 --host 127.0.0.1 --no-api-key --offline
```

It speaks the OpenAI chat API on `http://127.0.0.1:8123/v1/chat/completions`, or open `http://127.0.0.1:8123/` to chat.
The first start takes a minute (and warms the file cache); on the 64 GB machine it reads ~400 tok/s on prompts and
generates ~35-39 tok/s.

## Smaller Macs: the memory knobs

The experts (40 GB) and the n-gram table (51 GB) are memory-mapped and streamed from the SSD, so the model does not
need them all in RAM. The hard resident footprint during generation is ~5-7 GB plus the KV cache. The limit is Apple's
GPU working-set budget (about 75% of RAM), which has to hold that resident set. On less RAM, add these to the config's
`env` / `args`:

- **`"STRATA_METAL_LATENT_WARM_GB": "N"`** (env): read ahead only the hottest N GB of experts at start instead of all
  40, leaving RAM free. Start around `8` on 32 GB, `4` on 24 GB, `2` on 16 GB. Cold experts then fault from the SSD on
  first use (slower first tokens, but correct). Lower N = less memory used, more SSD reads.
- **`--max-context 16384`** (or `8192`) in `args`: the KV cache grows with the context window; a smaller window frees
  memory.
- Leave `--spec 4 --mtp ./mtp` as is; speculative decoding helps on any Mac.

Rough expectations (untested below 64 GB):

| RAM | Expect |
|---|---|
| 48-64 GB | as tested: full read-ahead, ~400 tok/s prompt, ~37 tok/s decode |
| 32-36 GB | `WARM_GB=8`, `--max-context 16384`; a bit slower, cold experts fault occasionally |
| 24 GB | `WARM_GB=4`, `--max-context 8192`; noticeably slower, more SSD streaming |
| 16 GB | experimental: `WARM_GB=2`, `--max-context 8192`; may be slow or hit the working-set limit |
| < 16 GB | unlikely to fit the resident set beside macOS |

If decode drops to a few tok/s and stays there, the machine is short of memory: lower `WARM_GB` and `--max-context`,
and close other apps. The engine reports real free memory at start (`serve-mac.log`, "MiB of VRAM free").
