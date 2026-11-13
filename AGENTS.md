# Working on the Metal port

This branch (`stelau/metal-145b`) runs Edge0's latent-MoE models on Apple Silicon. Upstream ships them for CUDA only. The plan and its state live in the GitHub issues of this fork, not in this file: start from the tracking issue.

## What is here

Everything is under `edge0-strata/`, Edge0's fork of the Strata engine.

- `metal/`: upstream Strata's Metal engine, restored from Niko1221/Strata at the fork's merge base, plus the latent layer.
- `metal/kernels/latent.metal`: the three kernels that are the latent layer's own (`lat_colscale`, `lat_plan_rows`, `lat_group_sum`). D, U and the experts run on the existing `gemv.metal` and `moe.metal`.
- `metal/src/engine/window.cpp`: the decode window. `latent_init` maps the packs, `latent_moe` queues a latent layer, `latent_warm` reads the packs ahead at start.
- `metal/src/engine/stubs_edge0.cpp`: CUDA entry points the shared host code references and Metal does not implement. Each stops the engine and prints its name when reached, which is how a missing piece shows itself.
- `metal/tests/latent_parity.cpp`: one latent layer on the Metal kernels against a double-precision reference.

A latent layer: `xl[t][g] = (D_g . x[t]) / s` for 8 groups, each routed expert reads the row of its group label and computes `yl = down . (silu(gate . xl) * (up . xl))` at the latent width, and the output is the weighted sum of `U_g . yl`. U is linear, so the window sums the weighted `yl` by group and runs one GEMV a group.

## Build

Needs Xcode, CMake, Ninja, and two downloads into `edge0-strata/third_party/` (both ignored by git): Apple's metal-cpp headers (`metal-cpp/`, URL and SHA-256 in `metal/setup_mac.py`) and llama.cpp at the commit in `third_party/ggml/VERSION.txt` (`llama.cpp/`).

```sh
cd edge0-strata
cmake -S metal -B build-metal -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DMETAL_CPP_DIR=$PWD/third_party/metal-cpp -DSTRATA_GGML_DIR=$PWD/third_party/llama.cpp
ninja -C build-metal strata latent_parity
```

Shaders compile at run time, so a shader error appears on the first run, not in the build.

## Test

```sh
./build-metal/latent_parity                                   # random weights
./build-metal/latent_parity --pack $MODEL_DIR --layer 20 --ntok 8   # the pack's own weights
```

It prints the relative error against the reference (about 3e-7 is normal), the layer's time, and PASS or FAIL. It loads one layer, about 1 GB, so it is safe to run at any time. Run it after any change to the latent path.

## Run

`$MODEL_DIR` is the downloaded `Edge0/Edge0-145B-q3k` folder (98 GiB). The server takes a config file in that folder:

```json
{
  "model_name": "edge0-145b-q3k",
  "exe": "<repo>/edge0-strata/build-metal/strata",
  "args": ["--pack", ".", "--native", "./model-skeleton.gguf", "--native-dense-gguf", "./model-skeleton.gguf",
           "--ple-gguf", "./ple-fp8.gguf", "--expert-profile", "./profile.bin", "--expert-cache", "64",
           "--prefill", "auto", "--spec", "2", "--max-context", "32768", "--mmap-experts"],
  "cwd": ".", "tokenizer": "./tokenizer", "log": "./serve-mac.log", "port": 8123, "lib_dirs": [],
  "env": {"STRATA_LATENT_DENSE": "./latent_dense_q8.bin", "STRATA_LATENT_DENSE_FMT": "8",
          "STRATA_LATENT_A": "./experts.bin", "STRATA_LATENT_LABELS": "./labels.bin"}
}
```

```sh
cd $MODEL_DIR
<repo>/edge0-strata/.venv/bin/python <repo>/edge0-strata/serve/server.py \
  --engine strata --config ./model-mac.json --port 8123 --host 127.0.0.1 --no-api-key --offline
```

The Python environment is `uv venv .venv && uv pip install --python .venv/bin/python -r requirements.txt` in `edge0-strata/`. The server speaks the OpenAI chat API on `/v1/chat/completions`. A native pack needs `--spec` of 2 or more.

## Memory: read this before starting the engine

The engine wires the whole pack once the GPU has used it: about 50 GB on the 145B q3k pack, and it cannot shrink under pressure. On a 64 GB Mac:

- Check free memory first. If other programs hold more than about 15 GiB, the engine pushes them to swap and the machine crawls. Do not start it then; say so and stop.
- One heavy job at a time. Never run the engine beside another model, a large build, or parallel agents that run things.
- The read-ahead at start (`latent_warm`) reads the packs into the file cache when there is room, and prompts are fast after it (about 43 tok/s through decode windows). Without it, prompts page the experts in at about 0.4 GB/s. `STRATA_METAL_LATENT_WARM=1` forces it, `=0` turns it off.
- Stop the server when a run is done: `pkill -f "serve/server.py --engine strata"`.

## How work is tracked

- Each piece of work is an issue. Before starting one, read it; when measuring something, put the numbers in a comment on it; tick its checkboxes as they are met.
- A change that is not covered by an issue gets one first, even a short one.
- Commit messages name the issue (`refs #3`, `closes #1`).
- Every claim of speed or correctness comes with the measurement and how it was taken. An estimate is labelled as one.

## Conventions

- Comments and docstrings describe the code as it is, never how it got there. History goes in commit messages.
- Plain hyphens, colons and commas in prose, comments and commits. No em or en dashes.
- Stage files by path. Never `git add -A` or `git add .`.
- Nothing personal in the repository or its issues: no home-directory paths, names, email addresses or employer references. Use `$MODEL_DIR` and `<repo>`. The repository's local git config sets a no-reply commit identity; keep it.
- Do not open pull requests against upstream without being asked.

## Measured so far (M4 Max, 64 GB, 145B q3k)

- Latent layer on Metal: 0.69 ms a layer for a 1-token window, 1.8 ms for 8 tokens.
- Decode: 34 to 37 tok/s on short generations, 25 to 28 tok/s across a long eval run. No draft head yet.
- Prompt: 43 to 54 tok/s through decode windows with the packs read ahead.
- Edge0's own figures for comparison: 60.9 tok/s decode and about 750 to 900 tok/s prompt on an M2 Ultra (unreleased Metal build), 99.7 tok/s decode on an RTX 5090 D.
