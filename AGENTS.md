# Working on the Metal port

This branch (`stelau/metal-145b`) runs Edge0's latent-MoE models on Apple Silicon. Upstream ships them for CUDA only. The plan and its state live in the GitHub issues of this fork, not in this file: start from the tracking issue.

## What is here

Everything is under `edge0-strata/`, Edge0's fork of the Strata engine.

- `metal/`: upstream Strata's Metal engine, restored from Niko1221/Strata at the fork's merge base, plus the latent layer.
- `metal/kernels/latent.metal`: the three kernels that are the latent layer's own (`lat_colscale`, `lat_plan_rows`, `lat_group_sum`). D, U and the experts run on the existing `gemv.metal` and `moe.metal`.
- `metal/src/engine/window.cpp`: the decode window. `latent_init` maps the packs, `latent_moe` queues a latent layer, `latent_warm` reads the packs ahead at start.
- `metal/src/engine/prompt_chain.inc` (`pc_chunk`) and `metal/kernels/latent_mm.metal`: the prompt path, chunks of up to 2048 rows on the matrix units. Its latent branch runs D and U as dense GEMMs and the experts as grouped GEMMs at the latent width.
- `metal/src/engine/stubs_edge0.cpp`: CUDA entry points the shared host code references and Metal does not implement. Each stops the engine and prints its name when reached, which is how a missing piece shows itself.
- `metal/tests/latent_parity.cpp`: one latent layer on the Metal kernels against a double-precision reference.
- `metal/tests/latent_chunk_parity.cpp`: the same check for one latent layer of a prompt chunk, with the time of each stage.

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

```sh
./build-metal/latent_chunk_parity                                               # a chunk of 256 rows, random weights
./build-metal/latent_chunk_parity --pack $MODEL_DIR --layer 20 --ntok 2048      # the pack's own weights
```

The chunk path multiplies fp16 operands, so its error is about 7e-4. Run it after any change to the prompt chain's latent branch or to `moe_mm.metal`. Time it only while the engine is not running: both use the GPU.

## Run

`$MODEL_DIR` is the downloaded `Edge0/Edge0-145B-q3k` folder (98 GiB). The server takes a config file in that folder:

```json
{
  "model_name": "edge0-145b-q3k",
  "exe": "<repo>/edge0-strata/build-metal/strata",
  "args": ["--pack", ".", "--native", "./model-skeleton.gguf", "--native-dense-gguf", "./model-skeleton.gguf",
           "--ple-gguf", "./ple-fp8.gguf", "--expert-profile", "./profile.bin", "--expert-cache", "64",
           "--prefill", "auto", "--spec", "4", "--spec-min-p", "0.5", "--mtp", "./mtp", "--max-context", "32768", "--mmap-experts"],
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

The draft head (`--mtp`) needs the drafter as a GGUF, which the pack does not ship. Make it once:

```sh
.venv/bin/python tools/mtp_unrt.py --rt $MODEL_DIR/mtp --out $MODEL_DIR/mtp-q2_0.gguf
```

Without the draft head, drop `--mtp ./mtp`, `--spec-min-p 0.5` and the two `STRATA_METAL_MTP_*` / `_DRAFT_VOCAB` variables and use `--spec 2` (a native pack needs `--spec` of 2 or more).

The Python environment is `uv venv .venv && uv pip install --python .venv/bin/python -r requirements.txt` in `edge0-strata/`. The server speaks the OpenAI chat API on `/v1/chat/completions`.

Switches that matter here (all environment variables, set in the config's `env`):

- `STRATA_METAL_PROMPT=window`: read prompts through 8-token decode windows instead of the prompt chain. `STRATA_METAL_PROMPT_CHUNK=N`: the chain's chunk rows (1024 for a latent pack). `STRATA_METAL_PROMPT_KEEP=1`: keep the chunk buffers between prompts.
- `STRATA_METAL_KEEP_HOST_WEIGHTS=1`: keep the host code's device copies of the dense weights (unused on Metal, given back by default).
- `STRATA_METAL_LATENT_WARM_GB=N`: read ahead D/U plus the hottest N GB of experts by profile rank, instead of all 40 GB (the memory/speed dial; experts are file-backed mmap and OS-streamed, so this sets how much file cache they fill). `STRATA_METAL_LATENT_SKIP=1`: skip the D/U GEMVs of latent groups no expert routed to (+8% single-token decode).
- `STRATA_METAL_MMA_MIN=N`: the window size from which the large Q8_0 matrices run on the matrix units (4; 0 never).
- `STRATA_METAL_PROFILE=1`: window timings by size, printed at each prompt. `STRATA_METAL_KPROF=T` (and `_EVERY=N`): the GPU time of every kernel class of a T-token window. `STRATA_METAL_ALLOC_LOG=1`: every allocation of 16 MB or more.

## Memory: read this before starting the engine

The engine wires the whole pack once the GPU has used it: about 50 GB on the 145B q3k pack, and it cannot shrink under pressure. Beside that it holds about 3.5 GB of its own buffers, and 2.6 GB more while a long prompt is read. On a 64 GB Mac:

- Check free memory first. If other programs hold more than about 15 GiB, the engine pushes them to swap and the machine crawls. Do not start it then; say so and stop.
- One heavy job at a time. Never run the engine beside another model, a large build, or parallel agents that run things.
- The read-ahead at start (`latent_warm`) reads the packs into the file cache when there is room. Without it, the first prompts page the experts in at about 0.4 GB/s. `STRATA_METAL_LATENT_WARM=1` forces it, `=0` turns it off.
- When speed swings for no visible reason, look at memory before the kernels: `vmmap --summary <engine pid>`, row `IOAccelerator (graphics)`, shows the engine's own buffers and how much of them macOS has compressed or swapped; `sysctl vm.swapusage` shows the swap.
- Stop the server when a run is done: `pkill -f "serve/server.py --engine strata"`.

## How work is tracked

- Start each session with the upstream check in issue #8: if Edge0 has released its own macOS engine, compare with it before doing more.
- Each piece of work is an issue. Before starting one, read it; when measuring something, put the numbers in a comment on it; tick its checkboxes as they are met.
- A change that is not covered by an issue gets one first, even a short one.
- Keep the board current. Close an issue as soon as its stated scope is delivered, with a comment summarising what shipped and the measurement; do not leave finished work open.
- A new finding, follow-up, or adjacent idea gets its OWN new issue, not appended to an open one. Do not grow an existing issue's scope to absorb new work (no feature creep): the issue you close should be the issue you opened.
- Commit messages name the issue (`refs #3`, `closes #1`).
- Every claim of speed or correctness comes with the measurement and how it was taken. An estimate is labelled as one.

## Before a long eval run or a pull request: review and test first

Both cost hours or other people's attention, so the cheap checks come before, not after.

Before a long eval run:

- Smoke-test every step of the run end to end on something small (a tiny model, or two or three items of each suite) and read the output of each step, including the files it writes. A step that silently records harness errors as wrong answers wastes the whole run.
- Check the machine: power (`pmset -g batt`, and the adapter's wattage: a weak charger drains the battery and macOS then throttles the GPU to a tenth of its speed), free memory, and that nothing else heavy is running. Never build while an engine is serving.
- A few minutes into the run, read the per-request tok/s in the engine log. Timings from the first minutes after a start, or from a throttled machine, are not results.
- A change to the engine gets its correctness check first (`logprobe.py --diff` against the build it changes, the parity tests), then the long suites.

Before opening a pull request, here or upstream:

- Have each change reviewed independently before it is offered: one reviewer per change, given the diff and the surrounding code and asked what would break, not told that it works. Check every finding against the code, fix what is real, and say in the description what was found and left.
- Prove each change alone, built by itself on the branch it targets: output unchanged (or the intended change shown), the effect it claims measured, with the commands.
- Run every check the target repository asks for (its PR template, its testing guide, its benchmark), whether or not the change seems to touch what the check covers. It is their repository and they asked. If a check needs something that is not on the machine (a model, fixtures), get it and run it. If one truly cannot be run, the description says which and why.
- A description claims only what was run on that branch. A result from another branch or another engine is named as such or left out.
- Pull requests are opened as drafts and stay drafts until the reviews are answered, the checks have run and the description matches the branch.

## Pushing

Push to the fork as work lands: after each commit or small group of commits that builds and passes `latent_parity`. Unpushed work is unbacked work. The fork is public, so every push is checked first:

```sh
LEAK_PATTERNS=<private pattern file> tools/leak_check.sh     # the commits the fork does not have yet
git push fork <branch>
```

`tools/leak_check.sh` scans the added lines, commit messages and commit identities for home-directory paths, email addresses, credential formats and the private patterns, then runs gitleaks on the same commits. Push only on "clean" (exit 0). Exit 2 means the check was incomplete (no private pattern file, or gitleaks missing): fix that, do not push on it. On a hit, rewrite the commit so the detail never enters history; do not push and then delete.

A clone used for this work should run the check from a `pre-push` hook (`.git/hooks/pre-push`: set `LEAK_PATTERNS`, call `tools/leak_check.sh` with the range of each ref being pushed, refuse on any non-zero exit), so a push cannot skip it. Hooks are local to a clone: check that it exists before relying on it.

The private pattern file lists the names and words that must not appear. It lives outside the repository, because the list itself would be a leak here.

The same care applies to issues and comments, which no script checks: read them before posting.

## Conventions

- Comments and docstrings describe the code as it is, never how it got there. History goes in commit messages.
- Plain hyphens, colons and commas in prose, comments and commits. No em or en dashes.
- Stage files by path. Never `git add -A` or `git add .`.
- Nothing personal in the repository or its issues: no home-directory paths, names, email addresses or employer references. Use `$MODEL_DIR` and `<repo>`. The repository's local git config sets a no-reply commit identity; keep it.
- Do not open pull requests against upstream without being asked.

## Measured so far (M4 Max, 64 GB, 145B q3k)

- Prompt: 330 to 430 tok/s through the prompt chain from 2.5K to 32K tokens (a 32.3K-token prompt in 97 s). 100 tok/s through decode windows.
- Decode, 300 tokens, thinking off: 37 to 39 tok/s with the draft head after a short prompt, 33 to 36 after 8.6K tokens; 35 and 33 without it. A one-token window is 28 to 30 ms of GPU time at any context length up to 28K, a four-token window 63 ms.
- A one-token window's GPU time by kernel (`STRATA_METAL_KPROF=1`): the latent D and U GEMVs 10.3 ms of 27.9, the experts' gate/up and down 3.5 ms, the rest the dense projections.
- Memory: 49.8 GB of wrapped files, 3.5 GB of the engine's own buffers resident.
- Latent layer alone (`latent_parity`): 0.66 ms for a one-token window. A chunk's latent layer (`latent_chunk_parity`): 24 ms for 1024 rows.
- Edge0's own figures for comparison: 60.9 tok/s decode and about 750 to 900 tok/s prompt on an M2 Ultra (unreleased Metal build), 99.7 tok/s decode on an RTX 5090 D.
