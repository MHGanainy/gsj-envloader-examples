# gsj-envloader-examples

Three standalone training projects — **SFT**, **OPD** (on-policy
distillation), **RLVR** (verifiable reward) — built against the published
artifacts of [`gsj-envloader`](https://github.com/MHGanainy/gsj-envloader)
exactly as an external developer would: public URLs only, no access to the
library's dev harness. Every friction hit on the way is registered in
[`FINDINGS.md`](FINDINGS.md) — that register is the point of this repo as
much as the code.

The published artifacts consumed here (library `docs/publishing.md`):

| artifact | where |
|---|---|
| library wheel 0.5.0 | GitHub release asset, installed by URL (see any `*/requirements.txt`) |
| sandbox image | `ghcr.io/mhganainy/gsj-pi-harness:pi0.83.0-mcp1.5.0-2` (anonymous pull; its baked MCP shim is named directly via `mcp_launch.in_image`) |
| case repos | `https://github.com/MHGanainy/gsj-case-000{1..4}` (anonymous clone) |
| page corpus | `gsj-pages-20260204.tar.gz` release asset |
| gate pins | `gsj-pins-0.5.0.json` release asset (downloaded sha-verified by `setup_collector.sh`; dev-environment DATA — your own environment regenerates its own) |
| render templates | package data inside the wheel (configs omit `templates_dir`) — also published standalone as `gsj-pi-templates-0.5.0.tar.gz` |
| frozen collector deps | installed by the library's published `install_collector_env.sh` + `requirements.txt`, fetched at `v0.5.0` by `setup_collector.sh` |

## The two-environment reality

Every project needs **two** python environments:

1. **The collector env** (`collector-venv/`, shared, built once by
   `./setup_collector.sh`): the uni-agent gateway stack at the pinned sha
   plus the library wheel. It runs `gsj-collect` — the library's bounded
   seeding tool (0.5.0) that executes sandboxed episodes and fills a
   project's store, printing progress and exiting at the target.
   Heavy (torch, ray, transformers — the frozen upstream set).
2. **The trainer env** (`<project>/.venv`, per project, from that
   project's `requirements.txt`): the library wheel + torch/transformers/
   peft. It runs the attach job and the training loop against the store.

They meet only at the store directory on disk — no RPC, by the library's
design. The three project dirs are deliberately self-contained duplicates
(own venv, own config, own parquet): each is meant to be readable alone.

## Prerequisites

- A linux x86_64 GPU host (these configs pin an H200; one free GPU for
  training, one share of a GPU for vLLM serving), docker, python 3.12
  (`uv` used when present), git, ~20 GB disk for venvs + models.
- **Serving**: an OpenAI-compatible vLLM endpoint of `Qwen/Qwen3-0.6B`
  with LoRA updating enabled, reachable at the `serving.base_url` in the
  configs (`http://127.0.0.1:8100/v1` here). For OPD scoring the same
  port is temporarily swapped to `Qwen/Qwen3-4B` (the teacher) with
  prompt-logprobs available (stock vLLM has it). Serving is operator
  infrastructure — the library connects to it, this repo does not manage it.
- **Egress-locked hosts**: if the GPU host's docker daemon cannot reach
  ghcr.io, load the image from any host that can:
  `docker save ghcr.io/mhganainy/gsj-pi-harness:pi0.83.0-mcp1.5.0-2 | ssh <host> docker load`.

## Walkthrough

```bash
git clone https://github.com/MHGanainy/gsj-envloader-examples
cd gsj-envloader-examples
./setup_collector.sh          # collector venv + assets (pages, pins)
# edit */config.yaml if your checkout root differs — the configs pin
# absolute paths (no interpolation exists, by design; one root prefix)
```

Then per project, in order (each README has the full run book):

```bash
cd sft            # or opd, rlvr
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
cd ..
collector-venv/bin/gsj-collect --config sft/config.yaml     # seed: bounded, observable, stoppable
# opd only: swap serving to the 4B, then .venv/bin/python score.py --config config.yaml
# rlvr only: .venv/bin/python grade.py --config config.yaml
sft/.venv/bin/python sft/train.py --config sft/config.yaml
```

The seed step is one bounded command since library 0.5.0 (closing CP-27's
F-08/F-14/F-17): `gsj-collect` reads its targets from the config's
`collector.seeding` block (flags override), prints one line per episode
plus periodic progress, exits 0 at the target or round-complete, and a
single Ctrl-C drains in-flight episodes before exiting. The CP-27
store-polling loop and `pkill -KILL` recipe are gone.

## The taskbank

`build_taskbank.py` builds each project's `taskbank.parquet` from the
public case repos alone — timesteps discovered live via
`git ls-remote --heads` (one branch per historical timestep), the
`summarize` skill prompt, eval split = `case_0004`, sandbox identity = the
GHCR tag. 12 rows: 9 train / 3 eval per project. The parquets are
**committed** (the repo shows its data; a reviewer needs no network) and
**regenerable** (rerun the script after the case repos change).

## What this proves — and doesn't

Everything here reaches the library through its two published faces: the
one YAML config file (`gsj-collect`, the attach jobs, and `make_loader`
all build from it) and the package-root import surface. CP-27 found the
gaps (pins, templates, the frozen dependency set, a runnable seeding
tool); the library's CP-28 (v0.5.0) closed them, and this repo's
workarounds were **deleted** — `FINDINGS.md`'s Status column is the
closure record.
