# gsj-envloader-examples

Three standalone training projects — **SFT**, **OPD** (on-policy
distillation), **RLVR** (verifiable reward) — built against the published
artifacts and **staging endpoints** of
[`gsj-envloader`](https://github.com/MHGanainy/gsj-envloader), exactly as
an external developer would. Since CP-31 they are the library's
**zero-CLI proof**: a consumer installs the wheel, edits endpoint values
in one YAML, runs `python train.py`, and trains — no CLI invoked, no
scripts fetched, no mounts configured, no data staged. Every friction hit
on the way is registered in [`FINDINGS.md`](FINDINGS.md) — that register
is the point of this repo as much as the code.

What each project consumes (library `docs/publishing.md` + `staging/README.md`):

| input | where |
|---|---|
| library wheel 0.6.0 | GitHub release asset, installed by URL (see any `*/requirements.txt`) |
| sandbox image | `ghcr.io/mhganainy/gsj-pi-harness:pi0.83.0-3` (GHCR tag in the config; `pull: true` on an egress-capable host does the one pull for you) |
| case repos | the staging Forgejo, cloned anonymously by URL at episode time (`task.clone_url_for`) |
| retrieval | the external MCP service (`task.mcp_launch.url_base`, streamable-http, per-episode JWT) — no pages tree, no shim, nothing mounted |
| gate pins | the staging-inputs raw URL + sha256 in the config — fetched and cache-verified by the library itself |
| taskbank | the committed per-project `taskbank.parquet` — **byte-identical to the hosted staging artifact** (same sha256, pinned in the config); a one-line commented alternative consumes it by URL instead |
| render templates | package data inside the wheel (configs omit `templates_dir`) |
| serving | the always-on student endpoint (`serving.base_url`); OPD adds the always-on teacher endpoint under `user.teacher` |

## One venv per project (the two-environment reality is gone)

`collect_episodes` is a library call since 0.6.0, so the collector stack
and the trainer stack live in the SAME venv and one `python train.py`
drives collect → attach → train → save. The venv builds with pip alone —
no setup script exists in this repo anymore:

```bash
cd sft            # or opd, rlvr
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt -r collector-requirements.txt
.venv/bin/pip install --no-deps sglang==0.5.10.post1 \
  "verl @ git+https://github.com/verl-project/uni-agent.git@73b0f41efa88b311fd69129c6f835c012e925e73#subdirectory=verl"
```

Three pip invocations, not one — the honest remainder: the frozen
collector set is not flat-installable (its sglang pin was captured
`--no-deps`, library F-01), and verl installs `--no-deps` from the
uni-agent repo's submodule. Everything resolves from public URLs;
`collector-requirements.txt` is a committed, provenance-headed copy of
the library's freeze with the sglang line removed.

## Prerequisites (operator infrastructure, not this repo's job)

- The staging estate up: the Forgejo case host, the MCP service
  (`/health` → `"state": "ready"`), the **student** vLLM
  (`Qwen/Qwen3-0.6B` at `serving.base_url`), and — for OPD — the
  **teacher** vLLM (`Qwen/Qwen3-4B` at `user.teacher.base_url`, an
  always-on second endpoint; the serve-swap era is over).
- Docker with the sandbox image present (`pull: true` in the config does
  it automatically on a host whose daemon has registry egress; this
  estate's does not, so the image arrived by `docker save | ssh docker load`).
- `GSJ_MCP_TOKEN_SECRET` exported in the environment running `train.py`
  (the config carries the env var's NAME, never its value).
- A GPU for training (pick it with `CUDA_VISIBLE_DEVICES` at invocation).

## Run

```bash
cd sft            # or opd, rlvr — each README has the full run book
export GSJ_MCP_TOKEN_SECRET=<the estate's secret>
CUDA_VISIBLE_DEVICES=6 .venv/bin/python train.py
```

That one command collects sandboxed episodes against the staging
endpoints (gates G1–G7, provenance, store), runs the regime's attach step
in-process where one exists (OPD: teacher scoring against the always-on
teacher endpoint; RLVR: verifiable grading with ground truth from the MCP
service's own `/health` census), trains, saves the adapter, and prints
the serve/commit accounting.

**What a consumer edits** (the whole list, per config): the endpoint
hosts (Forgejo / MCP / serving — deployment topology), the consumer-owned
scratch paths (`store.root`, `task.work_root`, `task.episodes_root`, the
taskbank's absolute path), and `user:` (lr/steps/out — never read by the
library).

## The taskbank

`build_taskbank.py` builds each project's `taskbank.parquet` from the
public case refs alone — timesteps discovered live via
`git ls-remote --heads`, the `summarize` skill prompt, eval split =
`case_0004`, sandbox identity = the GHCR tag. 12 rows: 9 train / 3 eval
per project. The parquets are **committed** (the repo shows its data) and
**byte-identical to the hosted staging artifact** — the same sha256 the
configs pin, so the local default and the commented URL alternative are
the same bytes; pick either.

## What this proves — and doesn't

Everything reaches the library through its two published faces: the one
YAML config file (`collect_episodes`, the in-process attach steps, and
`make_loader` all build from it) and the package-root import surface.
CP-27 found the artifact gaps, CP-28 (v0.5.0) closed them; CP-31 (v0.6.0)
removed the CLI and the staged data — collection is a library call and
every environment input is an endpoint or a sha-pinned URL. What remains
non-zero is the venv build (three pip invocations, F-01's long shadow)
and the estate itself (serving, MCP, docker are operator infrastructure).
`FINDINGS.md` is the honest ledger.
