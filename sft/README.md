# sft — supervised fine-tuning, zero-CLI

The smallest consumer: ONE `python train.py` collects **completed**
episodes against the staging endpoints (in-process — `collect_episodes`,
no CLI), then teacher-forces the student over its own recorded tokens
with a CE step on exactly the positions the collector marked trainable.
No attach job, no mix — the base record already carries everything SFT
needs. The training loop is the library README's five-line story
(`SFTCollator` bakes the target alignment into `labels`;
`model(**batch).loss` is the whole loss).

## Prepare (once)

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt -r collector-requirements.txt
.venv/bin/pip install --no-deps sglang==0.5.10.post1 \
  "verl @ git+https://github.com/verl-project/uni-agent.git@73b0f41efa88b311fd69129c6f835c012e925e73#subdirectory=verl"
```

Prerequisites beyond that (root README): the staging estate up (Forgejo,
MCP service `ready`, student vLLM serving), the sandbox image in the
local docker daemon, and `GSJ_MCP_TOKEN_SECRET` exported.

## What to edit

1. Endpoint hosts in `config.yaml` (Forgejo `clone_url_for`, MCP
   `url_base`, `serving.base_url`) — your estate's topology.
2. The scratch paths (`store.root`, `task.work_root`,
   `task.episodes_root`) + the taskbank's absolute path — your checkout.
3. `user:` (lr / steps / out) — yours; the library never reads it.

The taskbank default is the **committed** `taskbank.parquet` (the repo
shows its data; the sha256 pin in the config verifies the local file and
equals the hosted staging artifact's sha — same bytes either way). The
commented one-liner in `config.yaml` consumes it by URL instead.

## Run

```bash
export GSJ_MCP_TOKEN_SECRET=<the estate's secret>
CUDA_VISIBLE_DEVICES=6 .venv/bin/python train.py
```

Collection: 9 episodes (the config's `collector.seeding.episodes`), each
an ephemeral sandbox container driven by pi against the served 0.6B —
retrieval over the remote MCP service under a per-episode JWT, cases
cloned from the staging Forgejo, pins fetched sha-verified — then
gate-verified (G1–G7) and finalized into `sft/store/`. The script prints
each structured progress event, the CollectReport, and a per-record gate
spot-check (gates empty, G2/G3 hashes, mounts = 2). Training:
`user.steps` = 4 optimizer steps at `loader.batch_size` = 2; the adapter
lands in `sft/adapters/run1`; the accounting block shows per-record serve
counts (the §7 commit-retires contract, observable).

## Recorded run (H200, 2026-08-07 — the CP-31 zero-CLI proof, v0.6.0)

See the run transcript in this README's section below once recorded.
