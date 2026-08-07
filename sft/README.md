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
.venv/bin/pip install -r requirements.txt \
  -r https://raw.githubusercontent.com/MHGanainy/gsj-envloader/baac10bad425b97ef45ec4d3129197f646e87360/staging/collector/collector-requirements.txt
.venv/bin/pip install --no-deps sglang==0.5.10.post1 \
  "verl @ git+https://github.com/verl-project/uni-agent.git@73b0f41efa88b311fd69129c6f835c012e925e73#subdirectory=verl"
```

The collector set is the library's canonical GENERATED artifact
(gsj-envloader CP-32 / ADR-0045 — the per-project committed copies are
gone; the `--no-deps` line is also printed in the artifact's own header).
The v0.7.0 raw-at-tag URL goes live with the library's CP-33 release;
until then substitute any library commit-sha raw URL of the same path.

Prerequisites beyond that (root README): the staging estate up (Forgejo,
MCP service `ready`, student vLLM serving), the sandbox image in the
local docker daemon, and `GSJ_MCP_TOKEN_SECRET` exported.

## What to edit

1. Endpoint hosts in `config.yaml` (Forgejo `clone_url_for`, MCP
   `url_base`, `serving.base_url`) — your estate's topology.
2. The scratch paths (`store.root`, `task.work_root`,
   `task.episodes_root`) + the taskbank's absolute path — your checkout.
3. Training parameters (lr / steps / out) — the **TRAINING PARAMETERS**
   constants at the top of `train.py` (the CP-33 config split:
   `config.yaml` is the library surface only; the run is code).

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
`STEPS` = 4 optimizer steps at `loader.batch_size` = 2; the adapter
lands in `sft/adapters/run1`; the accounting block shows per-record serve
counts (the §7 commit-retires contract, observable).

## Recorded run (H200, 2026-08-07 — the CP-31 zero-CLI proof, v0.6.0)

One command (`CUDA_VISIBLE_DEVICES=6 .venv/bin/python train.py`), fresh
store, verbatim highlights (the benign uvicorn teardown tracebacks
elided — FINDINGS F-19):

    [collect] target: {'episodes': 9, 'rounds': 1, 'train_rows': 9, 'deadline_s': 1800.0}
    [collect] episode: {'uid': 'ep-31e486a50d5c0d11', 'finish_state': 'completed', 'gate_failures': (), 'wall_s': 8.895}
    ...  [9 episodes, walls 7.2-14.4 s]
    [collect] done: {'exit_code': 0, 'reason': 'target reached: 9/9 new trainable'}
    [sft] CollectReport: exit=0 reason='target reached: 9/9 new trainable' attempted=9 new_trainable=9 counts={'completed': 9} gate_failures={} wall=54.2s
    [sft] ep-31e486a50d5c0d11: gates=[] (G2/G3/G5 green) mounts=2 G2=f56e8a6e9ea9dd1c... G3=a7a7956b4842b79f...
    ...  [identical on all 9 records]
    [sft] spot-check: 9 clean / 9 landed
    [sft] tokenizer-hash assert OK: 949e1ec83f61520a25c75426edc4a43acc36f29a
    [sft] step  0 loss 0.2371 positions 932 | lag_histogram {0: 2}
    [sft] step  1 loss 0.1328 positions 908 | lag_histogram {0: 4}
    [sft] step  2 loss 0.1709 positions 760 | lag_histogram {0: 6}
    [sft] step  3 loss 0.2541 positions 544 | lag_histogram {0: 8}
    [sft] adapter saved to /home/sysadmin/gsj-envloader-examples/sft/adapters/run1
    [sft] consistency: committed serves = 8, steps x batch = 4 x 2 = 8 -> OK; retired 8

Exit 0. Every environment input arrived over an endpoint: cases from the
staging Forgejo, retrieval via the remote MCP service under per-episode
JWTs (mounts = 2: checkout + agent dir, nothing else), pins fetched
sha-verified by the library, the committed taskbank sha-verified in
place. The G2 docker singleton (`f56e8a6e…`) and G3 roster (`a7a7956b…`)
reproduced on all nine records.
