# rlvr — verifiable reward, zero-CLI

Third regime, same substrate, one `python train.py`: collect → grade
(in-process — `grade.grade(config)`) → REINFORCE-style advantage-weighted
CE over `RLVRCollator` batches (`aux["rewards"]`). The reward is
verifiable from **endpoints alone**: `page:N` citations in the episode's
deliverable, checked against the case's page count read from the MCP
service's own `/health` census (the shipped-pages-tree era is over) and
the row's timestep cutoff. Absence of a deliverable grades 0.0 (never
skipped): the zero mass is what gives artifact-writing episodes positive
advantage. This is a demonstration of the substrate, not an RL library —
no PPO, no KL penalty, no value model.

The 900 s episode wall (vs 480 s elsewhere) is deliberate: the reward
grades the *deliverable*, and the model needs wall time to finish writing
it. Expect reward **sparsity** at this scale regardless (F-16): the 0.6B
rarely writes citing deliverables — all-zero rewards are the honest
common case, and training then steps with zero advantage and zero
gradient, exactly as REINFORCE prescribes.

## Prepare (once)

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt \
  -r https://raw.githubusercontent.com/MHGanainy/gsj-envloader/v0.7.0/devharness/uniagent/collector-requirements.txt
.venv/bin/pip install --no-deps sglang==0.5.10.post1 \
  "verl @ git+https://github.com/verl-project/uni-agent.git@73b0f41efa88b311fd69129c6f835c012e925e73#subdirectory=verl"
```

The collector set is the library's canonical GENERATED artifact
(gsj-envloader CP-32 / ADR-0045 — the per-project committed copies are
gone; the `--no-deps` line is also printed in the artifact's own header).
The v0.7.0 raw-at-tag URL goes live with the library's CP-33 release;
until then substitute any library commit-sha raw URL of the same path.

Prerequisites beyond that (root README): the staging estate up (Forgejo,
MCP service `ready`, student vLLM serving), the sandbox image, and
`GSJ_MCP_TOKEN_SECRET` exported.

## What to edit

1. Endpoint hosts in `config.yaml` (Forgejo, MCP, serving) — your
   estate's topology.
2. The scratch paths + the taskbank's absolute path — your checkout.
3. Training parameters (lr / steps / out) — the **TRAINING PARAMETERS**
   constants at the top of `train.py` (the CP-33 config split:
   `config.yaml` is the library surface only; the run is code).

## Run

```bash
export GSJ_MCP_TOKEN_SECRET=<the estate's secret>
CUDA_VISIBLE_DEVICES=6 .venv/bin/python train.py
```

One command: collect 9 episodes at the 900 s wall (budget ~15 min worst
case) → grade every trainable record (write-once, idempotent) → train
4 steps with per-item gradient accumulation + `logits_to_keep` (a 900 s
wall can produce a ~30k-token truncated tape; a full-batch vocab-wide
log-softmax at that length does not fit next to co-resident serving) →
save the adapter → print accounting.

## Recorded run (H200, 2026-08-07 — the CP-31 zero-CLI proof, v0.6.0)

One command, fresh store, verbatim highlights (teardown noise elided —
FINDINGS F-19):

    [collect] done: {'exit_code': 0, 'reason': 'target reached: 9/9 new trainable'}
    [rlvr] CollectReport: exit=0 reason='target reached: 9/9 new trainable' attempted=9 new_trainable=9 counts={'completed': 9} gate_failures={} wall=46.2s
    [rlvr] spot-check: 9 clean / 9 landed
    [grade] page census from the MCP service /health: {'case_0001': 18, 'case_0002': 22, 'case_0003': 15, 'case_0004': 20}
    [grade] 9 record(s) to grade in /home/sysadmin/gsj-envloader-examples/rlvr/store
    [grade]   ep-2da91a997b0b3c68: reward=0.000 cited=0 valid=0 cutoff=9 artifact=out/ep-2da91a997b0b3c68.md
    [grade]   ep-3a75dcb71ada1446: reward=0.000 cited=0 valid=0 cutoff=4 artifact=ABSENT
    ...  [7 ABSENT, 2 artifacts citing nothing]
    [grade] distribution over 9 graded: 9 zero / 0 nonzero (nonzero: [])
    [grade] done: 0 still pending (a clean re-run grades 0 — attach is write-once)
    [rlvr] tokenizer-hash assert OK: 949e1ec83f61520a25c75426edc4a43acc36f29a
    [rlvr] step  0 loss +0.0000 mean_reward 0.000 adv_spread 0.000 positions 414 | lag_histogram {0: 2}
    [rlvr] step  1 loss +0.0000 mean_reward 0.000 adv_spread 0.000 positions 632 | lag_histogram {0: 4}
    [rlvr] step  2 loss +0.0000 mean_reward 0.000 adv_spread 0.000 positions 861 | lag_histogram {0: 6}
    [rlvr] step  3 loss +0.0000 mean_reward 0.000 adv_spread 0.000 positions 790 | lag_histogram {0: 8}
    [rlvr] adapter saved to /home/sysadmin/gsj-envloader-examples/rlvr/adapters/run1
    [rlvr] consistency: committed serves = 8, steps x batch = 4 x 2 = 8 -> OK; retired 8

Exit 0. **All nine rewards graded 0.0** (2 artifacts citing nothing, 7
absent) — F-16's expected sparsity at this scale, reproduced and recorded
honestly; training then executes the degenerate case exactly as REINFORCE
prescribes (zero advantage, zero gradient) with exact accounting. The
grading ground truth came from the MCP service's `/health` census — the
same endpoint the episodes retrieved from; no pages tree exists anywhere
in this repo anymore. Episodes finished fast this run (walls 6.5–11.0 s;
the 900 s wall is headroom, not typical cost).
