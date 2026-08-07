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
.venv/bin/pip install -r requirements.txt -r collector-requirements.txt
.venv/bin/pip install --no-deps sglang==0.5.10.post1 \
  "verl @ git+https://github.com/verl-project/uni-agent.git@73b0f41efa88b311fd69129c6f835c012e925e73#subdirectory=verl"
```

Prerequisites beyond that (root README): the staging estate up (Forgejo,
MCP service `ready`, student vLLM serving), the sandbox image, and
`GSJ_MCP_TOKEN_SECRET` exported.

## What to edit

1. Endpoint hosts in `config.yaml` (Forgejo, MCP, serving) — your
   estate's topology.
2. The scratch paths + the taskbank's absolute path — your checkout.
3. `user:` (lr / steps / out) — yours.

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

See the run transcript in this README's section below once recorded.
