# opd — on-policy distillation, zero-CLI

Same substrate as sft/, two additions folded into the SAME
`python train.py`: an in-process **attach step** (`score.score(config)` —
the teacher's per-token logp over each tape, written into the `opd.`
namespace against the **always-on teacher endpoint**; the serve-swap era
is over) and a divergence-consuming loss (detached score-function RKL
surrogate over `OPDCollator`'s `[B, L]` teacher column). The config's
ready dict is what makes the loop on-policy: `serve_count: 0` (never
re-serve), `policy_lag lte 0` (fresh), `opd._complete` (scored), and a
`mix` pinning 100% student tapes.

The teacher lives under `user.teacher` (base_url + model_id + revision):
the library's `serving:` section is the single student endpoint by
schema, and teacher scoring is consumer craft — §9's developer half. The
prod swap for BOTH endpoints is two URLs, nothing else.

## Prepare (once)

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt -r collector-requirements.txt
.venv/bin/pip install --no-deps sglang==0.5.10.post1 \
  "verl @ git+https://github.com/verl-project/uni-agent.git@73b0f41efa88b311fd69129c6f835c012e925e73#subdirectory=verl"
```

Prerequisites beyond that (root README): the staging estate up — Forgejo,
MCP service `ready`, the **student** vLLM (`serving.base_url`) AND the
**teacher** vLLM (`user.teacher.base_url`, `Qwen/Qwen3-4B` with
prompt-logprobs available — stock vLLM has it) — plus the sandbox image
and `GSJ_MCP_TOKEN_SECRET`.

## What to edit

1. Endpoint hosts in `config.yaml` (Forgejo, MCP, student serving, and
   `user.teacher.base_url`) — your estate's topology.
2. The scratch paths + the taskbank's absolute path — your checkout.
3. `user:` (lr / steps / out) — yours.

## Run

```bash
export GSJ_MCP_TOKEN_SECRET=<the estate's secret>
CUDA_VISIBLE_DEVICES=6 .venv/bin/python train.py
```

One command: collect 9 episodes (student endpoint) → score them against
the teacher endpoint (§6.2 tokenizer-identity assert, the
+1-generation-slot window pre-check, write-once attach) → train 4 steps →
save the adapter → print accounting. Watch `mean_div` — THAT is the
per-step RKL estimate (the surrogate `loss` is not a divergence).

## Recorded run (H200, 2026-08-07 — the CP-31 zero-CLI proof, v0.6.0)

See the run transcript in this README's section below once recorded.
