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

One command, fresh store, verbatim highlights (teardown noise elided —
FINDINGS F-19). Collection absorbed one transient serving hiccup exactly
as designed (F-21): a hung generation request burned its 480 s wall into
`infra_error`, the row was retried, the loop reached target:

    [collect] episode: {'uid': 'ep-8e49a27ab360747e', 'finish_state': 'infra_error', 'wall_s': 480.434}
    [collect] episode: {'uid': 'ep-2be59fdda95579bc', 'finish_state': 'completed', 'wall_s': 7.704}
    ...
    [collect] done: {'exit_code': 0, 'reason': 'target reached: 9/9 new trainable'}
    [opd] CollectReport: exit=0 reason='target reached: 9/9 new trainable' attempted=10 new_trainable=9 counts={'infra_error': 1, 'completed': 9} gate_failures={} wall=528.2s
    [opd] spot-check: 10 clean / 10 landed
    [score] teacher Qwen/Qwen3-4B@1cfa9a7208912126459214e8b04321603b3df60c served as 'Qwen/Qwen3-4B' at http://127.0.0.1:8101/v1 (window 32768); tokenizer OID 949e1ec83f61520a25c75426edc4a43acc36f29a
    [score] 9 record(s) to score in /home/sysadmin/gsj-envloader-examples/opd/store
    [score]   ep-2be59fdda95579bc: R=2375 masked=239 mean_teacher_logp=-1.7831
    ...  [9 records, means -0.28..-1.78]
    [score] done: scored 9; 0 still pending (a clean re-run retries exactly those — attach is write-once)
    [opd] tokenizer-hash assert OK: 949e1ec83f61520a25c75426edc4a43acc36f29a
    [opd] step  0 loss -1.0135 mean_div +1.1004 positions 637 | lag_histogram {0: 2}
    [opd] step  1 loss -0.3120 mean_div +0.4601 positions 976 | lag_histogram {0: 4}
    [opd] step  2 loss -0.7366 mean_div +0.7513 positions 778 | lag_histogram {0: 6}
    [opd] step  3 loss -0.9144 mean_div +0.8971 positions 693 | lag_histogram {0: 8}
    [opd] adapter saved to /home/sysadmin/gsj-envloader-examples/opd/adapters/run1
    [opd] RKL estimate first=+1.1004 last=+0.8971; loss first=-1.0135 last=-0.9144 (4 optimizer steps)
    [opd] consistency: committed serves = 8, steps x batch = 4 x 2 = 8 -> OK; retired 8

Exit 0. **Scoring ran immediately after collection, in the same process,
against the always-on teacher — no serve swap happened anywhere** (the
CP-27 recipe stopped/started vLLM around the scorer; that era is over).
All 9 scored this run (no window-limited tape landed); the §6.2 identity
held across the endpoint pair (the 0.6B and 4B tokenize identically —
OID `949e1ec8…`). The infra_error record is hygiene-quarantined below
every ready: 8 of the 9 scored records served exactly once and retired,
the 9th stranded below a full batch, visible in the accounting.
