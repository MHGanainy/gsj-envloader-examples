# opd — on-policy distillation on served trajectories

Same substrate as sft/, two additions: an **attach job** (`score.py` —
the teacher's per-token logp over each tape, written into the
`opd.` namespace) and a divergence-consuming loss (detached
score-function RKL surrogate over `OPDCollator`'s `[B, L]` teacher
column). The config's ready dict is what makes the loop on-policy:
`serve_count: 0` (never re-serve), `policy_lag lte 0` (fresh),
`opd._complete` (scored), and a `mix` pinning 100% student tapes.

## Prepare (once)

From the repo root: `./setup_collector.sh`, then

```bash
cd opd
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
```

## Seed the store (student = 0.6B serving)

```bash
cd ..   # repo root
collector-venv/bin/gsj-collect --config opd/config.yaml
```

9 episodes (the config's `collector.seeding` target): bounded, observable,
stoppable — exits 0 at the target; one Ctrl-C drains (library 0.5.0,
F-08/F-14/F-17 closed).

## Score (teacher = 4B serving)

Swap the serving endpoint to the teacher — stop the 0.6B vLLM, start
`Qwen/Qwen3-4B` (revision pinned in `user.teacher`) on the same port —
then:

```bash
opd/.venv/bin/python opd/score.py --config opd/config.yaml
```

The scorer asserts the teacher tokenizes identically to the served
provenance (§6.2, one hash compare), fetches per-token prompt-logprobs
over each record's exact `input_ids` (vLLM `prompt_logprobs`), and
attaches full-R float32 `opd.teacher_logp_sampled`, write-once. Re-runs
score 0 — idempotent. Swap serving back to the 0.6B afterwards (training
itself needs no serving).

## Train

```bash
opd/.venv/bin/python opd/train.py --config opd/config.yaml
```

Watch `mean_div` — THAT is the per-step RKL estimate (the surrogate
`loss` is not a divergence). 4 steps × batch 2 over 9 scored records
(the 9th strands below a full batch — visible in the accounting).

## Recorded run (H200, 2026-08-06)

Seed: 9/9 episodes gate-clean (8 completed + 1 **context-truncated
marathon**, P=2964 R=29804), GHCR rev-2 provenance, G2 docker singleton.

Scoring (4B teacher at a 32768 window), verbatim highlights:

    [score] teacher Qwen/Qwen3-4B@1cfa9a72... served as 'Qwen/Qwen3-4B' (window 32768); tokenizer OID 949e1ec83f61520a25c75426edc4a43acc36f29a
    [score]   ep-e3d7709096fdb3c1: SKIPPED — P+R+1 = 32769 exceeds the teacher window 32768 (the +1 is the API's generation slot); ...
    [score]   ep-bd148e89ebe3349b: R=863 masked=291 mean_teacher_logp=-1.2227
    [score]   ep-caf930138ec82f3b: R=1563 masked=342 mean_teacher_logp=-1.0829
    [score] done: scored 7; 1 still pending (a clean re-run retries exactly those — attach is write-once)

(8 of 9 scored across two invocations — the first run died on the
marathon tape before containment existed: FINDINGS F-15. The unscored
record never satisfies `opd._complete`, so the ready dict walls it off —
the layered contracts turned a poison record into a clean exclusion.)

Training, verbatim:

    [opd] device=cuda model=Qwen/Qwen3-0.6B steps=4 batch_size=2
    [opd] tokenizer-hash assert OK: 949e1ec83f61520a25c75426edc4a43acc36f29a
    [opd] step  0 loss -0.5682 mean_div +0.9450 positions 653 | lag_histogram {0: 2}
    [opd] step  1 loss -0.8015 mean_div +1.0385 positions 495 | lag_histogram {0: 4}
    [opd] step  2 loss -0.8478 mean_div +0.8309 positions 622 | lag_histogram {0: 6}
    [opd] step  3 loss -1.1865 mean_div +0.8925 positions 522 | lag_histogram {0: 8}
    [opd] adapter saved to opd/adapters/run1
    [opd] RKL estimate first=+0.9450 last=+0.8925; loss first=-0.5682 last=-1.1865 (4 optimizer steps)
    [opd] consistency: committed serves = 8, steps x batch = 4 x 2 = 8 -> OK

The RKL estimate (`mean_div`) is positive and drifting down — the same
shape as the library's own recorded OPD run. The 8 scored records served
exactly once each and retired; the marathon shows `serve_count 0,
consumed False` in the store.
