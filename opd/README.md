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
collector-venv/bin/gsj-run --config opd/config.yaml --driver uniagent
```

9 episodes (1 per train row); the service idles at `round_complete` —
Ctrl-C it (FINDINGS F-08).

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

## Recorded run

(added after the H200 run — see the section appended below)
