# sft — supervised fine-tuning on served trajectories

The smallest consumer: serve **completed** episodes, teacher-force the
student over its own recorded tokens, take a CE step on exactly the
positions the collector marked trainable. No attach job, no mix — the
base record already carries everything SFT needs. The training loop is
the library README's five-line story (`SFTCollator` bakes the target
alignment into `labels`; `model(**batch).loss` is the whole loss).

## Prepare (once)

From the repo root: `./setup_collector.sh`, then

```bash
cd sft
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
```

Prerequisites beyond that (root README): the sandbox image in the local
docker daemon (anonymous GHCR pull, or save|load on egress-locked hosts)
and vLLM serving Qwen3-0.6B at `serving.base_url`.

## Seed the store

```bash
cd ..   # repo root
collector-venv/bin/gsj-collect --config sft/config.yaml
```

9 episodes (the config's `collector.seeding.episodes` target; CLI flags
override), each an ephemeral sandbox container driven by pi against the
served 0.6B, gate-verified (G1–G7), finalized into `sft/store/`.
`gsj-collect` (library 0.5.0 — CP-27's F-08/F-14/F-17 closed) prints one
line per episode start/finish plus periodic `n/target trainable` progress,
exits 0 at the target or round-complete, and drains in-flight episodes on
a single Ctrl-C (a second one hard-exits). No polling shell, no
`pkill` — the CP-27 recipe is deleted.

## Train

```bash
sft/.venv/bin/python sft/train.py --config sft/config.yaml
```

`user.steps` = 4 optimizer steps at `loader.batch_size` = 2; run-dry ends
iteration early if fewer than 8 records satisfy the ready dict. The
adapter lands in `sft/adapters/run1`; the accounting block at the end
shows per-record serve counts (the §7 commit-retires contract, observable).

## Recorded run (H200, 2026-08-07 — the CP-28 closure re-run, v0.5.0)

Seeded with `gsj-collect` off the config's `collector.seeding` block —
including a **deliberate SIGINT mid-collection** to prove the drain
(library F-14), then a flag-override re-run to target. Both transcripts
verbatim (the CUDA warning suppressed; note ZERO uvicorn teardown
tracebacks — the F-17 log filter at work):

    [collect] target: 9 new trainable episode(s) over 9 train row(s), deadline 1800s
    [collect] episode started (total 1)
    [collect] episode started (total 2)
    [collect] episode ep-04b07595ce4c6809 finish_state=completed gates=ok wall=10.9s
    [collect] episode ep-7bfb9a3cd8e8d0a1 finish_state=completed gates=ok wall=11.7s
    [collect] progress: 2/9 trainable, 2 attempted, spi=n/a, unconsumed=1133
    [collect] episode started (total 3)
    [collect] episode started (total 4)
    [collect] episode ep-7d9c14c609071cc8 finish_state=completed gates=ok wall=10.5s
    [collect] episode ep-de70d6d0945864ed finish_state=completed gates=ok wall=6.9s
    [collect] progress: 4/9 trainable, 4 attempted, spi=n/a, unconsumed=1781
    [collect] episode started (total 5)
    [collect] episode started (total 6)
    [collect] SIGINT — draining (bound 600s); signal again to hard-exit
    [collect] episode ep-6ad0b6bd21062111 finish_state=completed gates=ok wall=11.8s
    [collect] episode ep-971f0761cbe1e819 finish_state=completed gates=ok wall=8.2s
    [collect] progress: 6/9 trainable, 6 attempted, spi=n/a, unconsumed=2875
    [collect] done: interrupted — drained after the first signal (exit 3)
    [collect] store summary — 6 servable record(s):
    ...
    EXIT=3

The two in-flight episodes FINISHED (drained, gate-checked, stored) —
no `pkill`, no orphaned containers. Re-run to target (`--episodes 3`,
the flag overriding the config's 9 — the documented precedence):

    [collect] target: 3 new trainable episode(s) over 9 train row(s), deadline 1800s
    ...
    [collect] done: target reached: 4/3 new trainable (exit 0)
    [collect] store summary — 10 servable record(s):
    EXIT=0

(The concurrent last pass overshot by one — 4 attempted against target 3,
reported honestly.) Store: **10/10 completed, zero gate failures**;
provenance on every record: `harness_image
ghcr.io/mhganainy/gsj-pi-harness:pi0.83.0-mcp1.5.0-2`, `execution docker`,
G2 `system_prompt_hash` = the container singleton
`f56e8a6e9ea9dd1c19be89c6754a4e8d3d1c0f89e04bb21f60237aa2e8837df4` —
reproduced with the image's own baked shim (`in_image: true`, no mount)
and the wheel's packaged templates; codec `tokenizer_hash 949e1ec8…`
identical to the CP-27 run — the `{model_id, revision}` pin pair resolved
to exactly what the hand-pasted snapshot path did.

Training, verbatim (GPU 6):

    [sft] device=cuda model=Qwen/Qwen3-0.6B steps=4 batch_size=2
    [sft] tokenizer-hash assert OK: 949e1ec83f61520a25c75426edc4a43acc36f29a
    [sft] step  0 loss 0.1411 positions 414 | lag_histogram {0: 2}
    [sft] step  1 loss 0.1421 positions 748 | lag_histogram {0: 4}
    [sft] step  2 loss 0.1369 positions 482 | lag_histogram {0: 6}
    [sft] step  3 loss 0.1407 positions 583 | lag_histogram {0: 8}
    [sft] adapter saved to sft/adapters/run1
    [sft] loss first=0.1411 last=0.1407 (4 optimizer steps)
    [sft] consistency: committed serves = 8, steps x batch = 4 x 2 = 8 -> OK; retired 8

The CP-27 run (v0.4.0, with the workarounds) recorded 9/9 completed and
the same G2 singleton; it remains in this file's git history.
