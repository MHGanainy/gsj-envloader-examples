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
collector-venv/bin/gsj-run --config sft/config.yaml --driver uniagent
```

One round = 9 episodes (1 per train row), each an ephemeral sandbox
container driven by pi against the served 0.6B, gate-verified (G1–G7),
finalized into `sft/store/`. With `regenerate: wait_all` the service
**idles** at `round_complete` when the round is done — watch the log and
Ctrl-C it; there is no collect-and-exit flag (FINDINGS F-08). Quick
doneness probe from another shell:

```bash
sft/.venv/bin/python -c "from gsj.envloader import TrajectoryStore; \
s = TrajectoryStore.open('sft/store'); print(len(s.query(where={'consumed': False}))); s.close()"
```

## Train

```bash
sft/.venv/bin/python sft/train.py --config sft/config.yaml
```

`user.steps` = 4 optimizer steps at `loader.batch_size` = 2; run-dry ends
iteration early if fewer than 8 records satisfy the ready dict. The
adapter lands in `sft/adapters/run1`; the accounting block at the end
shows per-record serve counts (the §7 commit-retires contract, observable).

## Recorded run (H200, 2026-08-06)

Seed: one `gsj-run` round, 9/9 episodes **completed**, all gate-clean
(zero quarantined), one per train row, in ≈8 min at concurrency 2.
Provenance on every record: `harness_image
ghcr.io/mhganainy/gsj-pi-harness:pi0.83.0-mcp1.5.0-2`, `execution
docker`, G2 `system_prompt_hash` = the container singleton
`f56e8a6e9ea9dd1c19be89c6754a4e8d3d1c0f89e04bb21f60237aa2e8837df4` —
reproduced under this repo's own fresh `work_root`.

Training, verbatim:

    [sft] device=cuda model=Qwen/Qwen3-0.6B steps=4 batch_size=2
    [sft] tokenizer-hash assert OK: 949e1ec83f61520a25c75426edc4a43acc36f29a
    [sft] step  0 loss 0.1169 positions 703 | lag_histogram {0: 2}
    [sft] step  1 loss 0.1835 positions 446 | lag_histogram {0: 4}
    [sft] step  2 loss 0.1991 positions 515 | lag_histogram {0: 6}
    [sft] step  3 loss 0.1476 positions 546 | lag_histogram {0: 8}
    [sft] adapter saved to sft/adapters/run1
    [sft] loss first=0.1169 last=0.1476 (4 optimizer steps)
    [sft] consistency: committed serves = 8, steps x batch = 4 x 2 = 8 -> OK; retired 8

The 9th record stayed unserved (batch size 2 does not divide 9 — the
run-dry contract leaves the stranded tail visible in the accounting).
