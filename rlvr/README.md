# rlvr — verifiable reward on served trajectories

Third regime, same substrate: `grade.py` scores each episode's
deliverable **verifiably** — `page:N` citations checked against the page
corpus and the row's timestep cutoff — and attaches the scalar
`rlvr.reward`; `train.py` does REINFORCE-style advantage-weighted CE over
`RLVRCollator` batches (`aux["rewards"]`). Absence of a deliverable
grades 0.0 (never skipped): the zero mass is what gives artifact-writing
episodes positive advantage. This is a demonstration of the substrate,
not an RL library — no PPO, no KL penalty, no value model.

The 900 s episode wall (vs 480 s elsewhere) is deliberate: the reward
grades the *deliverable*, and the model needs wall time to finish writing
it — at 480 s the reward signal starves to all-zero.

## Prepare (once)

From the repo root: `./setup_collector.sh`, then

```bash
cd rlvr
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
```

## Seed the store

```bash
cd ..   # repo root
collector-venv/bin/gsj-collect --config rlvr/config.yaml
```

9 episodes at the 900 s wall (budget ~15 min worst case; usually far
less). Bounded and observable: exits 0 at the `collector.seeding` target;
one Ctrl-C drains (library 0.5.0, F-08/F-14/F-17 closed).

## Grade (offline — no GPU, no serving)

```bash
rlvr/.venv/bin/python rlvr/grade.py --config rlvr/config.yaml
```

Ground truth is derived from inputs this repo already has: page counts by
counting `assets/pages/<case>/page_*.md`, the cutoff from the row's
timestep, the artifact from `<episodes_root>/<uid>/harvest/` (the config
names `episodes_root`; the record's `env.artifact.path` alone would not
find it — FINDINGS F-10). Write-once: re-runs grade 0.

## Train

```bash
rlvr/.venv/bin/python rlvr/train.py --config rlvr/config.yaml
```

Per-item gradient accumulation with `logits_to_keep` — a 900 s wall can
produce a ~30k-token truncated tape, and a full-batch vocab-wide
log-softmax at that length does not fit next to a co-resident serving
engine (see the docstring). Zero-advantage batches step with zero
gradient; that is REINFORCE working as written, not a bug.

## Recorded run (H200, 2026-08-06)

Seed: 9/9 episodes **completed**, gate-clean, GHCR rev-2 provenance, G2
docker singleton, at the 900 s wall.

Grading, verbatim highlights:

    [grade] 9 record(s) to grade in /home/sysadmin/gsj-envloader-examples/rlvr/store
    [grade]   ep-dbc136375b626555: reward=0.000 cited=0 valid=0 cutoff=12 artifact=out/ep-dbc136375b626555.md
    [grade]   ep-cf2160edb9c1425c: reward=0.000 cited=0 valid=0 cutoff=5 artifact=ABSENT
    [grade] distribution over 9 graded: 9 zero / 0 nonzero (nonzero: [])
    [grade] done: 0 still pending (a clean re-run grades 0 — attach is write-once)

**All nine rewards graded 0.0** (2 artifacts citing nothing, 7 absent) —
the actor-model-quality reality at this scale, recorded honestly
(FINDINGS F-16; upstream saw 1 nonzero in 24 episodes on the same
model). Training then executes the degenerate case exactly as REINFORCE
prescribes — zero advantage, zero gradient, exact accounting:

    [rlvr] device=cuda model=Qwen/Qwen3-0.6B steps=4 batch_size=2
    [rlvr] tokenizer-hash assert OK: 949e1ec83f61520a25c75426edc4a43acc36f29a
    [rlvr] step  0 loss +0.0000 mean_reward 0.000 adv_spread 0.000 positions 542 | lag_histogram {0: 2}
    [rlvr] step  1 loss +0.0000 mean_reward 0.000 adv_spread 0.000 positions 727 | lag_histogram {0: 4}
    [rlvr] step  2 loss +0.0000 mean_reward 0.000 adv_spread 0.000 positions 663 | lag_histogram {0: 6}
    [rlvr] step  3 loss +0.0000 mean_reward 0.000 adv_spread 0.000 positions 618 | lag_histogram {0: 8}
    [rlvr] adapter saved to rlvr/adapters/run1
    [rlvr] consistency: committed serves = 8, steps x batch = 4 x 2 = 8 -> OK

The substrate (serving, collator `aux["rewards"]`, commit accounting)
is fully exercised; the reward signal needs more episodes or a stronger
actor to become non-degenerate.
