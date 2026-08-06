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
collector-venv/bin/gsj-run --config rlvr/config.yaml --driver uniagent
```

9 episodes at the 900 s wall (budget ~15 min worst case; usually far
less). The service idles at `round_complete` — Ctrl-C it (FINDINGS F-08).

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

## Recorded run

(added after the H200 run — see the section appended below)
