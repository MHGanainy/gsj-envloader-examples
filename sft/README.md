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

## Recorded run

(added after the H200 run — see the section appended below)
