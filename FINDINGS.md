# FINDINGS — the external register

Every friction hit while building and running these projects against the
published gsj-envloader artifacts, as an external developer. Severity:
**BLOCKER** (could not proceed on that path) / **FRICTION** (workaround
required) / **DOC** (docs insufficient, ambiguous, or silent) /
**COSMETIC**. `library-side: yes` rows are candidates for the library's
own register. Each row ends with a one-line reproduction.

| id | where | severity | what happened | root cause | workaround | library-side? |
|---|---|---|---|---|---|---|
| F-01 | setup_collector.sh, collector deps | FRICTION | `pip install -r` of the frozen uniagent requirements dies with ResolutionImpossible (sglang pin vs openai pin) on linux | the freeze was captured with sglang installed `--no-deps` — that install-mode nuance lives only in the file's prose header, the file itself is not flat-installable | install everything-but-sglang via `-r`, then the sglang pin `--no-deps` (setup_collector.sh) | yes — publish an installable lockfile/constraints, or encode the recipe in a script. Repro: `pip install -r devharness/uniagent/requirements.txt` on linux |
| F-02 | setup, gate pins | FRICTION | the pins file (`pins.dev.json`) is not a published artifact and cannot be regenerated externally (`gsj-pin` needs the capture harness) | pins are generated library-repo DATA; publishing.md publishes image/cases/wheel/pages but not the pins the collector config requires | byte-copy from the library source tree @ v0.4.0, committed as `pins/` with a provenance note | yes — a release asset (or doc naming the copy recipe). Repro: search docs/publishing.md for "pins" — absent |
| F-03 | setup, render templates | DOC | the three `.pi` templates (`templates_dir`) are load-bearing G4/G7-pinned TEMPLATE data, distributed only inside the library repo source tree | not part of the published artifact set | fetch raw from the library repo @ tag v0.4.0 (setup_collector.sh) | yes — same class as F-02. Repro: grep docs/publishing.md for "templates" — absent |
| F-04 | configs, `mcp_launch.server` | FRICTION | the config cannot name the image's baked shim (`/opt/gsj-mcp/server.py` would be host-dereferenced); the parent-dir bind mount over `/opt/gsj-mcp` is unconditional | library HOLES H-28 (known, open) — no config key suppresses/retargets the mount | `docker create`+`docker cp` the shim out of the image, point `server:` at the extracted copy — the mount then re-mounts the image's own bytes | yes — H-28, experienced exactly as registered. Repro: set `mcp_launch.server: /opt/gsj-mcp/server.py` and watch the empty host-path mount shadow the baked shim |
| F-05 | setup, collector venv torch | COSMETIC | the frozen `torch==2.13.0` resolves to a cu130 build on a CUDA-12.8-capped host; `torch.cuda.is_available()` is False in the collector venv | the freeze pins a PyPI version whose linux wheel targets a newer CUDA; no linux install note exists | none needed — the collector path does no GPU work (generation is delegated to vLLM); noted so the warning is not chased | yes (doc note) — repro: import torch in the collector venv on a 12.8-driver host |
| F-06 | build_taskbank.py | COSMETIC | `CaseSpec` is the one taskbank name not importable from the package root | library HOLES H-29 (known) | `from gsj.envloader.taskbank import CaseSpec` | yes — H-29. Repro: `python -c "from gsj.envloader import CaseSpec"` |
| F-07 | configs, docker mode | FRICTION | `pi_launch.entry`, `pi_launch.extension`, `mcp_launch.python` are schema-required but never used in docker mode (the in-image paths run) — a stranger must invent values for fields that do nothing | one schema serves both execution modes; the docker-mode field subset is not marked | set them to the in-image paths as self-documenting placeholders | yes — doc the docker-mode field subset (or relax the schema). Repro: docker-mode config with `pi_launch.entry: /nonexistent` collects fine |
| F-08 | seeding, gsj-run | FRICTION | no bounded "collect one round and exit": `gsj-run` runs the service loop forever; with `regenerate: wait_all` it idles at round-complete and must be killed by the operator watching the store | the round/seeding provisioning knobs have no library-schema home (the upstream `user.seeding` convention is read by library-repo dev tooling that is not part of the wheel) | run `gsj-run` in background, poll the store's unconsumed count, SIGINT at target | yes — a `--rounds 1`/exit-at-round-complete flag, or ship the seeding tool. Repro: `gsj-run --config sft/config.yaml` never exits |
| F-09 | configs, `driver.snapshot_path` | FRICTION | the codec tokenizer source is a host-bound resolved HF-cache path that must be hand-edited per host (no env interpolation in the config) | library HOLES H-27 (known): the file pins the resolution, not the (id, revision) pair | setup_collector.sh downloads the pinned snapshot and prints the exact path to paste | yes — H-27. Repro: move the config to any other host |
| F-10 | rlvr/grade.py | DOC | the record's `env.artifact.path` is checkout-relative and the checkout is reset after harvest — artifact *content* lives under `<episodes_root>/<uid>/harvest/`, which no record field names | library HOLES H-22 (known): the forensics layout is a tooling convention, not a contract surface | read `task.episodes_root` from the same config file and join the convention path | yes — H-22; softened by the one-file config carrying `episodes_root`. Repro: locate a deliverable from a record alone |
| F-11 | opd/train.py vs rlvr/train.py | DOC | README §4 says length budgeting is "trainer craft (micro-batching)" but never quantifies the hazard: a single truncated-at-context tape pads the batch to L≈32k and a full-batch vocab-wide float32 log-softmax is ~40 GB — the doc-faithful OPD loop carries that latent OOM while RLVR needed the `logits_to_keep` micro-batch form | no `L_max × V` warning near the no-truncation note / `torch_batches` docs | micro-batch + `logits_to_keep` (rlvr/train.py); opd/train.py deliberately keeps the doc-taught full-batch form as the fidelity probe | yes — one README sentence naming the L×V arithmetic. Repro: serve one 32k tape into a full-batch log-softmax loss |
| F-12 | attach jobs | DOC | `store.attach(uid, ns, payload)` accepts both bare (`{"reward": …}`) and prefixed (`{"rlvr.reward": …}`) payload keys — nothing consumer-facing says which is canonical; we learned it from the store source | key normalization is implemented but undocumented; the library's own two example jobs use opposite forms | either works; bare keys used here | yes — one docstring line. Repro: attach both forms, read the same column back |
| F-13 | all three config.yaml | FRICTION | every absolute path is duplicated across three files and hand-edited on any host change; no env-var interpolation exists (documented as deliberate) | the one-file design trades interpolation away (credential-hygiene rationale) | keep the paths trivially greppable (one root prefix); setup prints the two values that vary | yes (accepted design — a doc'd `sed` recipe would do). Repro: relocate the checkout, count edits |

## Run-phase findings

| id | where | severity | what happened | root cause | workaround | library-side? |
|---|---|---|---|---|---|---|
| F-14 | stopping gsj-run | FRICTION | a detached `gsj-run` ignores both SIGINT and SIGTERM (the embedded per-session gateway server installs its own handlers); only SIGKILL stops it | `run_forever` relies on KeyboardInterrupt, which the uvicorn session servers intercept | `pkill -KILL` after confirming round-complete via the store | yes — compounding F-08. Repro: `kill -INT`/`kill -TERM` an idling gsj-run; it survives both |
| F-15 | opd/score.py, live | FRICTION+DOC | the first scoring run died on HTTP 400: a context-truncated tape has P+R = the serving window, and the completions route spends a +1 generation slot (`max_tokens=0` is rejected) — a tape AT the window cannot be scored through the API; nothing consumer-facing warns about the +1 slot | serving-API prompt-scoring needs one generation slot; the library's own example carries this lore inline, the docs don't | window pre-check + per-record containment (score.py); the record stays pending, `opd._complete` never set, the ready dict walls it off — the layered contracts turned a poison record into a clean exclusion | yes — one doc sentence on the scoring recipe. Repro: score a P+R=32768 tape against a 32768-window teacher |
| F-16 | rlvr, live | DOC | at demo scale (9 episodes, 900 s wall, Qwen3-0.6B) the citation reward graded **all-zero** (2 artifacts, 0 citations; 7 no-shows) — RLVR training executes correctly but every step has zero advantage and zero gradient | model quality: the 0.6B rarely writes citing deliverables (a known upstream model-family gap), and 9 episodes is below the scale at which nonzero rewards appear (upstream saw 1 in 24) | none applied — recorded honestly; more episodes or a stronger actor are the real fixes | no (model/scale, not the library) — but a doc note on expected reward sparsity at demo scale would spare the next consumer the confusion |
| F-17 | gsj-run observability | FRICTION | `gsj-run` prints **nothing** about progress — no episode started/completed lines, no round status; its entire stdout is a CUDA warning plus benign-but-alarming uvicorn `CancelledError` tracebacks labeled ERROR at every session teardown; the only way to watch a run is to poll the store from a second process | metrics go to an in-process `InMemoryMetrics` sink nothing scrapes; per-session server shutdown noise is unfiltered | store-polling loop (see the project READMEs); mentally filter the tracebacks | yes — progress logging (or a metrics dump) + log hygiene. Repro: run any seed and watch the log |

## What worked without friction — for the record

- Anonymous consumption of every published artifact: wheel by URL (two
  hosts), GHCR image, case repos (`ls-remote` timesteps == the fixtures'
  manifest exactly), pages tarball (sha256 match).
- The one-file config: `load_config` fail-fast caught nothing because
  strict validation made the file right before first run; the SAME file
  drove `gsj-run`, both attach jobs, and all three trainers.
- Gates on a foreign host: 27/27 episodes gate-clean, zero quarantined;
  the G2 docker singleton (`f56e8a6e…`) reproduced under a brand-new
  `work_root` — host-path independence held exactly as documented.
- §6.2 identity: `check_tokenizer` green on every batch of all three
  trainers; the teacher-side hash assert green across the 0.6B/4B swap.
- §7 accounting: every run's `committed = steps × batch` exact; write-once
  attach idempotency observed live in both jobs.

## What a stranger needs that doesn't exist yet

1. **A published collection-environment artifact set**: the pins file
   (F-02), the render templates (F-03), and an *installable* frozen
   collector dependency spec (F-01) are all load-bearing and all live
   only inside the library repo's source tree today.
2. **A bounded, observable, stoppable collector run**: `gsj-run` has no
   collect-one-round-and-exit mode (F-08), swallows SIGINT/SIGTERM when
   detached (F-14), and reports no progress (F-17) — seeding a store
   currently means a second process polling the store and a SIGKILL.
3. **A docker-mode config profile**: three schema-required fields are
   dead in docker mode (F-07) and the snapshot path is host-bound (F-09).
4. **Four doc sentences**: the +1 generation slot in serving-API scoring
   (F-15), the L×V loss-memory arithmetic (F-11), attach-payload key
   normalization (F-12), and the `episodes_root` forensics convention
   (F-10).
5. **An index-published wheel** — the requirements one-line swap the
   library already anticipates.
