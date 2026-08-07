# FINDINGS — the external register

Every friction hit while building and running these projects against the
published gsj-envloader artifacts, as an external developer. Severity:
**BLOCKER** (could not proceed on that path) / **FRICTION** (workaround
required) / **DOC** (docs insufficient, ambiguous, or silent) /
**COSMETIC**. `library-side: yes` rows are candidates for the library's
own register. Each row ends with a one-line reproduction.

| id | where | severity | what happened | root cause | workaround | library-side?  status (CP-28, library v0.5.0) |
|---|---|---|---|---|---|---|---|
| F-01 | setup_collector.sh, collector deps | FRICTION | `pip install -r` of the frozen uniagent requirements dies with ResolutionImpossible (sglang pin vs openai pin) on linux | the freeze was captured with sglang installed `--no-deps` — that install-mode nuance lives only in the file's prose header, the file itself is not flat-installable | install everything-but-sglang via `-r`, then the sglang pin `--no-deps` (setup_collector.sh) | yes — publish an installable lockfile/constraints, or encode the recipe in a script. Repro: `pip install -r devharness/uniagent/requirements.txt` on linux | **closed (v0.5.0)** — the recipe is published as an executable script (`install_collector_env.sh`, fetched at the tag and run by `setup_collector.sh`); the inline grep dance here is deleted |
| F-02 | setup, gate pins | FRICTION | the pins file (`pins.dev.json`) is not a published artifact and cannot be regenerated externally (`gsj-pin` needs the capture harness) | pins are generated library-repo DATA; publishing.md publishes image/cases/wheel/pages but not the pins the collector config requires | byte-copy from the library source tree @ v0.4.0, committed as `pins/` with a provenance note | yes — a release asset (or doc naming the copy recipe). Repro: search docs/publishing.md for "pins" — absent | **closed (v0.5.0)** — `gsj-pins-0.5.0.json` is a release asset (sha-verified download in `setup_collector.sh`); the committed `pins/` copy is deleted |
| F-03 | setup, render templates | DOC | the three `.pi` templates (`templates_dir`) are load-bearing G4/G7-pinned TEMPLATE data, distributed only inside the library repo source tree | not part of the published artifact set | fetch raw from the library repo @ tag v0.4.0 (setup_collector.sh) | yes — same class as F-02. Repro: grep docs/publishing.md for "templates" — absent | **closed (v0.5.0)** — the templates ship as package data in the wheel (configs omit `templates_dir`) and as release asset `gsj-pi-templates-0.5.0.tar.gz`; the raw-fetch is deleted |
| F-04 | configs, `mcp_launch.server` | FRICTION | the config cannot name the image's baked shim (`/opt/gsj-mcp/server.py` would be host-dereferenced); the parent-dir bind mount over `/opt/gsj-mcp` is unconditional | library HOLES H-28 (known, open) — no config key suppresses/retargets the mount | `docker create`+`docker cp` the shim out of the image, point `server:` at the extracted copy — the mount then re-mounts the image's own bytes | yes — H-28, experienced exactly as registered. Repro: set `mcp_launch.server: /opt/gsj-mcp/server.py` and watch the empty host-path mount shadow the baked shim | **closed (v0.5.0)** — `mcp_launch.in_image: true` names the baked `/opt/gsj-mcp/server.py` directly (no mount emitted); the docker-cp extraction is deleted |
| F-05 | setup, collector venv torch | COSMETIC | the frozen `torch==2.13.0` resolves to a cu130 build on a CUDA-12.8-capped host; `torch.cuda.is_available()` is False in the collector venv | the freeze pins a PyPI version whose linux wheel targets a newer CUDA; no linux install note exists | none needed — the collector path does no GPU work (generation is delegated to vLLM); noted so the warning is not chased | yes (doc note) — repro: import torch in the collector venv on a 12.8-driver host | **closed (v0.5.0)** — documented beside the install recipe (library publishing.md §4: harmless, the collector does no GPU work) |
| F-06 | build_taskbank.py | COSMETIC | `CaseSpec` is the one taskbank name not importable from the package root | library HOLES H-29 (known) | `from gsj.envloader.taskbank import CaseSpec` | yes — H-29. Repro: `python -c "from gsj.envloader import CaseSpec"` | **closed (v0.5.0)** — `CaseSpec` is root-exported; the submodule-path import is deleted |
| F-07 | configs, docker mode | FRICTION | `pi_launch.entry`, `pi_launch.extension`, `mcp_launch.python` are schema-required but never used in docker mode (the in-image paths run) — a stranger must invent values for fields that do nothing | one schema serves both execution modes; the docker-mode field subset is not marked | set them to the in-image paths as self-documenting placeholders | yes — doc the docker-mode field subset (or relax the schema). Repro: docker-mode config with `pi_launch.entry: /nonexistent` collects fine | **closed (v0.5.0)** — config-reference marks `pi_launch.entry`/`node_bin`/`extension` and `mcp_launch.python` subprocess-mode-only (the schema keeps them required; the placeholder-values idiom stands, now documented) |
| F-08 | seeding, gsj-run | FRICTION | no bounded "collect one round and exit": `gsj-run` runs the service loop forever; with `regenerate: wait_all` it idles at round-complete and must be killed by the operator watching the store | the round/seeding provisioning knobs have no library-schema home (the upstream `user.seeding` convention is read by library-repo dev tooling that is not part of the wheel) | run `gsj-run` in background, poll the store's unconsumed count, SIGINT at target | yes — a `--rounds 1`/exit-at-round-complete flag, or ship the seeding tool. Repro: `gsj-run --config sft/config.yaml` never exits | **closed (v0.5.0)** — `gsj-collect`: bounded exit at `--episodes`/`--rounds`/one-round default with stated exit codes; `collector.seeding` is the schema home |
| F-09 | configs, `driver.snapshot_path` | FRICTION | the codec tokenizer source is a host-bound resolved HF-cache path that must be hand-edited per host (no env interpolation in the config) | library HOLES H-27 (known): the file pins the resolution, not the (id, revision) pair | setup_collector.sh downloads the pinned snapshot and prints the exact path to paste | yes — H-27. Repro: move the config to any other host | **closed (v0.5.0)** — `driver: {model_id, revision}` resolves via the HF cache at first run; the hand-pasted snapshot path is deleted |
| F-10 | rlvr/grade.py | DOC | the record's `env.artifact.path` is checkout-relative and the checkout is reset after harvest — artifact *content* lives under `<episodes_root>/<uid>/harvest/`, which no record field names | library HOLES H-22 (known): the forensics layout is a tooling convention, not a contract surface | read `task.episodes_root` from the same config file and join the convention path | yes — H-22; softened by the one-file config carrying `episodes_root`. Repro: locate a deliverable from a record alone | **closed (v0.5.0)** — the `<episodes_root>/<uid>/harvest/<path>` convention is documented (config-reference `episodes_root` row + consumer notes, README §4) |
| F-11 | opd/train.py vs rlvr/train.py | DOC | README §4 says length budgeting is "trainer craft (micro-batching)" but never quantifies the hazard: a single truncated-at-context tape pads the batch to L≈32k and a full-batch vocab-wide float32 log-softmax is ~40 GB — the doc-faithful OPD loop carries that latent OOM while RLVR needed the `logits_to_keep` micro-batch form | no `L_max × V` warning near the no-truncation note / `torch_batches` docs | micro-batch + `logits_to_keep` (rlvr/train.py); opd/train.py deliberately keeps the doc-taught full-batch form as the fidelity probe | yes — one README sentence naming the L×V arithmetic. Repro: serve one 32k tape into a full-batch log-softmax loss | **closed (v0.5.0)** — the `L_max × V` (~40 GB) arithmetic is documented next to the no-truncation note (README §4, collate docstring, config-reference notes) |
| F-12 | attach jobs | DOC | `store.attach(uid, ns, payload)` accepts both bare (`{"reward": …}`) and prefixed (`{"rlvr.reward": …}`) payload keys — nothing consumer-facing says which is canonical; we learned it from the store source | key normalization is implemented but undocumented; the library's own two example jobs use opposite forms | either works; bare keys used here | yes — one docstring line. Repro: attach both forms, read the same column back | **closed (v0.5.0)** — bare keys documented canonical (README §4 + config-reference notes; the `attach` docstring line itself waits on a store.py freeze-lift — store.py was frozen in CP-28) |
| F-13 | all three config.yaml | FRICTION | every absolute path is duplicated across three files and hand-edited on any host change; no env-var interpolation exists (documented as deliberate) | the one-file design trades interpolation away (credential-hygiene rationale) | keep the paths trivially greppable (one root prefix); setup prints the two values that vary | yes (accepted design — a doc'd `sed` recipe would do). Repro: relocate the checkout, count edits | **open — accepted-by-design** (library HOLES H-36): no interpolation is deliberate (credential hygiene); the one-root-prefix grep-and-sed convention is the softener |

## Run-phase findings

| id | where | severity | what happened | root cause | workaround | library-side?  status (CP-28, library v0.5.0) |
|---|---|---|---|---|---|---|---|
| F-14 | stopping gsj-run | FRICTION | a detached `gsj-run` ignores both SIGINT and SIGTERM (the embedded per-session gateway server installs its own handlers); only SIGKILL stops it | `run_forever` relies on KeyboardInterrupt, which the uvicorn session servers intercept | `pkill -KILL` after confirming round-complete via the store | yes — compounding F-08. Repro: `kill -INT`/`kill -TERM` an idling gsj-run; it survives both | **closed (v0.5.0)** — `gsj-collect` installs real SIGINT/SIGTERM handlers: first signal = graceful drain (config/flag bound), second signal or drain expiry = stated hard exit; proven live in the CP-28 re-run |
| F-15 | opd/score.py, live | FRICTION+DOC | the first scoring run died on HTTP 400: a context-truncated tape has P+R = the serving window, and the completions route spends a +1 generation slot (`max_tokens=0` is rejected) — a tape AT the window cannot be scored through the API; nothing consumer-facing warns about the +1 slot | serving-API prompt-scoring needs one generation slot; the library's own example carries this lore inline, the docs don't | window pre-check + per-record containment (score.py); the record stays pending, `opd._complete` never set, the ready dict walls it off — the layered contracts turned a poison record into a clean exclusion | yes — one doc sentence on the scoring recipe. Repro: score a P+R=32768 tape against a 32768-window teacher | **closed (v0.5.0)** — the +1-generation-slot constraint is documented (README §4 Level 2 + config-reference consumer notes); score.py's pre-check stays as the worked implementation |
| F-16 | rlvr, live | DOC | at demo scale (9 episodes, 900 s wall, Qwen3-0.6B) the citation reward graded **all-zero** (2 artifacts, 0 citations; 7 no-shows) — RLVR training executes correctly but every step has zero advantage and zero gradient | model quality: the 0.6B rarely writes citing deliverables (a known upstream model-family gap), and 9 episodes is below the scale at which nonzero rewards appear (upstream saw 1 in 24) | none applied — recorded honestly; more episodes or a stronger actor are the real fixes | no (model/scale, not the library) — but a doc note on expected reward sparsity at demo scale would spare the next consumer the confusion | **external/model-scale** — unchanged: the substrate is correct, the 0.6B at 9 episodes rarely earns a citation reward; a stronger actor or more episodes is the fix |
| F-17 | gsj-run observability | FRICTION | `gsj-run` prints **nothing** about progress — no episode started/completed lines, no round status; its entire stdout is a CUDA warning plus benign-but-alarming uvicorn `CancelledError` tracebacks labeled ERROR at every session teardown; the only way to watch a run is to poll the store from a second process | metrics go to an in-process `InMemoryMetrics` sink nothing scrapes; per-session server shutdown noise is unfiltered | store-polling loop (see the project READMEs); mentally filter the tracebacks | yes — progress logging (or a metrics dump) + log hygiene. Repro: run any seed and watch the log | **closed (v0.5.0)** — per-episode start/finish lines, periodic `n/target trainable` progress, a final store summary, and the uvicorn `CancelledError` teardown filter (`--verbose` unfilters); proven live in the CP-28 re-run |

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

## CP-31 — the zero-CLI run (library v0.6.0, endpoint-only)

The claim under test: *install a wheel, edit endpoint values in one YAML,
run `python train.py`, and train — no CLI invoked, no scripts fetched, no
mounts configured, no data staged.* All three regimes ran green on the
staging estate (H200, 2026-08-07), each as ONE command from ONE venv.
New rows:

| id | where | severity | what happened | root cause | workaround | library-side? | status |
|---|---|---|---|---|---|---|---|
| F-18 | venv build, all three projects | FRICTION | one venv now runs both halves (`collect_episodes` is in-process since 0.6.0) — but BUILDING it takes three pip invocations, not one `pip install -r`: the frozen collector set is still not flat-installable (the sglang pin was captured `--no-deps` — F-01's shadow, now over every consumer instead of one shared collector venv), and verl installs `--no-deps` from the uni-agent submodule | the CP-05 freeze's install-mode nuances have no flat-file expression | a committed `collector-requirements.txt` (the freeze minus sglang, provenance header) + one documented `--no-deps` pip line; `setup_collector.sh` DELETED — pip against public URLs is the whole story (uni-agent and verl install non-editable from git direct references; pip initializes the submodule itself) | yes — publish an installable constraints set, or make the hermes-parser dep optional. Repro: `pip install -r` the raw freeze on linux | **closed** (library CP-32, ADR-0045 — the canonical generated `collector-requirements.txt` lives in the library repo + `driver_factory` refuses a parser-less env; the per-project copies are deleted, the venv is 2 pip invocations) |
| F-19 | every `collect_episodes` run, live | FRICTION | the CP-27 F-17 uvicorn `CancelledError` teardown tracebacks are BACK: the library call installs no log filters (documented, correct for a library), but the CLI's filter is not exported/documented as the consumer answer — the recorded OPD run interleaved 12 alarming `ERROR Traceback` blocks into an otherwise clean transcript | log hygiene was fixed CLI-side only (v0.5.0); `gsj.envloader.collect.install_log_filter()` exists but is discoverable only by reading source | none applied (transcripts read around the noise) | yes — document/root-export the filter (one README sentence minimum). Repro: any in-process collection; count `^ERROR` lines | **closed** (library CP-32 — `install_log_filter` root-exported at 0.7.0; `train.py` calls it via the 0.6.0-compatible module path, `--no-log-filter` opts out) |
| F-20 | sft collection target vs its ready (pre-run review; did NOT bite) | DOC | `collect_episodes` counts "trainable" as completed OR truncated, but SFT's ready serves completed-only — a truncated episode counts toward the 9-episode target yet never serves, silently shortening the training run | the target's trainable definition is regime-agnostic; a regime's ready can be stricter | over-provision `collector.seeding.episodes` (or widen the ready); the recorded run was 9/9 completed, so nothing was lost | yes — one doc sentence next to `collector.seeding.episodes`. Repro: a truncated episode under a completed-only ready | **open** → library H-45 (doc) |
| F-21 | opd collection, live | COSMETIC (recorded as evidence, not friction) | one generation request against the student endpoint hung; the episode burned its full 480 s wall, was killed and classified `infra_error`, and the collector retried the row — 10 attempted / 9 trainable, exit 0, the poison record hygiene-quarantined below every ready | a transient serving hiccup — exactly what the wall + retry + hygiene machinery exists for | none needed — the containment IS the observation | no — the machinery worked as designed (the 480 s wall dominated the run's 528 s wall-clock; without the hiccup collection is ~50 s) | recorded |

### The three recorded runs, in numbers

- **sft**: collect 9/9 completed (walls 7.2–14.4 s, total 54.2 s) → spot-check 9 clean/9 (G2 `f56e8a6e…`, G3 `a7a7956b…`, mounts=2 on every record) → 4 steps, loss 0.2371→0.2541, lag `{0: 8}` → `committed 8 = 4×2 OK; retired 8` → exit 0.
- **opd**: collect 9/9 completed +1 infra_error retry (wall 528.2 s, F-21) → score **9/9 against the always-on teacher endpoint** (`:8101`, window 32768; zero skips — no marathon tape this run; means −0.28…−1.78; §6.2 OID `949e1ec8…` green through the endpoint pair; **no serve swap existed anywhere**) → 4 micro-batched steps, RKL +1.1004→+0.8971 → `committed 8 = 4×2 OK; retired 8` → exit 0.
- **rlvr**: collect 9/9 completed (walls 6.5–11.0 s, total 46.2 s) → grade 9/9 with ground truth from the MCP service `/health` census (`{case_0001: 18, case_0002: 22, case_0003: 15, case_0004: 20}` — no pages tree anywhere), distribution 9 zero / 0 nonzero (2 artifacts citing nothing, 7 absent — F-16's expected sparsity, recorded honestly) → 4 degenerate REINFORCE steps (zero advantage ⇒ zero gradient, loss +0.0000) → `committed 8 = 4×2 OK; retired 8` → exit 0.

### What worked without friction — the endpoint-only additions

- ONE command per project did everything: collect (remote MCP under
  per-episode JWTs, cases cloned from the staging Forgejo, pins fetched
  sha-verified into the content-addressed cache) → attach in-process →
  train → adapter save → exact accounting. No subprocess, no shell-out,
  no `gsj-collect` anywhere in any `train.py`.
- The committed taskbanks rebuilt **byte-identical** to the hosted staging
  artifact (sha256 `9eb8e3c2…19da`) — the config's local default and its
  commented URL alternative are provably the same bytes, and the sha pin
  verifies the local file on every run.
- `torch==2.13.0` from the cu129 index satisfies the freeze pin verbatim
  AND sees CUDA on the 12.8 driver — F-05 is fully dissolved (the CP-27
  "collector venv can't see CUDA" era is over; one torch for both halves).
- The always-on teacher killed the serve swap: OPD's scorer read
  `user.teacher.base_url` and scored immediately after collection, in the
  same process — CP-27's ~6 min of swap choreography is gone.
- The G2 docker singleton, G3 roster, and G5 cutoff held on all 27 clean
  records across the three fresh stores, under fresh work_roots, through
  the remote transport — zero gate failures anywhere.

## What a stranger needs that doesn't exist yet — rewritten against v0.6.0 (endpoint-only)

The CP-27 list burned down at v0.5.0; the CP-28 list (an index wheel, the
path convention, reward sparsity) now reads:

1. **An index-published wheel** — unchanged; the one-line swap the
   requirements files anticipate (install-by-URL stands in).
2. **A flat-installable collector stack** (F-18, the one real venv
   friction left): until a constraints set is published, the venv is three
   pip invocations, and `collector-requirements.txt` is this repo's
   committed bridge.
3. **A running estate** — by design, not a gap: the Forgejo host, the MCP
   service, the two serving endpoints, and a docker daemon with the image
   are operator infrastructure the consumer points URLs at. The prod swap
   is those URLs, nothing else.
4. **Config path portability stays a convention** (F-13, accepted): one
   root prefix, edited per host.
5. **Demo-scale RLVR reward sparsity** (F-16, external): expect all-zero
   rewards at 9 episodes on the 0.6B — the CP-31 run reproduced it
   exactly (9 zero / 0 nonzero).
