#!/usr/bin/env python3
"""Teacher scoring — the OPD attach job, written from the library README §4
Level 2 ("query what is missing, compute, attach write-once") by an
external consumer.

Per unscored trainable record: assert the teacher tokenizes identically
(§6.2 — one `git_blob_oid` call against the served provenance), request
the teacher's per-token logprobs over the record's exact `input_ids` via
the served endpoint's prompt-logprobs capability (vLLM: the
`prompt_logprobs` extra body param on /v1/completions with a token-id
prompt), slice the response span, attach as full-R float32
`opd.teacher_logp_sampled` with `complete=True`.

Idempotent by construction: attach is write-once and the query's
`missing=` excludes already-scored records — a clean re-run scores 0.

Run AFTER swapping the serving endpoint to the teacher (Qwen3-4B here;
`user.teacher` in config.yaml pins its identity):

    .venv/bin/python score.py --config config.yaml
"""

from __future__ import annotations

import argparse
import json
import urllib.request
from pathlib import Path

import numpy as np
from huggingface_hub import hf_hub_download

from gsj.envloader import TrajectoryStore, git_blob_oid, load_config

# Trainable, hygiene-clean, student-behavior records only. (Hygiene —
# infra_error / gate failures — is already below every store query.)
SCORE_WHERE = {
    "env.outcome.finish_state": {"in": ["completed", "truncated"]},
    "env.policy.behavior": "student",
}


def post_json(url: str, payload: dict, timeout_s: float = 600.0) -> dict:
    req = urllib.request.Request(url, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout_s) as response:
        return json.load(response)


def served_model(endpoint: str) -> tuple[str, int | None]:
    with urllib.request.urlopen(f"{endpoint}/models", timeout=10) as response:
        data = json.load(response)["data"]
    if not data:
        raise SystemExit("serving endpoint lists no models")
    return data[0]["id"], data[0].get("max_model_len")


def teacher_logps(endpoint: str, model: str, ids: list[int]) -> list[float | None]:
    """Entry i is logp(token_i | tokens_<i) — target-aligned by the API's
    construction; entry 0 is None (no context, and P >= 1 keeps it out of
    the response span)."""
    body = post_json(f"{endpoint}/completions", {
        "model": model, "prompt": ids, "max_tokens": 1, "temperature": 0.0,
        "prompt_logprobs": 0,
    })
    rows = body["choices"][0].get("prompt_logprobs")
    if rows is None:
        raise SystemExit("endpoint returned no prompt_logprobs — not vLLM, or "
                         "the capability is disabled")
    out: list[float | None] = []
    for position, row in enumerate(rows):
        if row is None:
            out.append(None)
            continue
        key = str(ids[position])
        if key not in row:
            raise SystemExit(f"position {position}: token {key} missing from "
                             f"prompt_logprobs row")
        out.append(float(row[key]["logprob"]))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--endpoint", default=None,
                        help="teacher endpoint (default: serving.base_url from "
                             "the config — the teacher is served on the same "
                             "port after the swap)")
    args = parser.parse_args()

    config = load_config(args.config)
    endpoint = (args.endpoint or config.serving.base_url).rstrip("/")
    teacher = config.user.get("teacher", {})
    teacher_id = teacher.get("model_id", "Qwen/Qwen3-4B")
    teacher_rev = teacher.get("revision")

    # §6.2: the teacher MUST tokenize identically to the served provenance.
    tokenizer_json = Path(hf_hub_download(teacher_id, "tokenizer.json",
                                          revision=teacher_rev))
    teacher_hash = git_blob_oid(tokenizer_json)
    model, max_model_len = served_model(endpoint)
    print(f"[score] teacher {teacher_id}@{teacher_rev} served as {model!r} "
          f"(window {max_model_len}); tokenizer OID {teacher_hash}")

    store = TrajectoryStore.open(config.store.root)
    uids = store.query(missing="opd._complete", where=SCORE_WHERE)
    print(f"[score] {len(uids)} record(s) to score in {config.store.root}")
    meta = {"checkpoint": f"{teacher_id}@{teacher_rev}", "support": "sampled",
            "align": "target", "route": "prompt_logprobs", "v": 1}

    means: list[float] = []
    skipped: list[str] = []
    for uid, record in zip(uids, store.load(uids)):
        served_hash = record.env.provenance["codec"]["tokenizer_hash"]
        if served_hash != teacher_hash:
            raise SystemExit(f"tokenizer mismatch (§6.2): teacher {teacher_hash}"
                             f" != served {served_hash} (uid {uid})")
        P, R = len(record.prompts), len(record.responses)
        # The completions route spends one generation slot (max_tokens=1;
        # max_tokens=0 is rejected), so a tape AT the serving window cannot
        # be scored through the API — learned live on a context-truncated
        # tape (P+R = the window). Skip loudly; the record stays pending
        # (never satisfies opd._complete, so the ready dict walls it off).
        if max_model_len is not None and P + R + 1 > max_model_len:
            print(f"[score]   {uid}: SKIPPED — P+R+1 = {P + R + 1} exceeds the "
                  f"teacher window {max_model_len} (the +1 is the API's "
                  f"generation slot); needs a local-prefill fallback or a "
                  f"larger serving window")
            skipped.append(uid)
            continue
        try:
            logps = teacher_logps(endpoint, model,
                                  [int(t) for t in record.input_ids])
            response_logps = logps[P : P + R]
            if len(response_logps) != R or any(lp is None for lp in response_logps):
                raise RuntimeError(f"teacher returned {len(response_logps)} "
                                   f"logps for R={R}")
        except Exception as error:  # one poison record must not abort the run
            print(f"[score]   {uid}: FAILED — {error}")
            skipped.append(uid)
            continue
        column = np.asarray(response_logps, dtype=np.float32)  # full-R float32
        store.attach(uid, "opd", {"teacher_logp_sampled": column,
                                  "teacher_meta": meta}, complete=True)
        mask = np.asarray(record.loss_mask, dtype=bool)
        mean_lp = float(column[mask].mean()) if mask.any() else float("nan")
        means.append(mean_lp)
        print(f"[score]   {uid}: R={R} masked={int(mask.sum())} "
              f"mean_teacher_logp={mean_lp:.4f}")

    remaining = store.query(missing="opd._complete", where=SCORE_WHERE)
    print(f"[score] done: scored {len(means)}; {len(remaining)} still pending "
          f"(a clean re-run retries exactly those — attach is write-once)")
    store.close()
    if skipped:
        raise SystemExit(f"{len(skipped)} record(s) not scored (listed above); "
                         f"they stay pending and unserved")


if __name__ == "__main__":
    main()
