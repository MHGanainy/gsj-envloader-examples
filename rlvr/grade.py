"""Verifiable-reward grading — the RLVR attach step, in-process (CP-31: an
importable function `train.py` calls, not a CLI). Written from the
library README §4 Level 2 by an external consumer.

The reward is verifiable from ENDPOINTS alone: the episode's deliverable
cites pages as `page:N` (the case repos' AGENTS.md convention), scored
against (a) the case's page count — read from the MCP service's own
public `/health` census (the same service the episodes retrieved from;
no shipped pages tree exists in the endpoint-only world) — and (b) the
row's timestep cutoff:

    reward = citations within cutoff / max(total citations, 1)

No artifact => reward 0.0, graded, never skipped: absence of work is a
verifiable outcome, and the advantage baseline in train.py needs that
mass in distribution.

Artifact resolution: the record carries only the checkout-relative
`env.artifact.path` ("out/<file>"), and the checkout is reset after
harvest — the durable copy lives under the episode forensics dir,
`<task.episodes_root>/<uid>/harvest/`, which the SAME config file names.

Idempotent: attach is write-once; a clean re-run grades 0.
"""

from __future__ import annotations

import json
import re
import urllib.request
from pathlib import Path

from gsj.envloader import TrajectoryStore

CITATION = re.compile(r"page:(\d+)")
GRADE_WHERE = {"env.outcome.finish_state": {"in": ["completed", "truncated"]}}


def page_counts_from_service(url_base: str) -> dict[str, int]:
    """case_id -> page count, from the MCP service's /health census."""
    url = f"{url_base.rstrip('/')}/health"
    with urllib.request.urlopen(url, timeout=10) as response:
        health = json.load(response)
    cases = health.get("cases") or {}
    counts = {case_id: int(info["pages"]) for case_id, info in cases.items()}
    if not counts:
        raise RuntimeError(f"{url}: no case census in /health "
                           f"(state={health.get('state')!r})")
    return counts


def grade(config) -> dict:
    """Grade every ungraded trainable record; returns a summary dict."""
    counts = page_counts_from_service(str(config.task.mcp_launch.url_base))
    print(f"[grade] page census from the MCP service /health: {counts}")
    episodes_root = Path(config.task.episodes_root)

    store = TrajectoryStore.open(config.store.root)
    uids = store.query(missing="rlvr._complete", where=GRADE_WHERE)
    print(f"[grade] {len(uids)} record(s) to grade in {config.store.root}")

    rewards: list[float] = []
    for uid, record in zip(uids, store.load(uids)):
        md = record.extra_info["tools_kwargs"]["task"]["metadata"]
        cutoff = min(int(md["timestep"]), counts[md["case_repo_id"]])
        rel = record.env.artifact.path
        artifact = (episodes_root / uid / "harvest" / Path(rel).relative_to("out")
                    if rel else None)
        if artifact is None or not artifact.is_file():
            cited: list[int] = []          # no artifact: reward 0.0, graded
        else:
            cited = [int(n) for n in
                     CITATION.findall(artifact.read_text(encoding="utf-8"))]
        n_valid = sum(1 for n in cited if 1 <= n <= cutoff)
        reward = n_valid / max(len(cited), 1)
        store.attach(uid, "rlvr", {
            "reward": float(reward),
            "n_cited": len(cited),
            "n_valid": n_valid,
            "grader_meta": {"grader": "cited-pages-within-cutoff",
                            "ground_truth": "mcp-service /health census",
                            "case": md["case_repo_id"],
                            "timestep": int(md["timestep"]), "cutoff": cutoff,
                            "artifact": rel, "v": 2},
        }, complete=True)
        rewards.append(reward)
        print(f"[grade]   {uid}: reward={reward:.3f} cited={len(cited)} "
              f"valid={n_valid} cutoff={cutoff} artifact={rel or 'ABSENT'}")

    zeros = sum(1 for r in rewards if r == 0.0)
    print(f"[grade] distribution over {len(rewards)} graded: {zeros} zero / "
          f"{len(rewards) - zeros} nonzero "
          f"(nonzero: {sorted(round(r, 3) for r in rewards if r > 0.0)})")
    remaining = store.query(missing="rlvr._complete", where=GRADE_WHERE)
    print(f"[grade] done: {len(remaining)} still pending "
          f"(a clean re-run grades 0 — attach is write-once)")
    store.close()
    return {"graded": len(rewards), "zeros": zeros, "rewards": rewards}
