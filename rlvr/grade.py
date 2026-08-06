#!/usr/bin/env python3
"""Verifiable-reward grading — the RLVR attach job, written from the
library README §4 Level 2 by an external consumer.

The reward is verifiable from public inputs alone: the episode's
deliverable cites pages as `page:N` (the case repos' AGENTS.md
convention), and we score citations against (a) the case's page count —
derived by counting the page corpus this repo already carries
(`assets/pages/<case_id>/page_*.md`) — and (b) the row's timestep cutoff:

    reward = citations within cutoff / max(total citations, 1)

No artifact => reward 0.0, graded, never skipped: absence of work is a
verifiable outcome, and the advantage baseline in train.py needs that
mass in distribution.

Artifact resolution: the record carries only the checkout-relative
`env.artifact.path` ("out/<file>"), and the checkout is reset after
harvest — the durable copy lives under the episode forensics dir,
`<task.episodes_root>/<uid>/harvest/`, which the SAME config file names.

Offline (no GPU, no serving):  .venv/bin/python grade.py --config config.yaml
Idempotent: attach is write-once; a clean re-run grades 0.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

from gsj.envloader import TrajectoryStore, load_config

CITATION = re.compile(r"page:(\d+)")
GRADE_WHERE = {"env.outcome.finish_state": {"in": ["completed", "truncated"]}}


def page_counts(pages_root: Path) -> dict[str, int]:
    """case_id -> page count, from the corpus tree itself."""
    counts: dict[str, int] = {}
    for case_dir in sorted(pages_root.iterdir()):
        if case_dir.is_dir():
            counts[case_dir.name] = len(list(case_dir.glob("page_*.md")))
    if not counts:
        raise SystemExit(f"no case dirs under {pages_root}")
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()

    config = load_config(args.config)
    counts = page_counts(Path(config.task.mcp_launch.pages_root))
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
                            "case": md["case_repo_id"],
                            "timestep": int(md["timestep"]), "cutoff": cutoff,
                            "artifact": rel, "v": 1},
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


if __name__ == "__main__":
    main()
