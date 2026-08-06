#!/usr/bin/env python3
"""Build the three projects' taskbank.parquet from the PUBLIC case repos.

Inputs are public URLs only (library docs/publishing.md):
  - case repos  https://github.com/MHGanainy/gsj-case-000{1..4}   (anonymous)
  - sandbox     ghcr.io/mhganainy/gsj-pi-harness:pi0.83.0-mcp1.5.0-2

Timesteps are DISCOVERED, not copied from any manifest: each case repo
publishes one branch per historical timestep (`timestep-N`), so
`git ls-remote --heads` is the source of truth an external consumer
actually has. Rows are (case x timestep x prompt) with the `summarize`
skill prompt (resolved against the checkout at rollout — the row stores no
message text); the eval split is case-level: case_0004.

The parquet is written into each project dir (sft/, opd/, rlvr/) — three
identical copies, deliberately: each project is self-contained. The files
are committed so the repo shows its data, and regenerable with this script
(rerun it after the case repos change).

API note: every taskbank name imports from the package root — `CaseSpec`
included since library 0.5.0 (H-29 closed; the F-06 submodule-path
workaround is deleted).

Run (any venv with the wheel):
    python -m venv .venv && .venv/bin/pip install \
      "gsj-envloader @ https://github.com/MHGanainy/gsj-envloader/releases/download/v0.5.0/gsj_envloader-0.5.0-py3-none-any.whl"
    .venv/bin/python build_taskbank.py
"""

from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path

import pyarrow.parquet as pq

from gsj.envloader import CaseSpec, PromptSpec, build_taskbank, write_taskbank

CASES = {
    "case_0001": "https://github.com/MHGanainy/gsj-case-0001.git",
    "case_0002": "https://github.com/MHGanainy/gsj-case-0002.git",
    "case_0003": "https://github.com/MHGanainy/gsj-case-0003.git",
    "case_0004": "https://github.com/MHGanainy/gsj-case-0004.git",
}
EVAL_CASES = ["case_0004"]
SANDBOX_IMAGE = "ghcr.io/mhganainy/gsj-pi-harness:pi0.83.0-mcp1.5.0-2"
PROMPTS = [PromptSpec.skill("summarize")]
PROJECTS = ["sft", "opd", "rlvr"]
TIMESTEP_REF = re.compile(r"refs/heads/timestep-(\d+)$")


def discover_timesteps(url: str) -> list[int]:
    """`git ls-remote --heads` -> sorted timestep-N branch numbers.
    GIT_TERMINAL_PROMPT=0 asserts the anonymous-access contract: a repo
    that would ask for credentials fails loudly instead of hanging."""
    env = dict(os.environ, GIT_TERMINAL_PROMPT="0")
    out = subprocess.run(
        ["git", "ls-remote", "--heads", url],
        check=True, capture_output=True, text=True, env=env,
    ).stdout
    steps = sorted(
        int(m.group(1))
        for line in out.splitlines()
        if (m := TIMESTEP_REF.search(line))
    )
    if not steps:
        raise SystemExit(f"{url}: no timestep-* branches found — wrong repo?")
    return steps


def main() -> None:
    here = Path(__file__).resolve().parent
    cases = {}
    for case_id, url in CASES.items():
        steps = discover_timesteps(url)
        print(f"[bank] {case_id}: timesteps {steps}  ({url})")
        cases[case_id] = CaseSpec(timesteps=steps, prompts=PROMPTS)

    table = build_taskbank(cases, EVAL_CASES, sandbox_image=SANDBOX_IMAGE)
    for project in PROJECTS:
        path = here / project / "taskbank.parquet"
        write_taskbank(table, path)
        read_back = pq.read_table(path)
        splits = read_back.column("split").to_pylist()
        print(f"[bank] {path.relative_to(here)}: {read_back.num_rows} rows "
              f"({splits.count('train')} train / {splits.count('eval')} eval)")


if __name__ == "__main__":
    main()
