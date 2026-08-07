#!/usr/bin/env python3
"""SFT, zero-CLI: ONE `python train.py` collects episodes against the
staging estate (in-process — `collect_episodes`, no CLI, no scripts) and
trains on them. Written from the library README §4 and
docs/config-reference.md by an external consumer.

    config = load_config("config.yaml")     # every environment input is an
                                            #   endpoint or a sha-pinned URL
    report = collect_episodes(config)       # sandboxed episodes, gates, store
    loader = make_loader(config)            # the SAME file serves the store
    ... torch_batches -> loss -> commit -> adapter save -> accounting

`user:` carries our lr/steps/out; the library never reads it.
"""

from __future__ import annotations

import argparse
import os
import sys
from itertools import islice
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

from gsj.envloader import (
    SFTCollator,
    TrajectoryStore,
    check_tokenizer,
    collect_episodes,
    load_config,
    make_loader,
)


def collect_event(event: dict) -> None:
    """Render the structured progress events as one line each."""
    kind = event.get("type")
    rest = {k: v for k, v in event.items() if k != "type"}
    print(f"[collect] {kind}: {rest}", flush=True)


def collect_and_verify(config, tag: str) -> list[str]:
    """In-process collection + the gate spot-check over what landed."""
    secret_name = config.task.mcp_launch.token_secret
    if secret_name and not os.environ.get(secret_name):
        sys.exit(f"[{tag}] {secret_name} is not set — the MCP service tokens "
                 f"are minted from it (export it, then rerun)")

    report = collect_episodes(config, progress=collect_event)
    print(f"[{tag}] CollectReport: exit={report.exit_code} "
          f"reason={report.reason!r} attempted={report.attempted} "
          f"new_trainable={report.new_trainable} counts={report.counts} "
          f"gate_failures={report.gate_failures} wall={report.wall_s:.1f}s")
    if report.new_trainable == 0:
        sys.exit(f"[{tag}] no trainable episodes landed — aborting before "
                 f"training (reason: {report.reason})")

    # Spot-check the contracts on every landed record: gates green, the
    # G2/G3 hashes visible (compare against the published staging pins),
    # and exactly two mounts (checkout rw + agent dir ro — remote MCP
    # means no /pages, no shim mount).
    store = TrajectoryStore.open(config.store.root)
    for uid in report.uids:
        record = store.load([uid])[0]
        failures = list(record.env.outcome.gate_failures)
        assert failures == [], f"{uid}: gate_failures={failures}"
        prov = record.env.provenance
        argv = list(prov["invocation"]["argv"])
        mounts = argv.count("-v")
        assert mounts == 2, f"{uid}: {mounts} mounts (want 2)"
        print(f"[{tag}] {uid}: gates=[] (G2/G3/G5 green) mounts=2 "
              f"G2={prov['system_prompt_hash'][:16]}... "
              f"G3={prov['tool_roster_hash'][:16]}...")
    store.close()
    return list(report.uids)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path,
                        default=Path(__file__).resolve().parent / "config.yaml")
    args = parser.parse_args()

    config = load_config(args.config)          # fail-fast at the file
    user = config.user
    steps = int(user.get("steps", 4))
    lr = float(user.get("lr", 1e-4))
    out = args.config.parent / user.get("out", "adapters/run1")

    # ---- collect (in-process; targets from collector.seeding) ----
    collect_and_verify(config, "sft")

    # ---- train ----
    loader = make_loader(config)               # ready arrives as data
    timeout_s = config.loader.timeout_s if config.loader.timeout_s is not None else 20.0

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(0)
    # The trainer model IS the codec identity: the driver's host-portable
    # {model_id, revision} pin pair, resolved through the HF cache.
    model_id = str(config.driver["model_id"])
    revision = str(config.driver["revision"])
    model = AutoModelForCausalLM.from_pretrained(model_id, revision=revision,
                                                 dtype=torch.bfloat16)
    model = get_peft_model(model, LoraConfig(
        r=8, lora_alpha=16, lora_dropout=0.0, bias="none", task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
    ))
    model.to(device).train()
    optimizer = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad), lr=lr)

    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id
    # our tokenizer.json is the §6.2 identity-assert surface
    tokenizer_json = Path(hf_hub_download(model_id, "tokenizer.json",
                                          revision=revision))

    print(f"[sft] device={device.type} model={model_id} steps={steps} "
          f"batch_size={config.loader.batch_size}")

    losses: list[float] = []
    uids_seen: set[str] = set()
    committed = 0
    completed_steps = 0

    batches = loader.torch_batches(SFTCollator(pad_token_id=pad_id),
                                   timeout_s=timeout_s)
    for batch in islice(batches, steps):
        trainer_hash = check_tokenizer(batch, tokenizer_json)
        if not uids_seen:
            print(f"[sft] tokenizer-hash assert OK: {trainer_hash}")
        uids_seen.update(batch.uids)

        moved = {k: v.to(device) for k, v in batch.items()}
        loss = model(**moved).loss             # HF's shifted CE from the labels
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        loader.commit(batch)                   # consumed AFTER the optimizer step
        committed += len(batch.uids)
        losses.append(float(loss.detach()))
        n_positions = int((batch["labels"] != -100).sum())
        print(f"[sft] step {completed_steps:2d} loss {losses[-1]:.4f} "
              f"positions {n_positions} | lag_histogram {loader.lag_histogram()}")
        completed_steps += 1
    if completed_steps < steps:
        print(f"[sft] run dry after {completed_steps} step(s): no full batch "
              f"servable within {timeout_s:.0f}s under this ready")

    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(out))
    print(f"[sft] adapter saved to {out}")
    if losses:
        print(f"[sft] loss first={losses[0]:.4f} last={losses[-1]:.4f} "
              f"({completed_steps} optimizer steps)")
    loader.close()

    # accounting through a fresh store handle — the store is the only shared
    # surface, so this is exactly what any other process would see
    store = TrajectoryStore.open(config.store.root)
    print("\n=== serve/commit accounting ===")
    retired = 0
    for uid in sorted(uids_seen):
        view = store.view(uid)
        retired += int(bool(view["consumed"]))
        print(f"  {uid}  serve_count={view['serve_count']} "
              f"retired={bool(view['consumed'])}")
    expected = completed_steps * config.loader.batch_size
    print(f"[sft] consistency: committed serves = {committed}, steps x batch = "
          f"{completed_steps} x {config.loader.batch_size} = {expected} -> "
          f"{'OK' if committed == expected else 'MISMATCH'}; retired {retired}")
    store.close()


if __name__ == "__main__":
    main()
