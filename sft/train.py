#!/usr/bin/env python3
"""SFT over served gsj-envloader trajectories — written from the library
README §4 (Level 1) and docs/config-reference.md by an external consumer.

The library's claim is that this is the whole story:

    for batch in loader.torch_batches(SFTCollator(pad_token_id=pad_id),
                                      timeout_s=...):
        loss = model(**batch).loss
        loss.backward(); optimizer.step(); optimizer.zero_grad()
        loader.commit(batch)

This file is that loop plus our own trainer choices (LoRA, bf16), taking
everything else from the one config file: `train.py --config config.yaml`.
`user:` carries our lr/steps/out; the library never reads it.
"""

from __future__ import annotations

import argparse
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
    load_config,
    make_loader,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B",
                        help="trainer model — must tokenize identically to the "
                             "served provenance (check_tokenizer enforces it)")
    args = parser.parse_args()

    config = load_config(args.config)          # fail-fast at the file
    user = config.user
    steps = int(user.get("steps", 4))
    lr = float(user.get("lr", 1e-4))
    out = args.config.parent / user.get("out", "adapters/run1")

    loader = make_loader(config)               # ready (and mix) arrive as data
    timeout_s = config.loader.timeout_s if config.loader.timeout_s is not None else 20.0

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(0)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16)
    model = get_peft_model(model, LoraConfig(
        r=8, lora_alpha=16, lora_dropout=0.0, bias="none", task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
    ))
    model.to(device).train()
    optimizer = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad), lr=lr)

    # Pin the tokenizer fetch to the SAME revision the collection codec used
    # (driver.snapshot_path's basename is that snapshot's commit sha): an
    # unpinned fetch follows HF main and could fail the identity assert for
    # reasons external to this host.
    revision = (Path(str(config.driver["snapshot_path"])).name
                if "snapshot_path" in config.driver else None)
    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=revision)
    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id
    # our tokenizer.json is the §6.2 identity-assert surface
    tokenizer_json = Path(hf_hub_download(args.model, "tokenizer.json",
                                          revision=revision))

    print(f"[sft] device={device.type} model={args.model} steps={steps} "
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
