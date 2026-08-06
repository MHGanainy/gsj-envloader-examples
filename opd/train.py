#!/usr/bin/env python3
"""OPD over served gsj-envloader trajectories — written from the library
README §4 by an external consumer.

Same substrate as SFT, different loss. Upstream, `score.py` attached the
teacher's per-token logp (`opd.teacher_logp_sampled`); the config's ready
dict serves only scored, fresh, never-served student tapes. `OPDCollator`
scatters the teacher column to `[B, L]` under its dotted name in `.aux`.

The loss is the detached score-function RKL surrogate (README §4's
three-regime table; the worked shift-gather snippet in §4 Level 2 gives
the student logp of the served token):

    s_lp[b, j] = log softmax(logits[b, j-1])[input_ids[b, j]]
    div  = s_lp - teacher            # RKL integrand on the sampled support
    loss = (div.detach() * s_lp * mask).sum() / mask.sum()

`mean(div)` over masked positions is the actual RKL estimate and is
printed per step — the surrogate loss itself is not a divergence.

    .venv/bin/python train.py --config config.yaml
"""

from __future__ import annotations

import argparse
from itertools import islice
from pathlib import Path

import torch
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

from gsj.envloader import (
    OPDCollator,
    TrajectoryStore,
    check_tokenizer,
    load_config,
    make_loader,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    args = parser.parse_args()

    config = load_config(args.config)
    user = config.user
    steps = int(user.get("steps", 4))
    lr = float(user.get("lr", 1e-4))
    out = args.config.parent / user.get("out", "adapters/run1")

    loader = make_loader(config)               # ready + mix arrive as data
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

    # pinned to the codec's snapshot revision (driver.snapshot_path basename)
    # so the §6.2 assert cannot break on an upstream HF main push
    revision = (Path(str(config.driver["snapshot_path"])).name
                if "snapshot_path" in config.driver else None)
    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=revision)
    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id
    tokenizer_json = Path(hf_hub_download(args.model, "tokenizer.json",
                                          revision=revision))

    print(f"[opd] device={device.type} model={args.model} steps={steps} "
          f"batch_size={config.loader.batch_size}")

    losses: list[float] = []
    divergences: list[float] = []
    uids_seen: set[str] = set()
    committed = 0
    completed_steps = 0

    batches = loader.torch_batches(OPDCollator(pad_token_id=pad_id),
                                   timeout_s=timeout_s)
    for batch in islice(batches, steps):
        trainer_hash = check_tokenizer(batch, tokenizer_json)   # §6.2, per batch
        if not uids_seen:
            print(f"[opd] tokenizer-hash assert OK: {trainer_hash}")
        uids_seen.update(batch.uids)

        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        mask = batch.aux["loss_mask"].to(device).float()               # [B, L]
        teacher = batch.aux["opd.teacher_logp_sampled"].to(device)     # [B, L]

        out_logits = model(input_ids=input_ids,
                           attention_mask=attention_mask).logits
        # student logp of the served token (README §4 Level 2 snippet):
        # logits at j-1 are the distribution over the token at j; the left
        # pad restores [B, L] indexing (position 0 is never masked).
        logp = out_logits[:, :-1].float().log_softmax(-1)
        s_lp = F.pad(logp.gather(-1, input_ids[:, 1:, None])[..., 0], (1, 0))
        div = s_lp - teacher
        n = mask.sum().clamp(min=1.0)
        loss = (div.detach() * s_lp * mask).sum() / n
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        loader.commit(batch)                   # consumed AFTER the optimizer step
        committed += len(batch.uids)
        losses.append(float(loss.detach()))
        mean_div = float((div.detach() * mask).sum() / n)  # the RKL estimate
        divergences.append(mean_div)
        print(f"[opd] step {completed_steps:2d} loss {losses[-1]:+.4f} "
              f"mean_div {mean_div:+.4f} positions {int(n)} | "
              f"lag_histogram {loader.lag_histogram()}")
        completed_steps += 1
    if completed_steps < steps:
        print(f"[opd] run dry after {completed_steps} step(s): no full batch "
              f"servable within {timeout_s:.0f}s under this ready "
              f"(serve_count: 0 — every record serves at most once)")

    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(out))
    print(f"[opd] adapter saved to {out}")
    if divergences:
        print(f"[opd] RKL estimate first={divergences[0]:+.4f} "
              f"last={divergences[-1]:+.4f}; loss first={losses[0]:+.4f} "
              f"last={losses[-1]:+.4f} ({completed_steps} optimizer steps)")
    loader.close()

    store = TrajectoryStore.open(config.store.root)
    print("\n=== serve/commit accounting ===")
    for uid in sorted(uids_seen):
        view = store.view(uid)
        print(f"  {uid}  serve_count={view['serve_count']} "
              f"retired={bool(view['consumed'])}")
    expected = completed_steps * config.loader.batch_size
    print(f"[opd] consistency: committed serves = {committed}, steps x batch = "
          f"{completed_steps} x {config.loader.batch_size} = {expected} -> "
          f"{'OK' if committed == expected else 'MISMATCH'}")
    store.close()


if __name__ == "__main__":
    main()
