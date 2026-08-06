#!/usr/bin/env python3
"""RLVR over served gsj-envloader trajectories — written from the library
README §4 by an external consumer.

Third regime, same substrate: `grade.py` attached the verifiable scalar
(`rlvr.reward`); the config's ready dict serves only graded, fresh,
never-served tapes; `RLVRCollator` stacks the reward into
`aux["rewards"] [B]`. The loss is REINFORCE-style advantage-weighted CE
(README §4's three-regime table):

    adv  = rewards - rewards.mean()          # batch-mean baseline, [B]
    ce   = -log softmax(logits)[served token]
    loss = (adv[:, None] * ce * loss_mask).sum() / loss_mask.sum()

Memory craft, learned the hard way: a 900 s wall can produce a
truncated-at-context tape (R ~ 30k), and the collated batch pads every
row to that length — a full-batch vocab-wide float32 log-softmax at
L=32k does NOT fit next to a co-resident serving engine. So the sum is
computed one sequence at a time (gradient accumulation) with
`logits_to_keep` bounding the logits to the response window, and
zero-advantage items skipped outright (their REINFORCE gradient is
exactly zero). The library README warns that length budgeting is trainer
craft; the concrete L x V hazard it leaves to us.

    .venv/bin/python train.py --config config.yaml
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
    RLVRCollator,
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

    loader = make_loader(config)               # ready arrives as data
    timeout_s = config.loader.timeout_s if config.loader.timeout_s is not None else 20.0

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(0)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16)
    model = get_peft_model(model, LoraConfig(
        r=8, lora_alpha=16, lora_dropout=0.0, bias="none", task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
    ))
    model.gradient_checkpointing_enable()      # long tapes: keep activations small
    model.enable_input_require_grads()
    model.config.use_cache = False
    model.to(device).train()
    optimizer = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad), lr=lr)

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id
    tokenizer_json = Path(hf_hub_download(args.model, "tokenizer.json"))

    print(f"[rlvr] device={device.type} model={args.model} steps={steps} "
          f"batch_size={config.loader.batch_size}")

    losses: list[float] = []
    uids_seen: set[str] = set()
    committed = 0
    completed_steps = 0

    batches = loader.torch_batches(RLVRCollator(pad_token_id=pad_id),
                                   timeout_s=timeout_s)
    for batch in islice(batches, steps):
        trainer_hash = check_tokenizer(batch, tokenizer_json)   # §6.2, per batch
        if not uids_seen:
            print(f"[rlvr] tokenizer-hash assert OK: {trainer_hash}")
        uids_seen.update(batch.uids)

        rewards = batch.aux["rewards"].to(device)          # [B] graded scalars
        adv = rewards - rewards.mean()                     # batch-mean baseline
        prompt_lens = batch.aux["prompt_lens"]
        lens = prompt_lens + batch.aux["response_lens"]
        n_total = float(batch.aux["loss_mask"].sum()) or 1.0

        loss_value = 0.0
        for i in range(len(batch.uids)):
            if float(adv[i]) == 0.0:
                continue                       # zero advantage => zero gradient
            li, P = int(lens[i]), int(prompt_lens[i])
            out_i = model(
                input_ids=batch["input_ids"][i : i + 1, :li].to(device),
                attention_mask=batch["attention_mask"][i : i + 1, :li].to(device),
                logits_to_keep=li - P + 1,     # response window only
            )
            # returned logits cover positions P-1..li-1; [:-1] pairs the
            # logits at j-1 with the served token at j
            logp = out_i.logits[0, :-1].float().log_softmax(-1)
            targets = batch["input_ids"][i, P:li].to(device)
            ce = -logp.gather(-1, targets[:, None])[:, 0]              # [R]
            mask = batch.aux["loss_mask"][i, P:li].to(device).float()  # [R]
            item_loss = adv[i] * (ce * mask).sum() / n_total
            item_loss.backward()               # accumulate across items
            loss_value += float(item_loss.detach())
        optimizer.step()
        optimizer.zero_grad()

        loader.commit(batch)                   # consumed AFTER the optimizer step
        committed += len(batch.uids)
        losses.append(loss_value)
        print(f"[rlvr] step {completed_steps:2d} loss {loss_value:+.4f} "
              f"mean_reward {float(rewards.mean()):.3f} "
              f"adv_spread {float(adv.max() - adv.min()):.3f} "
              f"positions {int(n_total)} | lag_histogram {loader.lag_histogram()}")
        completed_steps += 1
    if completed_steps < steps:
        print(f"[rlvr] run dry after {completed_steps} step(s): no full batch "
              f"servable within {timeout_s:.0f}s under this ready "
              f"(serve_count: 0 — every record serves at most once)")

    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(out))
    print(f"[rlvr] adapter saved to {out}")
    if losses:
        print(f"[rlvr] loss first={losses[0]:+.4f} last={losses[-1]:+.4f} "
              f"({completed_steps} optimizer steps)")
    loader.close()

    store = TrajectoryStore.open(config.store.root)
    print("\n=== serve/commit accounting ===")
    for uid in sorted(uids_seen):
        view = store.view(uid)
        print(f"  {uid}  serve_count={view['serve_count']} "
              f"retired={bool(view['consumed'])}")
    expected = completed_steps * config.loader.batch_size
    print(f"[rlvr] consistency: committed serves = {committed}, steps x batch = "
          f"{completed_steps} x {config.loader.batch_size} = {expected} -> "
          f"{'OK' if committed == expected else 'MISMATCH'}")
    store.close()


if __name__ == "__main__":
    main()
