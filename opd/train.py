#!/usr/bin/env python3
"""OPD, zero-CLI: ONE `python train.py` collects episodes against the
staging estate (in-process), teacher-scores them against the ALWAYS-ON
teacher endpoint (in-process — `score.score(config)`, no serve swap, no
CLI), and trains. Written from the library README §4 by an external
consumer.

Same substrate as SFT, different loss. The config's ready dict serves
only scored, fresh, never-served student tapes; `OPDCollator` scatters
the teacher column to `[B, L]` under its dotted name in `.aux`. The loss
is the detached score-function RKL surrogate (README §4's three-regime
table):

    s_lp = student logp of the served token (the worked shift-gather)
    div  = s_lp - teacher            # RKL integrand on the sampled support
    loss = (div.detach() * s_lp * mask).sum() / mask.sum()

`mean(div)` over masked positions is the actual RKL estimate, printed
per step — the surrogate loss itself is not a divergence.

Memory craft (the library's own documented hazard): the ready serves
`truncated` tapes, one truncated-at-context tape pads the batch to
L ≈ 32k, and a full-batch vocab-wide float32 log-softmax at that length
does not fit next to co-resident serving — so the sum is computed one
sequence at a time (gradient accumulation) with `logits_to_keep`
bounding the logits to the response window, the recipe the library
names next to its no-truncation note.
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
    OPDCollator,
    TrajectoryStore,
    check_tokenizer,
    collect_episodes,
    load_config,
    make_loader,
)

import score as scorer


def collect_event(event: dict) -> None:
    kind = event.get("type")
    rest = {k: v for k, v in event.items() if k != "type"}
    print(f"[collect] {kind}: {rest}", flush=True)


def collect_and_verify(config, tag: str) -> list[str]:
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

    store = TrajectoryStore.open(config.store.root)
    clean = 0
    for uid in report.uids:
        record = store.load([uid])[0]
        failures = list(record.env.outcome.gate_failures)
        if failures:
            # a retried attempt can land a quarantined record: unservable
            # under any ready (hygiene), kept for forensics — never fatal
            print(f"[{tag}] {uid}: QUARANTINED gate_failures={failures}")
            continue
        prov = record.env.provenance
        argv = list(prov["invocation"]["argv"])
        mounts = argv.count("-v")
        assert mounts == 2, f"{uid}: {mounts} mounts (want 2)"
        print(f"[{tag}] {uid}: gates=[] (G2/G3/G5 green) mounts=2 "
              f"G2={prov['system_prompt_hash'][:16]}... "
              f"G3={prov['tool_roster_hash'][:16]}...")
        clean += 1
    print(f"[{tag}] spot-check: {clean} clean / {len(report.uids)} landed")
    store.close()
    return list(report.uids)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path,
                        default=Path(__file__).resolve().parent / "config.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    user = config.user
    steps = int(user.get("steps", 4))
    lr = float(user.get("lr", 1e-4))
    out = args.config.parent / user.get("out", "adapters/run1")

    # ---- collect (in-process), then score against the always-on teacher ----
    collect_and_verify(config, "opd")
    summary = scorer.score(config)
    if summary["scored"] == 0:
        sys.exit("[opd] nothing scored — the ready dict would starve; aborting")

    # ---- train ----
    loader = make_loader(config)               # ready + mix arrive as data
    timeout_s = config.loader.timeout_s if config.loader.timeout_s is not None else 20.0

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(0)
    model_id = str(config.driver["model_id"])
    revision = str(config.driver["revision"])
    model = AutoModelForCausalLM.from_pretrained(model_id, revision=revision,
                                                 dtype=torch.bfloat16)
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

    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id
    tokenizer_json = Path(hf_hub_download(model_id, "tokenizer.json",
                                          revision=revision))

    print(f"[opd] device={device.type} model={model_id} steps={steps} "
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

        teacher_full = batch.aux["opd.teacher_logp_sampled"]           # [B, L]
        mask_full = batch.aux["loss_mask"]                             # [B, L]
        prompt_lens = batch.aux["prompt_lens"]
        lens = prompt_lens + batch.aux["response_lens"]
        n_total = float(mask_full.sum()) or 1.0

        # One sequence at a time + logits_to_keep over the response window
        # (gradient accumulation) — the library's worked recipe for the
        # L_max x V hazard; a truncated tape makes L ~ 32k.
        loss_value = 0.0
        div_sum = 0.0
        for i in range(len(batch.uids)):
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
            s_lp = logp.gather(-1, targets[:, None])[:, 0]             # [R]
            teacher_i = teacher_full[i, P:li].to(device)               # [R]
            mask_i = mask_full[i, P:li].to(device).float()             # [R]
            div = s_lp - teacher_i
            item_loss = (div.detach() * s_lp * mask_i).sum() / n_total
            item_loss.backward()               # accumulate across items
            loss_value += float(item_loss.detach())
            div_sum += float((div.detach() * mask_i).sum())
        optimizer.step()
        optimizer.zero_grad()

        loader.commit(batch)                   # consumed AFTER the optimizer step
        committed += len(batch.uids)
        losses.append(loss_value)
        mean_div = div_sum / n_total           # the RKL estimate
        divergences.append(mean_div)
        print(f"[opd] step {completed_steps:2d} loss {loss_value:+.4f} "
              f"mean_div {mean_div:+.4f} positions {int(n_total)} | "
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
    retired = 0
    for uid in sorted(uids_seen):
        view = store.view(uid)
        retired += int(bool(view["consumed"]))
        print(f"  {uid}  serve_count={view['serve_count']} "
              f"retired={bool(view['consumed'])}")
    expected = completed_steps * config.loader.batch_size
    print(f"[opd] consistency: committed serves = {committed}, steps x batch = "
          f"{completed_steps} x {config.loader.batch_size} = {expected} -> "
          f"{'OK' if committed == expected else 'MISMATCH'}; retired {retired}")
    store.close()


if __name__ == "__main__":
    main()
