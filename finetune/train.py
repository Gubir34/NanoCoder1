"""LoRA fine-tuning of a small model on agentcoder worker trajectories. Runs fine on CPU (bf16).

    python finetune/train.py --base Qwen/Qwen2.5-Coder-0.5B-Instruct --data finetune/data/train.jsonl \
        --out finetune/adapters/agentcoder-worker-0.5b

Only assistant turns are trained. The prompt prefix (system + tools + task) runs without
gradients and is reused as a KV cache, which roughly halves the cost per sample.

Small Qwen models ship <tool_call>/</tool_call> with identical, untrained embedding rows (tied with
the output layer), so LoRA alone can never make the model emit them. Those rows are trained too.
"""
import argparse
import functools
import json
import math
import os
import random
import time

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache


def encode(tok, sample):
    """Return (input_ids, labels mask, prefix_len) for one chat transcript."""
    msgs, tools = sample["messages"], sample["tools"]
    full = tok.apply_chat_template(msgs, tools=tools, tokenize=False)
    spans = []
    for i, m in enumerate(msgs):
        if m["role"] != "assistant":
            continue
        before = tok.apply_chat_template(msgs[:i], tools=tools, tokenize=False, add_generation_prompt=True)
        after = tok.apply_chat_template(msgs[:i + 1], tools=tools, tokenize=False)
        assert full.startswith(before) and full.startswith(after), "chat template is not prefix-stable"
        spans.append((len(before), len(after)))
    enc = tok(full, return_offsets_mapping=True, add_special_tokens=False)
    ids, offs = enc["input_ids"], enc["offset_mapping"]
    mask = [any(a <= s < b for a, b in spans) for s, _ in offs]
    # the last prefix token must run with gradients: it predicts the first assistant token
    prefix = mask.index(True) - 1
    return ids, mask, prefix


class PrefixCache(DynamicCache):
    """A cache that is only read: update() returns prefix + new K/V but stores nothing.

    With activation checkpointing a layer runs twice (forward, then again in backward); a normal
    cache would get the rest's K/V appended twice. HF's own checkpointing avoids that by dropping
    the cache altogether, which silently trains on the rest without its prefix.
    """

    def update(self, key_states, value_states, layer_idx, *args, **kwargs):
        layer = self.layers[layer_idx]
        return torch.cat([layer.keys, key_states], -2), torch.cat([layer.values, value_states], -2)


def checkpoint_layers(core):
    """Recompute each decoder layer's activations in backward instead of keeping them."""
    from torch.utils.checkpoint import checkpoint
    for layer in core.layers:
        layer.forward = functools.partial(checkpoint, layer.forward, use_reentrant=False)


@torch.no_grad()
def decision_check(model, tok, data):
    """Average loss hides a wrong *first* token (e.g. text instead of <tool_call>). Count exact hits."""
    hits = 0
    for ids, mask, prefix in data:
        first = mask.index(True)
        with torch.autocast("cpu", dtype=torch.bfloat16):
            logits = model(input_ids=torch.tensor([ids[:first]])).logits[0, -1]
        hits += int(logits.argmax()) == ids[first]
    return hits


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="Qwen/Qwen2.5-Coder-0.5B-Instruct")
    ap.add_argument("--data", default="finetune/data/train.jsonl",
                    help="JSONL file(s), comma-separated; they are shuffled together")
    ap.add_argument("--out", default="finetune/adapters/agentcoder-worker-0.5b")
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--token-lr", type=float, default=5e-3, help="lr for the trained special-token rows")
    ap.add_argument("--accum", type=int, default=4)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--max-len", type=int, default=6144)
    ap.add_argument("--limit", type=int, default=0, help="use only the first N samples")
    ap.add_argument("--threads", type=int, default=os.cpu_count())
    ap.add_argument("--stats", action="store_true", help="print token stats and exit")
    ap.add_argument("--grad-ckpt", action="store_true",
                    help="recompute activations in backward: slower, but 1.5B+ fits in ~15GB RAM")
    ap.add_argument("--max-minutes", type=float, default=0,
                    help="stop after this long, saving a checkpoint in --out; rerun with --resume to continue")
    ap.add_argument("--resume", action="store_true", help="continue from the checkpoint in --out")
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    torch.manual_seed(0)

    tok = AutoTokenizer.from_pretrained(args.base)
    samples = [json.loads(line) for path in args.data.split(",") for line in open(path)]
    random.Random(0).shuffle(samples)
    if args.limit:
        samples = samples[: args.limit]
    data = []
    for s in samples:
        ids, mask, prefix = encode(tok, s)
        if len(ids) <= args.max_len:
            data.append((ids, mask, prefix))
    lens = sorted(len(d[0]) for d in data)
    tgt = sum(sum(d[1]) for d in data)
    print(f"{len(data)} samples · tokens median {lens[len(lens) // 2]} max {lens[-1]} · "
          f"trained tokens {tgt} ({tgt / sum(lens):.0%}) · prefix median {sorted(d[2] for d in data)[len(data) // 2]}")
    if args.stats:
        return

    model = AutoModelForCausalLM.from_pretrained(args.base, torch_dtype=torch.bfloat16)
    special = [i for i in tok.convert_tokens_to_ids(["<tool_call>", "</tool_call>"]) if isinstance(i, int)]
    cfg = LoraConfig(r=args.rank, lora_alpha=args.rank * 2, lora_dropout=0.05, task_type="CAUSAL_LM",
                     target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
                     trainable_token_indices=special)
    model = get_peft_model(model, cfg)
    if args.grad_ckpt:
        checkpoint_layers(model.base_model.model.model)
    for p in model.parameters():
        if p.requires_grad:
            p.data = p.data.float()
    model.print_trainable_parameters()
    core, head = model.base_model.model.model, model.base_model.model.lm_head

    params = [p for p in model.parameters() if p.requires_grad]
    token_params = [p for n, p in model.named_parameters() if p.requires_grad and "trainable_tokens" in n]
    lora_params = [p for p in params if all(p is not q for q in token_params)]
    # embedding rows start out identical to their neighbours; they need a much larger step to separate
    opt = torch.optim.AdamW([{"params": lora_params, "lr": args.lr},
                             {"params": token_params, "lr": args.token_lr}], weight_decay=0.0)
    print(f"token rows: {sum(p.numel() for p in token_params)} params at lr {args.token_lr}")
    total = math.ceil(len(data) * args.epochs / args.accum)
    warm = max(1, total // 20)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / total))))

    order = []
    rng = random.Random(0)
    while len(order) < len(data) * args.epochs:
        perm = list(range(len(data)))
        rng.shuffle(perm)
        order += perm
    order = order[: int(len(data) * args.epochs)]

    ckpt_path = os.path.join(args.out, "checkpoint.pt")
    named = {n: p for n, p in model.named_parameters() if p.requires_grad}
    start, step = 0, 0
    if args.resume and os.path.exists(ckpt_path):
        ck = torch.load(ckpt_path, weights_only=False)
        with torch.no_grad():
            for n, v in ck["params"].items():
                named[n].copy_(v)
        opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"])
        torch.set_rng_state(ck["rng"])
        start, step = ck["next"], ck["step"]
        print(f"resumed at sample {start}/{len(order)} (step {step}/{total})", flush=True)

    model.train()
    t0, losses = time.time(), []
    for i in range(start, len(order)):
        ids, mask, prefix = data[order[i]]
        x = torch.tensor([ids])
        with torch.autocast("cpu", dtype=torch.bfloat16):
            with torch.no_grad():
                pkv = core(input_ids=x[:, :prefix], use_cache=True).past_key_values
            pkv.__class__ = PrefixCache  # rest pass reads the prefix K/V without appending to it
            rest = x[0, prefix:]
            h = core(input_ids=rest[None], past_key_values=pkv, use_cache=True).last_hidden_state[0]
            # position j predicts rest[j+1]; keep positions whose next token is an assistant token
            sel = torch.tensor(mask[prefix + 1:])
            logits = head(h[:-1][sel]).float()
        loss = torch.nn.functional.cross_entropy(logits, rest[1:][sel])
        (loss / args.accum).backward()
        losses.append(loss.item())
        if (i + 1) % args.accum == 0 or i == len(order) - 1:
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            step += 1
            el, done = time.time() - t0, i + 1 - start
            if step % 5 == 0 or step == total:
                recent = losses[-args.accum * 5:]
                print(f"step {step}/{total} · loss {sum(recent) / len(recent):.4f} · "
                      f"lr {sched.get_last_lr()[0]:.2e} · {el / done:.1f}s/sample · "
                      f"eta {el / done * (len(order) - i - 1) / 60:.0f}m", flush=True)
            if args.max_minutes and el > args.max_minutes * 60 and i < len(order) - 1:
                os.makedirs(args.out, exist_ok=True)
                torch.save({"params": {n: p.detach().clone() for n, p in named.items()}, "opt": opt.state_dict(),
                            "sched": sched.state_dict(), "rng": torch.get_rng_state(), "next": i + 1,
                            "step": step}, ckpt_path)
                print(f"paused at step {step}/{total}, checkpoint in {ckpt_path}; rerun with --resume", flush=True)
                return
    model.eval()
    hits = decision_check(model, tok, data[:40])
    print(f"first-token check: {hits}/{min(40, len(data))} samples start their first reply with the right token")
    os.makedirs(args.out, exist_ok=True)
    model.save_pretrained(args.out)
    tok.save_pretrained(args.out)
    if os.path.exists(ckpt_path):
        os.remove(ckpt_path)
    print(f"saved adapter to {args.out} in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
