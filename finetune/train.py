"""LoRA fine-tuning of a small model on agentcoder worker trajectories. Runs fine on CPU (bf16).

    python finetune/train.py --base Qwen/Qwen2.5-Coder-0.5B-Instruct --data finetune/data/train.jsonl \
        --out finetune/adapters/agentcoder-worker-0.5b

Only assistant turns are trained. The prompt prefix (system + tools + task) runs without
gradients and is reused as a KV cache, which roughly halves the cost per sample.
"""
import argparse
import json
import math
import os
import random
import time

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="Qwen/Qwen2.5-Coder-0.5B-Instruct")
    ap.add_argument("--data", default="finetune/data/train.jsonl")
    ap.add_argument("--out", default="finetune/adapters/agentcoder-worker-0.5b")
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--accum", type=int, default=4)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--max-len", type=int, default=6144)
    ap.add_argument("--limit", type=int, default=0, help="use only the first N samples")
    ap.add_argument("--threads", type=int, default=os.cpu_count())
    ap.add_argument("--stats", action="store_true", help="print token stats and exit")
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    torch.manual_seed(0)

    tok = AutoTokenizer.from_pretrained(args.base)
    samples = [json.loads(line) for line in open(args.data)]
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
    cfg = LoraConfig(r=args.rank, lora_alpha=args.rank * 2, lora_dropout=0.05, task_type="CAUSAL_LM",
                     target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
    model = get_peft_model(model, cfg)
    for p in model.parameters():
        if p.requires_grad:
            p.data = p.data.float()
    model.print_trainable_parameters()
    core, head = model.base_model.model.model, model.base_model.model.lm_head

    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0)
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

    model.train()
    t0, losses, step = time.time(), [], 0
    for i, idx in enumerate(order):
        ids, mask, prefix = data[idx]
        x = torch.tensor([ids])
        with torch.autocast("cpu", dtype=torch.bfloat16):
            with torch.no_grad():
                pkv = core(input_ids=x[:, :prefix], use_cache=True).past_key_values
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
            if step % 5 == 0 or step == total:
                el = time.time() - t0
                recent = losses[-args.accum * 5:]
                print(f"step {step}/{total} · loss {sum(recent) / len(recent):.4f} · "
                      f"lr {sched.get_last_lr()[0]:.2e} · {el / (i + 1):.1f}s/sample · "
                      f"eta {el / (i + 1) * (len(order) - i - 1) / 60:.0f}m", flush=True)
    os.makedirs(args.out, exist_ok=True)
    model.save_pretrained(args.out)
    tok.save_pretrained(args.out)
    print(f"saved adapter to {args.out} in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
