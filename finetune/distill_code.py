"""Code answers with reasoning, from a teacher model, kept only if they pass the tests.

MBPP's reference solutions are bare code. Training on them taught the worker to skip the short
explanation it used to write before coding, and that cost accuracy. Here a teacher model (any
OpenAI-compatible server) answers each training problem in its own words + code; an answer enters
the data set only if its code passes the problem's tests (rejection sampling).

    python finetune/distill_code.py --mbpp /path/to/mbpp --url http://127.0.0.1:8090/v1 --model teacher \
        --out finetune/data/code_distill.jsonl
"""
import argparse
import json
import os
import random
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from code_data import CHAT_SYSTEM, TRAIN_SPLITS, chat_prompt, load, passes, usable  # noqa: E402
from code_eval import extract  # noqa: E402


def ask(url, model, prompt, temperature, max_tokens=700):
    import urllib.request
    body = {"model": model, "temperature": temperature, "max_tokens": max_tokens, "stream": False,
            "messages": [{"role": "system", "content": CHAT_SYSTEM}, {"role": "user", "content": prompt}]}
    req = urllib.request.Request(url.rstrip("/") + "/chat/completions", json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        return json.load(r)["choices"][0]["message"]["content"] or ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mbpp", required=True)
    ap.add_argument("--url", default="http://127.0.0.1:8090/v1")
    ap.add_argument("--model", required=True)
    ap.add_argument("--tries", type=int, default=2, help="greedy first, then sampled retries")
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--out", default=os.path.join(HERE, "data", "code_distill.jsonl"))
    ap.add_argument("--resume", action="store_true", help="append, continuing after the last kept problem")
    args = ap.parse_args()
    rng = random.Random(args.seed)
    probs = [p for p in map(usable, load(args.mbpp, TRAIN_SPLITS)) if p]
    prompts = [chat_prompt(p, rng) for p in probs]  # drawn up front so --resume sees the same prompts
    done, kept = 0, 0
    if args.resume and os.path.exists(args.out):
        seen = {json.loads(line)["messages"][1]["content"] for line in open(args.out)}
        kept = len(seen)
        done = max((i for i, q in enumerate(prompts, 1) if q in seen), default=0)
    with open(args.out, "a" if args.resume else "w") as fh:
        for i, (p, prompt) in enumerate(zip(probs, prompts), 1):
            if i <= done:
                continue
            for t in range(args.tries):
                reply = ask(args.url, args.model, prompt, 0.0 if t == 0 else 0.7)
                if "```" in reply and passes(extract(reply), p["tests"]):
                    fh.write(json.dumps({"tier": "chat", "kind": "code_chat", "tools": None, "messages": [
                        {"role": "system", "content": CHAT_SYSTEM}, {"role": "user", "content": prompt},
                        {"role": "assistant", "content": reply.strip()}]}) + "\n")
                    fh.flush()
                    kept += 1
                    break
            print(f"[{i}/{len(probs)}] kept {kept}", flush=True)
    print(f"kept {kept}/{len(probs)}")


if __name__ == "__main__":
    main()
