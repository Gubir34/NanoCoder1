"""Evaluate a model as a nanocoder sub-agent on held-out synthetic tasks.

Runs nanocoder's real agent loop against any OpenAI-compatible server and checks the outcome
(repo state + final report). Seeds differ from training data, so every repo/task is unseen.

    python finetune/eval.py --url http://localhost:8080/v1 --model local --n 30 --out results.json
"""
import argparse
import json
import os
import random
import shutil
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)

import nanocoder as nc  # noqa: E402
from tasks import sample_task  # noqa: E402

nc.w = lambda s: None


def run_one(url, model, seed, tier, max_steps):
    rng = random.Random(seed)
    root = tempfile.mkdtemp(prefix="nceval_")
    try:
        task = sample_task(rng, root, tier)
        agent = nc.Agent(nc.LLM(url, model), root, yolo=True, role=tier)
        agent.messages.append({"role": "user", "content": task.prompt})
        t0, err = time.time(), ""
        try:
            finished = agent.run_loop(max_steps)
        except nc.LLMError as e:
            finished, err = False, str(e)[:200]
        report = next((m.get("content") or "" for m in reversed(agent.messages) if m["role"] == "assistant"), "")
        ok, why = task.check(root, report) if finished else (False, err or "hit step limit")
        calls = [m for m in agent.messages if m["role"] == "tool" or (
            m["role"] == "user" and m["content"].startswith("<tool_result"))]
        return {"seed": seed, "tier": tier, "kind": task.kind, "ok": ok, "why": why,
                "steps": sum(m["role"] == "assistant" for m in agent.messages),
                "tool_errors": sum("Error" in m["content"][:40] for m in calls),
                "secs": round(time.time() - t0, 1), "report": report[:300]}
    finally:
        shutil.rmtree(root, ignore_errors=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8080/v1")
    ap.add_argument("--model", default="local")
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--seed", type=int, default=99)
    ap.add_argument("--max-steps", type=int, default=12)
    ap.add_argument("--out")
    args = ap.parse_args()
    results = []
    for i in range(args.n):
        tier = "small" if i % 5 in (0, 3) else "medium"  # 40% small, 60% medium
        r = run_one(args.url, args.model, args.seed * 1_000_000 + i, tier, args.max_steps)
        results.append(r)
        print(f"[{i + 1}/{args.n}] {r['tier']:<6} {r['kind']:<13} {'PASS' if r['ok'] else 'FAIL'} "
              f"steps={r['steps']} errs={r['tool_errors']} {r['secs']}s {'' if r['ok'] else r['why'][:80]}",
              flush=True)
    by = {}
    for r in results:
        by.setdefault(r["kind"], []).append(r["ok"])
    print("\nper task:", ", ".join(f"{k} {sum(v)}/{len(v)}" for k, v in sorted(by.items())))
    print(f"TOTAL {sum(r['ok'] for r in results)}/{len(results)} · "
          f"avg steps {sum(r['steps'] for r in results) / len(results):.1f} · "
          f"tool errors {sum(r['tool_errors'] for r in results)}")
    if args.out:
        with open(args.out, "w") as fh:
            json.dump(results, fh, indent=1)


if __name__ == "__main__":
    main()
