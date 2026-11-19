"""Generate fine-tuning trajectories for nanocoder worker agents.

Each sample: a fresh synthetic repo + task, solved by a scripted teacher that calls nanocoder's
real tools (so tool outputs are exactly what the model will see at inference). Samples that fail
their own check are dropped.

    python finetune/gen_data.py --n 600 --out finetune/data/train.jsonl --seed 1
"""
import argparse
import json
import os
import random
import shutil
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)

import nanocoder as nc  # noqa: E402
from tasks import sample_task  # noqa: E402

nc.w = lambda s: None  # silence nanocoder's terminal output


class NullLLM:
    model = "teacher"
    base_url = ""


class Env:
    """Runs teacher tool calls through a real nanocoder sub-agent and records the transcript."""

    def __init__(self, root, tier, prompt):
        self.agent = nc.Agent(NullLLM(), root, yolo=True, role=tier)
        self.messages = [self.agent.messages[0], {"role": "user", "content": prompt}]
        self.n = 0

    def call(self, text, name, args):
        if name not in self.agent.tools:
            raise RuntimeError(f"teacher used {name} not available to this tier")
        cid = f"call_{self.n}"
        self.n += 1
        self.messages.append({"role": "assistant", "content": text, "tool_calls": [
            {"id": cid, "type": "function", "function": {"name": name, "arguments": args}}]})
        try:
            out = self.agent.tools[name](**args)
            out = nc.truncate(out[0] if isinstance(out, tuple) else out)
        except nc.ToolError as e:
            out = f"Error: {e}"
        self.messages.append({"role": "tool", "tool_call_id": cid, "content": out})
        return out

    def final(self, text):
        self.messages.append({"role": "assistant", "content": text})
        self.report = text


def make_sample(seed, tier=None, kind=None):
    rng = random.Random(seed)
    tier = tier or ("small" if rng.random() < 0.35 else "medium")
    root = tempfile.mkdtemp(prefix="nc_")
    cwd = os.getcwd()
    try:
        task = sample_task(rng, root, tier, kind)
        env = Env(root, tier, task.prompt)
        task.teacher(env)
        ok, why = task.check(root, env.report)
        if not ok:
            raise AssertionError(f"teacher failed its own check: {why}")
        # make the transcript path-independent: the repo root differs per sample
        blob = json.dumps({"tier": tier, "kind": task.kind, "messages": env.messages,
                           "tools": env.agent.schemas()})
        return blob.replace(root, "/home/user/project")
    finally:
        os.chdir(cwd)
        shutil.rmtree(root, ignore_errors=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=600)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", default=os.path.join(HERE, "data", "train.jsonl"))
    args = ap.parse_args()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    kinds, bad = {}, 0
    with open(args.out, "w") as fh:
        i = 0
        while sum(kinds.values()) < args.n:
            seed = args.seed * 1_000_000 + i
            i += 1
            try:
                line = make_sample(seed)
            except AssertionError as e:
                bad += 1
                print(f"skip seed {seed}: {str(e)[:120]}", file=sys.stderr)
                continue
            fh.write(line + "\n")
            k = json.loads(line)
            kinds[(k["tier"], k["kind"])] = kinds.get((k["tier"], k["kind"]), 0) + 1
    for k in sorted(kinds):
        print(f"{k[0]:<7} {k[1]:<14} {kinds[k]}")
    print(f"total {sum(kinds.values())}, skipped {bad}")


if __name__ == "__main__":
    main()
