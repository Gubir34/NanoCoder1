"""Plain code-writing eval on MBPP's held-out test split (pass@1, greedy).

Sends the standard MBPP prompt (task + its tests) to an OpenAI-compatible server, extracts the code
block from the reply and runs the tests in an isolated interpreter (see code_data.passes).

    python finetune/code_eval.py --mbpp /path/to/mbpp --url http://127.0.0.1:8090/v1 --model tuned05 --n 100
"""
import argparse
import json
import os
import re
import sys
import time
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from code_data import CHAT_SYSTEM, chat_prompt, load, passes, usable  # noqa: E402

BLOCK_RE = re.compile(r"```(?:python|py)?\s*\n(.*?)```", re.S)


def ask(url, model, prompt, max_tokens=512):
    body = {"model": model, "temperature": 0, "max_tokens": max_tokens, "stream": False,
            "messages": [{"role": "system", "content": CHAT_SYSTEM}, {"role": "user", "content": prompt}]}
    req = urllib.request.Request(url.rstrip("/") + "/chat/completions", json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        return json.load(r)["choices"][0]["message"]["content"] or ""


def extract(reply):
    blocks = BLOCK_RE.findall(reply)
    return max(blocks, key=len) if blocks else reply


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mbpp", required=True)
    ap.add_argument("--url", default="http://127.0.0.1:8090/v1")
    ap.add_argument("--model", required=True)
    ap.add_argument("--n", type=int, default=100, help="first N usable test problems")
    ap.add_argument("--out")
    args = ap.parse_args()
    probs = [p for p in map(usable, load(args.mbpp, ["test"])) if p][: args.n]
    results, t0 = [], time.time()
    for i, p in enumerate(probs, 1):
        reply = ask(args.url, args.model, chat_prompt(p))
        ok = passes(extract(reply), p["tests"])
        results.append({"id": p["id"], "ok": ok, "reply": reply})
        print(f"[{i}/{len(probs)}] {p['id']:<4} {'PASS' if ok else 'FAIL'}", flush=True)
    n_ok = sum(r["ok"] for r in results)
    print(f"\nMBPP pass@1 {n_ok}/{len(results)} = {n_ok / max(1, len(results)):.1%} · {time.time() - t0:.0f}s")
    if args.out:
        with open(args.out, "w") as fh:
            json.dump(results, fh, indent=1)


if __name__ == "__main__":
    main()
