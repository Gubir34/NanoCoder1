"""Benchmark OpenAI-compatible inference servers the way nanocoder uses them.

For each server: a cold request with a ~1.5k-token agent prompt (cache-busted), then an agent-style
follow-up that appends a tool result to the same conversation (tests prefix caching).

    python scripts/bench_backends.py ollama=http://localhost:11434/v1@qwen2.5-coder:1.5b \
                                     llama=http://localhost:8080/v1@qwen
"""
import json
import os
import random
import sys
import time
import urllib.request

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import nanocoder as nc  # noqa: E402

TASK = "Write a Python module with a function parse_duration(s) that turns strings like '1h30m', '45s', " \
       "'2d4h' into seconds, with docstrings and input validation. Output only the code."
TOOL_RESULT = "\n".join(f"{i:>5}\tdef helper_{i}(x):\n      \treturn x * {i}" for i in range(1, 40))


def stream(url, model, messages, max_tokens=200):
    body = {"model": model, "messages": messages, "stream": True, "temperature": 0, "max_tokens": max_tokens,
            "stream_options": {"include_usage": True}}
    req = urllib.request.Request(url + "/chat/completions", json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    t0, ttft, n_chunks, usage = time.time(), None, 0, {}
    with urllib.request.urlopen(req, timeout=900) as r:
        for raw in r:
            line = raw.decode().strip()
            if not line.startswith("data:") or line == "data: [DONE]":
                continue
            d = json.loads(line[5:])
            usage = d.get("usage") or usage
            for ch in d.get("choices") or []:
                if (ch.get("delta") or {}).get("content"):
                    ttft = ttft or time.time() - t0
                    n_chunks += 1
    total = time.time() - t0
    out_tok = usage.get("completion_tokens") or n_chunks
    return {"ttft": ttft or total, "total": total, "prompt_tokens": usage.get("prompt_tokens"),
            "out_tokens": out_tok, "tok_s": out_tok / max(1e-6, total - (ttft or 0))}


def agent_messages():
    agent = nc.Agent(nc.LLM("", ""), os.getcwd(), role="medium")
    tools = "\n".join(json.dumps(t) for t in agent.schemas())
    system = f"[{random.random()}] " + agent.messages[0]["content"] + "\n\n# Tools\n" + tools
    return [{"role": "system", "content": system}, {"role": "user", "content": TASK}]


def main():
    servers = [a.partition("=") for a in sys.argv[1:]]
    reps = int(os.environ.get("REPS", "3"))
    for name, _, spec in servers:
        url, _, model = spec.partition("@")
        stream(url, model, [{"role": "user", "content": "hi"}], 4)  # warm-up / load model
        cold, warm = [], []
        for _ in range(reps):
            msgs = agent_messages()
            c = stream(url, model, msgs)
            msgs += [{"role": "assistant", "content": "Let me look at the helpers first."},
                     {"role": "user", "content": "Tool result:\n" + TOOL_RESULT + "\nNow write the module."}]
            cold.append(c)
            warm.append(stream(url, model, msgs))

        def avg(rs, k):
            return sum(r[k] for r in rs) / len(rs)
        print(f"{name:<10} prompt≈{cold[0]['prompt_tokens']} tok | cold: first token {avg(cold, 'ttft'):5.2f}s, "
              f"{avg(cold, 'tok_s'):5.1f} tok/s | agent turn: first token {avg(warm, 'ttft'):5.2f}s, "
              f"{avg(warm, 'tok_s'):5.1f} tok/s", flush=True)


if __name__ == "__main__":
    main()
