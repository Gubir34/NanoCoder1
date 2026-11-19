#!/usr/bin/env python3
"""AgentCoder Server - a small OpenAI-compatible inference server tuned for coding agents.

Agents resend the whole conversation every turn. This server keeps the KV cache of the last
request per model and only computes the new suffix, so a 3k-token agent turn usually costs a few
hundred tokens of prefill. Runs HF models in bf16 (fast on CPUs with AMX/AVX512-BF16, or on a GPU),
optionally with a LoRA adapter merged in.

    pip install torch transformers peft
    python agentcoder_server.py \
        --model qwen2.5-coder:0.5b=Qwen/Qwen2.5-Coder-0.5B-Instruct \
        --model agentcoder-worker:0.5b=Qwen/Qwen2.5-Coder-0.5B-Instruct+finetune/adapters/agentcoder-worker-0.5b
    agentcoder            # finds it automatically on port 8090
"""
from __future__ import annotations

import argparse
import json
import re
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

CALL_RE = re.compile(r"<tool_call>\s*(.*?)\s*(?:</tool_call>|$)", re.S)
TAG = "<tool_call>"


def pick_device(name: str) -> str:
    if name != "auto":
        return name
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class Engine:
    """One model + its prefix cache. Requests are serialized per engine."""

    def __init__(self, name: str, source: str, device: str):
        base, _, adapter = source.partition("+")
        self.name, self.device = name, device
        self.tok = AutoTokenizer.from_pretrained(base)
        model = AutoModelForCausalLM.from_pretrained(base, dtype=torch.bfloat16)
        if adapter:
            from peft import PeftModel
            model = PeftModel.from_pretrained(model, adapter).merge_and_unload()
        self.model = model.to(device).eval()
        self.stop_ids = {i for i in (self.tok.eos_token_id, self.tok.convert_tokens_to_ids("<|im_end|>"),
                                     self.tok.convert_tokens_to_ids("<|endoftext|>")) if isinstance(i, int)}
        self.lock = threading.Lock()
        self.cached_ids: list = []
        self.cache = None
        self.stats = {"requests": 0, "prompt_tokens": 0, "reused_tokens": 0}

    def _forward(self, ids: list):
        x = torch.tensor([ids], device=self.device)
        out = self.model(input_ids=x, past_key_values=self.cache, use_cache=True)
        self.cache = out.past_key_values
        return out.logits[0, -1]

    def _prefill(self, ids: list):
        common = 0
        for a, b in zip(self.cached_ids, ids):
            if a != b:
                break
            common += 1
        common = min(common, len(ids) - 1)  # always run at least one token to get logits
        if self.cache is None or common == 0:
            self.cache, common = DynamicCache(), 0
        else:
            extra = self.cache.get_seq_length() - common
            if extra > 0:
                self.cache.crop(-extra)  # negative = drop that many tokens (works on old and new transformers)
        self.stats["reused_tokens"] += common
        logits = None
        for i in range(common, len(ids), 1024):  # chunked prefill keeps memory flat on long prompts
            logits = self._forward(ids[i:i + 1024])
        self.cached_ids = list(ids)
        return logits, common

    @torch.inference_mode()
    def generate(self, prompt: str, max_tokens: int, temperature: float, on_text):
        ids = self.tok(prompt, add_special_tokens=False).input_ids
        self.stats["requests"] += 1
        self.stats["prompt_tokens"] += len(ids)
        logits, reused = self._prefill(ids)
        out_ids, sent = [], 0
        for _ in range(max_tokens):
            if temperature > 0:
                probs = torch.softmax(logits.float() / temperature, -1)
                nxt = int(torch.multinomial(probs, 1))
            else:
                nxt = int(torch.argmax(logits))
            if nxt in self.stop_ids:
                break
            out_ids.append(nxt)
            self.cached_ids.append(nxt)
            text = self.tok.decode(out_ids, skip_special_tokens=False)
            # stream text up to a tool call; hold back a possible partial "<tool_call>" tag
            cut = text.find(TAG)
            visible = text if cut < 0 else text[:cut]
            if cut < 0:
                for k in range(min(len(TAG) - 1, len(visible)), 0, -1):
                    if TAG.startswith(visible[-k:]):
                        visible = visible[:-k]
                        break
            if len(visible) > sent and not visible.endswith("�"):
                on_text(visible[sent:])
                sent = len(visible)
            logits = self._forward([nxt])
        text = self.tok.decode(out_ids, skip_special_tokens=False)
        return text, sent, len(ids), len(out_ids), reused


def normalize(messages: list) -> list:
    """OpenAI wire format -> HF chat template format (tool call arguments as dicts)."""
    out = []
    for m in messages:
        m = dict(m)
        if m.get("tool_calls"):
            calls = []
            for tc in m["tool_calls"]:
                fn = dict(tc.get("function") or {})
                if isinstance(fn.get("arguments"), str):
                    try:
                        fn["arguments"] = json.loads(fn["arguments"] or "{}")
                    except json.JSONDecodeError:
                        pass
                calls.append({"type": "function", "function": fn})
            m["tool_calls"] = calls
        if m.get("content") is None:
            m["content"] = ""
        out.append(m)
    return out


def split_calls(text: str):
    calls = []
    for i, mt in enumerate(CALL_RE.finditer(text)):
        try:
            obj = json.loads(mt.group(1))
        except json.JSONDecodeError:
            return text.strip(), []  # malformed: hand the raw text back, the agent will complain
        if not isinstance(obj, dict):
            return text.strip(), []
        calls.append({"index": i, "id": f"call_{uuid.uuid4().hex[:12]}", "type": "function",
                      "function": {"name": obj.get("name", ""), "arguments": json.dumps(obj.get("arguments", {}))}})
    return CALL_RE.sub("", text).strip(), calls


def make_handler(engines: dict, default_max: int):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *a):
            pass

        def _json(self, obj, code=200):
            data = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            if self.path.rstrip("/").endswith("/models"):
                return self._json({"object": "list", "data": [
                    {"id": n, "object": "model", "owned_by": "agentcoder"} for n in engines]})
            if self.path.rstrip("/").endswith("/stats"):
                return self._json({n: e.stats for n, e in engines.items()})
            self._json({"error": {"message": "not found"}}, 404)

        def do_POST(self):
            if not self.path.rstrip("/").endswith("/chat/completions"):
                return self._json({"error": {"message": "not found"}}, 404)
            body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
            eng = engines.get(body.get("model")) or (next(iter(engines.values())) if len(engines) == 1 else None)
            if not eng:
                return self._json({"error": {"message": f"unknown model {body.get('model')!r}"}}, 404)
            prompt = eng.tok.apply_chat_template(normalize(body.get("messages", [])), tools=body.get("tools") or None,
                                                 tokenize=False, add_generation_prompt=True)
            stream = bool(body.get("stream"))
            max_tokens = int(body.get("max_tokens") or default_max)
            temperature = float(body.get("temperature") or 0)
            cid, created = f"chatcmpl-{uuid.uuid4().hex[:16]}", int(time.time())

            def chunk(delta, finish=None, usage=None):
                d = {"id": cid, "object": "chat.completion.chunk", "created": created, "model": eng.name,
                     "choices": [{"index": 0, "delta": delta, "finish_reason": finish}] if delta is not None else []}
                if usage:
                    d["usage"] = usage
                self.wfile.write(f"data: {json.dumps(d)}\n\n".encode())
                self.wfile.flush()

            if stream:
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Cache-Control", "no-cache")
                self.send_header("Connection", "close")
                self.end_headers()
                chunk({"role": "assistant", "content": ""})
            t0 = time.time()
            with eng.lock:
                text, sent, n_in, n_out, reused = eng.generate(
                    prompt, max_tokens, temperature, (lambda t: chunk({"content": t})) if stream else (lambda t: None))
            content, calls = split_calls(text)
            usage = {"prompt_tokens": n_in, "completion_tokens": n_out, "total_tokens": n_in + n_out,
                     "prompt_tokens_details": {"cached_tokens": reused}}
            print(f"{eng.name}: {n_in} in ({reused} cached) · {n_out} out · {time.time() - t0:.1f}s", flush=True)
            if stream:
                if not calls and len(content) > sent:  # e.g. malformed tool call text
                    chunk({"content": text[sent:]})
                if calls:
                    chunk({"tool_calls": calls})
                chunk({}, "tool_calls" if calls else "stop")
                chunk(None, usage=usage)
                self.wfile.write(b"data: [DONE]\n\n")
                self.wfile.flush()
                self.close_connection = True
                return
            msg = {"role": "assistant", "content": content}
            if calls:
                msg["tool_calls"] = [{k: v for k, v in c.items() if k != "index"} for c in calls]
            self._json({"id": cid, "object": "chat.completion", "created": created, "model": eng.name,
                        "choices": [{"index": 0, "message": msg, "finish_reason": "tool_calls" if calls else "stop"}],
                        "usage": usage})
    return Handler


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--model", action="append", required=True, metavar="NAME=MODEL[+ADAPTER]",
                    help="served name = HF id or local path, optionally +LoRA adapter dir (repeatable)")
    ap.add_argument("--port", type=int, default=8090)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--device", default="auto", help="auto, cpu, cuda, mps")
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--max-tokens", type=int, default=2048)
    args = ap.parse_args()
    if args.threads:
        torch.set_num_threads(args.threads)
    device = pick_device(args.device)
    engines = {}
    for spec in args.model:
        name, eq, source = spec.partition("=")
        if not eq:
            name, source = spec.rsplit("/", 1)[-1], spec
        t0 = time.time()
        engines[name] = Engine(name, source, device)
        print(f"loaded {name} from {source} on {device} in {time.time() - t0:.0f}s", flush=True)
    print(f"AgentCoder Server on http://{args.host}:{args.port}/v1 · models: {', '.join(engines)}", flush=True)
    ThreadingHTTPServer((args.host, args.port), make_handler(engines, args.max_tokens)).serve_forever()


if __name__ == "__main__":
    main()
