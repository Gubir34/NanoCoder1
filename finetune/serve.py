"""Minimal OpenAI-compatible server for a HF model (+ optional LoRA adapter). No Ollama needed.

    python finetune/serve.py --base Qwen/Qwen2.5-Coder-0.5B-Instruct \
        --adapter finetune/adapters/nanocoder-worker-0.5b --name nanocoder-worker --port 8080
    nanocoder --base-url http://localhost:8080/v1 -m nanocoder-worker

Greedy decoding, tool calls parsed from Qwen-style <tool_call> blocks. One request at a time.
"""
import argparse
import json
import re
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

CALL_RE = re.compile(r"<tool_call>\s*(.*?)\s*(?:</tool_call>|$)", re.S)


def load(base, adapter=None):
    tok = AutoTokenizer.from_pretrained(base)
    model = AutoModelForCausalLM.from_pretrained(base, dtype=torch.bfloat16)
    if adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter).merge_and_unload()
    model.eval()
    return tok, model


def normalize(messages):
    """OpenAI wire format -> what the HF chat template expects (arguments as dicts)."""
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


def parse(text):
    calls = []
    for i, mt in enumerate(CALL_RE.finditer(text)):
        try:
            obj = json.loads(mt.group(1))
            calls.append({"id": f"call_{int(time.time() * 1000)}_{i}", "type": "function", "function": {
                "name": obj.get("name", ""), "arguments": json.dumps(obj.get("arguments", {}))}})
        except (json.JSONDecodeError, AttributeError):
            return text.strip(), []  # leave malformed calls in the text; the agent will complain
    return CALL_RE.sub("", text).strip(), calls


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="Qwen/Qwen2.5-Coder-0.5B-Instruct")
    ap.add_argument("--adapter")
    ap.add_argument("--name", default="local")
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--max-new-tokens", type=int, default=768)
    ap.add_argument("--threads", type=int, default=0)
    args = ap.parse_args()
    if args.threads:
        torch.set_num_threads(args.threads)
    tok, model = load(args.base, args.adapter)
    lock = threading.Lock()

    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def reply(self, obj, code=200):
            data = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            self.reply({"object": "list", "data": [{"id": args.name, "object": "model"}]})

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            prompt = tok.apply_chat_template(normalize(body["messages"]), tools=body.get("tools") or None,
                                             tokenize=False, add_generation_prompt=True)
            ids = tok(prompt, return_tensors="pt", add_special_tokens=False).input_ids
            with lock, torch.no_grad():
                out = model.generate(ids, max_new_tokens=body.get("max_tokens") or args.max_new_tokens,
                                     do_sample=False, pad_token_id=tok.eos_token_id)
            new = out[0, ids.shape[1]:]
            text = tok.decode(new, skip_special_tokens=True)
            content, calls = parse(text)
            msg = {"role": "assistant", "content": content}
            if calls:
                msg["tool_calls"] = calls
            self.reply({"object": "chat.completion", "model": args.name,
                        "choices": [{"index": 0, "message": msg, "finish_reason": "tool_calls" if calls else "stop"}],
                        "usage": {"prompt_tokens": ids.shape[1], "completion_tokens": len(new)}})

    print(f"serving {args.name} on http://localhost:{args.port}/v1", flush=True)
    ThreadingHTTPServer(("127.0.0.1", args.port), H).serve_forever()


if __name__ == "__main__":
    main()
