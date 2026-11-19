#!/usr/bin/env python3
"""nanocoder - a tiny, local, terminal coding agent.

Talks to any OpenAI-compatible server (Ollama, LM Studio, llama.cpp, vLLM, ...).
Zero dependencies: just Python 3.8+.
"""
from __future__ import annotations

import argparse
import atexit
import difflib
import fnmatch
import json
import os
import platform
import random
import re
import shutil
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from datetime import date

try:
    import readline  # enables line editing + history for input()
except ImportError:  # pragma: no cover - Windows without pyreadline
    readline = None

VERSION = "0.1.0"
DEFAULT_MODEL = "qwen2.5-coder:7b"
DEFAULT_BASE_URL = "http://localhost:11434/v1"
IGNORED_DIRS = {".git", "node_modules", "__pycache__", ".venv", "venv", "env", "dist", "build",
                ".next", ".mypy_cache", ".pytest_cache", ".idea", ".vscode", "target", ".tox"}
PROJECT_NOTES = ("NANOCODER.md", "AGENTS.md", "CLAUDE.md")
MAX_TOOL_OUTPUT = 20_000
MAX_STEPS = 40
VERBS = ["Thinking", "Pondering", "Tinkering", "Cooking", "Noodling", "Scheming",
         "Crunching", "Brewing", "Untangling", "Hacking"]

# ───────────────────────────── terminal helpers ─────────────────────────────

COLOR = sys.stdout.isatty() and "NO_COLOR" not in os.environ
if os.name == "nt":
    os.system("")  # enable ANSI escapes on Windows 10+

ANSI_RE = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]")
_last_char = "\n"


def w(s: str) -> None:
    """Write to stdout and remember whether the cursor sits at column 0."""
    global _last_char
    if not s:
        return
    sys.stdout.write(s)
    sys.stdout.flush()
    visible = ANSI_RE.sub("", s)
    if visible:
        _last_char = visible[-1]


def ensure_nl() -> None:
    if _last_char not in "\n\r":
        w("\n")


def sgr(code: str):
    return lambda s: f"\x1b[{code}m{s}\x1b[0m" if COLOR else str(s)


bold, dim, italic = sgr("1"), sgr("2"), sgr("3")
red, green, yellow, cyan = sgr("31"), sgr("32"), sgr("33"), sgr("36")
orange = sgr("38;5;208")
ACCENT = "\x1b[1;38;5;208m"


def term_width() -> int:
    return max(40, min(shutil.get_terminal_size((100, 24)).columns, 120))


def vlen(s: str) -> int:
    return len(ANSI_RE.sub("", s))


def clip(s: str, n: int) -> str:
    return s if len(s) <= n else s[: max(0, n - 1)] + "…"


class Spinner:
    FRAMES = ["·", "✢", "✳", "✶", "✻", "✽", "✻", "✶", "✳", "✢"]

    def __init__(self):
        self.label = "Thinking"
        self._stop = threading.Event()
        self._thread = None
        self.t0 = 0.0

    def start(self, label: str) -> None:
        self.label = label
        if self._thread or not sys.stdout.isatty():
            return
        ensure_nl()
        self.t0 = time.time()
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        i = 0
        while not self._stop.wait(0.1):
            secs = int(time.time() - self.t0)
            frame = self.FRAMES[i % len(self.FRAMES)]
            sys.stdout.write(f"\r\x1b[K{orange(frame)} {orange(self.label + '…')} "
                             f"{dim(f'({secs}s · ctrl+c to interrupt)')}")
            sys.stdout.flush()
            i += 1

    def stop(self) -> None:
        if self._thread:
            self._stop.set()
            self._thread.join()
            self._thread = None
            w("\r\x1b[K")


# ─────────────────────────── streaming markdown ────────────────────────────

class TagRouter:
    """Splits a token stream on <think>/<tool_call> tags, even when tags span chunks."""
    TAGS = {"<think>": "think", "</think>": "text", "<tool_call>": "hide", "</tool_call>": "text"}

    def __init__(self, sink):
        self.sink, self.mode, self.buf = sink, "text", ""

    def feed(self, s: str) -> None:
        self.buf += s
        while self.buf:
            i = self.buf.find("<")
            if i == -1:
                self.sink(self.mode, self.buf)
                self.buf = ""
                return
            if i:
                self.sink(self.mode, self.buf[:i])
                self.buf = self.buf[i:]
            for tag, mode in self.TAGS.items():
                if self.buf.startswith(tag):
                    self.mode, self.buf = mode, self.buf[len(tag):]
                    break
            else:
                if any(t.startswith(self.buf) for t in self.TAGS):
                    return  # partial tag, wait for more
                self.sink(self.mode, "<")
                self.buf = self.buf[1:]

    def flush(self) -> None:
        if self.buf:
            self.sink(self.mode, self.buf)
            self.buf = ""


class Markdown:
    """Char-by-char markdown renderer: headers, bullets, **bold**, `code`, fenced blocks."""

    def __init__(self):
        self.line_start, self.pend = True, ""
        self.code = self.bold = self.icode = self.header = self.star = False
        self.cur = ""

    def feed(self, s: str) -> None:
        for ch in s:
            self._ch(ch)

    def _ch(self, ch: str) -> None:
        if not self.line_start:
            return self._put(ch)
        self.pend += ch
        st = self.pend.lstrip()
        waiting = st == "" or "```".startswith(st) or st.startswith("```")
        if not self.code:
            waiting = waiting or re.fullmatch(r"#{1,6}|[-*+]", st) is not None
        if ch != "\n" and waiting:
            return
        if st.startswith("```"):
            lang = st[3:].strip()
            self.code = not self.code
            self.pend = ""
            if self.code and lang:
                w(dim(f"  ┄ {lang}\n"))
            return
        pend, self.pend, self.line_start = self.pend, "", False
        if self.code:
            w("  ")
        else:
            indent = pend[: len(pend) - len(pend.lstrip())]
            if re.match(r"#{1,6}\s", st):
                self.header, pend = True, indent + st.lstrip("#").lstrip()
            elif re.match(r"[-*+]\s", st):
                pend = indent + "• " + st[2:]
        for c in pend:
            self._put(c)

    def _style(self) -> str:
        if not COLOR:
            return ""
        if self.code:
            return "\x1b[38;5;117m"
        s = "\x1b[1m" if (self.bold or self.header) else ""
        return s + ("\x1b[38;5;216m" if self.icode else "")

    def _emit(self, c: str) -> None:
        st = self._style()
        if st != self.cur:
            w("\x1b[0m" + st if COLOR else "")
            self.cur = st
        w(c)

    def _put(self, c: str) -> None:
        if c == "\n":
            if self.star:
                self.star = False
                self._emit("*")
            if self.cur:
                w("\x1b[0m")
                self.cur = ""
            self.line_start, self.header, self.bold, self.icode = True, False, False, False
            w("\n")
            return
        if self.code:
            return self._emit(c)
        if self.star:
            self.star = False
            if c == "*":
                self.bold = not self.bold
                return
            self._emit("*")
        if c == "*":
            self.star = True
        elif c == "`":
            self.icode = not self.icode
        else:
            self._emit(c)

    def flush(self) -> None:
        if self.line_start and self.pend:
            pend, self.pend, self.line_start = self.pend, "", False
            for c in pend:
                self._put(c)
        if self.star:
            self.star = False
            self._emit("*")
        if self.cur:
            w("\x1b[0m")
            self.cur = ""


# ─────────────────────────────── LLM client ────────────────────────────────

class LLMError(Exception):
    def __init__(self, code: int, body: str):
        super().__init__(f"HTTP {code}: {body[:400]}" if code else body)
        self.code, self.body = code, body


class LLM:
    def __init__(self, base_url: str, model: str, api_key: str = "", temperature: float = 0.2):
        self.base_url, self.model = base_url.rstrip("/"), model
        self.api_key, self.temperature = api_key, temperature
        self.partial = ""
        self.usage = None

    def _req(self, path: str, body=None, timeout: float = 900):
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = "Bearer " + self.api_key
        data = json.dumps(body).encode() if body is not None else None
        req = urllib.request.Request(self.base_url + path, data=data, headers=headers,
                                     method="POST" if data else "GET")
        try:
            return urllib.request.urlopen(req, timeout=timeout)
        except urllib.error.HTTPError as e:
            raise LLMError(e.code, e.read().decode("utf-8", "replace")) from None
        except (urllib.error.URLError, OSError) as e:
            raise LLMError(0, f"can't reach {self.base_url} ({getattr(e, 'reason', e)})") from None

    def models(self) -> list:
        with self._req("/models", timeout=4) as r:
            return [m.get("id", "") for m in json.load(r).get("data", [])]

    def chat(self, messages: list, tools, sink):
        """Stream a completion. sink(mode, text) gets 'text' / 'think' / 'hide' / 'tool' events.
        Returns (raw_content, [{"id", "name", "arguments"}])."""
        body = {"model": self.model, "messages": messages, "stream": True,
                "temperature": self.temperature, "stream_options": {"include_usage": True}}
        if tools:
            body["tools"] = tools
        router = TagRouter(sink)
        content, calls, got, rest = "", {}, False, []
        self.partial = ""
        with self._req("/chat/completions", body) as resp:
            for raw in resp:
                line = raw.decode("utf-8", "replace").strip()
                if not line:
                    continue
                if not line.startswith("data:"):
                    rest.append(line)
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                try:
                    chunk = json.loads(data)
                except json.JSONDecodeError:
                    continue
                if chunk.get("error"):
                    raise LLMError(500, json.dumps(chunk["error"]))
                got = True
                self.usage = chunk.get("usage") or self.usage
                for choice in chunk.get("choices") or []:
                    delta = choice.get("delta") or choice.get("message") or {}
                    thought = delta.get("reasoning_content") or delta.get("reasoning") or delta.get("thinking")
                    if thought:
                        sink("think", thought)
                    if delta.get("content"):
                        content += delta["content"]
                        self.partial = content
                        router.feed(delta["content"])
                    for tc in delta.get("tool_calls") or []:
                        self._merge_call(calls, tc, sink)
        if not got and rest:  # server ignored stream=True and sent plain JSON
            try:
                obj = json.loads("\n".join(rest))
            except json.JSONDecodeError:
                raise LLMError(0, "unexpected response: " + "\n".join(rest)[:300]) from None
            if obj.get("error"):
                raise LLMError(500, json.dumps(obj["error"]))
            msg = (obj.get("choices") or [{}])[0].get("message") or {}
            if msg.get("reasoning_content"):
                sink("think", msg["reasoning_content"])
            content = msg.get("content") or ""
            router.feed(content)
            for i, tc in enumerate(msg.get("tool_calls") or []):
                tc.setdefault("index", i)
                self._merge_call(calls, tc, sink)
        router.flush()
        return content, [{"id": c["id"] or f"call_{i}_{random.randrange(16**6):06x}",
                          "name": c["name"], "arguments": c["arguments"] or "{}"}
                         for i, c in sorted(calls.items())]

    @staticmethod
    def _merge_call(calls: dict, tc: dict, sink) -> None:
        idx = tc.get("index")
        if idx is None:
            tid = tc.get("id")
            by_id = {v["id"]: k for k, v in calls.items()}
            idx = by_id.get(tid, len(calls)) if tid else (max(calls) if calls else 0)
        slot = calls.setdefault(idx, {"id": "", "name": "", "arguments": ""})
        if tc.get("id"):
            slot["id"] = tc["id"]
        fn = tc.get("function") or {}
        if fn.get("name") and not slot["name"]:
            slot["name"] = fn["name"]
            sink("tool", fn["name"])
        args = fn.get("arguments")
        if isinstance(args, dict):
            slot["arguments"] = json.dumps(args)
        elif args:
            slot["arguments"] += args


TOOL_CALL_RE = re.compile(r"<tool_call>\s*(.*?)\s*(?:</tool_call>|$)", re.S)
THINK_RE = re.compile(r"<think>.*?(?:</think>|$)", re.S)


def parse_text_calls(content: str) -> list:
    """Parse <tool_call>{"name":..,"arguments":..}</tool_call> blocks from plain text."""
    calls = []
    for i, m in enumerate(TOOL_CALL_RE.finditer(THINK_RE.sub("", content))):
        raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", m.group(1).strip())
        try:
            obj = json.loads(raw)
            name = obj.get("name", "")
            args = obj.get("arguments", obj.get("parameters", {}))
        except (json.JSONDecodeError, AttributeError):
            name, args = "(invalid)", raw
        calls.append({"id": f"call_t{i}_{random.randrange(16**6):06x}", "name": name,
                      "arguments": args if isinstance(args, str) else json.dumps(args)})
    return calls


# ───────────────────────────────── tools ───────────────────────────────────

def _tool(name, desc, props, required):
    return {"type": "function", "function": {"name": name, "description": desc, "parameters": {
        "type": "object", "properties": props, "required": required}}}


def _s(d): return {"type": "string", "description": d}
def _i(d): return {"type": "integer", "description": d}
def _b(d): return {"type": "boolean", "description": d}


TOOLS = [
    _tool("read_file", "Read a text file. Output lines are prefixed with line numbers (number + tab).",
          {"path": _s("File path, relative to the project root."),
           "offset": _i("1-based line to start from (default 1)."),
           "limit": _i("Max lines to return (default 500).")}, ["path"]),
    _tool("write_file", "Create a new file or fully overwrite an existing one.",
          {"path": _s("File path."), "content": _s("Full file content.")}, ["path", "content"]),
    _tool("edit_file", "Replace an exact snippet in a file. old_string must match the file exactly "
          "(whitespace included, no line-number prefixes) and be unique unless replace_all is true. "
          "Read the file first.",
          {"path": _s("File path."), "old_string": _s("Exact text to replace."),
           "new_string": _s("Replacement text."), "replace_all": _b("Replace every occurrence.")},
          ["path", "old_string", "new_string"]),
    _tool("bash", "Run a shell command at the project root and return stdout+stderr. Use it for git, "
          "tests, builds, running code, installing packages. Each call starts a fresh shell; "
          "no interactive commands.",
          {"command": _s("The command to run."), "timeout": _i("Seconds before kill (default 120).")},
          ["command"]),
    _tool("glob", "Find files by glob pattern like '**/*.py' or 'src/*.ts'. Newest first.",
          {"pattern": _s("Glob pattern."), "path": _s("Directory to search (default project root).")},
          ["pattern"]),
    _tool("grep", "Search file contents with a regular expression. Returns path:line:text matches.",
          {"pattern": _s("Regex to search for."), "path": _s("File or directory (default root)."),
           "glob": _s("Only search files matching this glob, e.g. '*.py'."),
           "ignore_case": _b("Case-insensitive search.")}, ["pattern"]),
]
DISPLAY = {"read_file": "Read", "write_file": "Write", "edit_file": "Edit",
           "bash": "Bash", "glob": "Glob", "grep": "Grep"}


class ToolError(Exception):
    pass


class ToolDenied(Exception):
    pass


def diff_lines(old: str, new: str, limit: int = 40) -> list:
    d = list(difflib.unified_diff(old.splitlines(), new.splitlines(), lineterm="", n=2))[2:]
    width = term_width() - 8
    out = []
    for line in d[:limit]:
        line = clip(line, width)
        color = green if line.startswith("+") else red if line.startswith("-") else \
            cyan if line.startswith("@@") else dim
        out.append(color(line))
    if len(d) > limit:
        out.append(dim(f"… +{len(d) - limit} more lines"))
    return out


def truncate(s: str, n: int = MAX_TOOL_OUTPUT) -> str:
    if len(s) <= n:
        return s
    half = n // 2
    return s[:half] + f"\n\n... [{len(s) - n} chars truncated] ...\n\n" + s[-half:]


# ───────────────────────────────── agent ───────────────────────────────────

class Stream:
    """Receives streamed events from the LLM and draws them."""

    def __init__(self, agent: "Agent"):
        self.a, self.md, self.state = agent, Markdown(), None
        self.think_t0 = 0.0
        self.tool_spin = False

    def __call__(self, mode: str, text: str) -> None:
        a = self.a
        if not text or mode == "hide":
            return
        if mode == "tool":
            self._close_think()
            self.md.flush()
            if not self.tool_spin:
                self.tool_spin = True
                a.spinner.start(f"Preparing {DISPLAY.get(text, text)}")
            return
        if mode == "think":
            if self.state != "think":
                self.state, self.think_t0 = "think", time.time()
                if a.show_think:
                    a.spinner.stop()
                    ensure_nl()
                    w(dim("✻ "))
                else:
                    a.spinner.label = "Thinking"
            if a.show_think:
                w(dim(italic(text)))
            return
        if self.state != "text":
            if not text.strip():
                return
            a.spinner.stop()
            self._close_think()
            ensure_nl()
            w(bold("● "))
            text, self.state = text.lstrip(), "text"
        self.md.feed(text)

    def _close_think(self) -> None:
        if self.state == "think":
            self.state = "thought"
            if not self.a.show_think:
                self.a.spinner.stop()
            ensure_nl()
            w(dim(f"✻ Thought for {int(time.time() - self.think_t0)}s\n"))

    def end(self) -> None:
        self._close_think()
        self.md.flush()
        ensure_nl()


class Agent:
    def __init__(self, llm: LLM, cwd: str, yolo=False, text_tools=False, ctx_limit=32768):
        self.llm, self.cwd = llm, os.path.abspath(cwd)
        self.yolo, self.text_tools, self.ctx_limit = yolo, text_tools, ctx_limit
        self.show_think = False
        self.always: set = set()
        self.spinner = Spinner()
        self.pending_ctx = ""
        self.stop_turn = False
        self.tools = {"read_file": self.t_read_file, "write_file": self.t_write_file,
                      "edit_file": self.t_edit_file, "bash": self.t_bash,
                      "glob": self.t_glob, "grep": self.t_grep}
        self.reset()

    # ── context ──
    def reset(self) -> None:
        self.messages = [{"role": "system", "content": self.system_prompt()}]

    def system_prompt(self) -> str:
        try:
            entries = sorted(e for e in os.listdir(self.cwd) if e not in IGNORED_DIRS)
        except OSError:
            entries = []
        listing = ", ".join(e + ("/" if os.path.isdir(os.path.join(self.cwd, e)) else "")
                            for e in entries[:60]) or "(empty)"
        p = f"""You are nanocoder, a small but sharp coding agent running in the user's terminal. \
You solve software engineering tasks by reading, searching and editing files and running commands with your tools.

# Environment
- Project root (cwd): {self.cwd}
- Platform: {platform.system()} {platform.release()} · Date: {date.today().isoformat()}
- Git repo: {"yes" if os.path.isdir(os.path.join(self.cwd, ".git")) else "no"}
- Top-level entries: {listing}

# How to work
- Look before you leap: use glob/grep/read_file to explore. Never guess what a file contains.
- Prefer small, targeted edits with edit_file. Copy old_string exactly from read_file output, without line numbers. Use write_file for new files or full rewrites.
- After changing code, verify it when you can (run tests, a linter, or the program) with bash.
- Keep going until the task is actually done. Don't ask permission for routine steps; ask only when the request is truly ambiguous.
- Be concise and direct. Reply in the user's language. Use Markdown. No filler, no long recaps.
- Never run destructive commands (rm -rf, git reset --hard, force push...) unless the user explicitly asked."""
        if self.text_tools:
            p += "\n\n# Tools\nCall a tool by writing exactly:\n<tool_call>\n" \
                 '{"name": "tool_name", "arguments": {"arg": "value"}}\n</tool_call>\n' \
                 "You may emit several tool calls in one reply. Then stop and wait: results come back " \
                 "as <tool_result> messages. When the task is done, reply normally without tool calls.\n" \
                 "Available tools:\n"
            for t in TOOLS:
                f = t["function"]
                params = ", ".join(f"{k}: {v['type']}{'' if k in f['parameters']['required'] else '?'}"
                                   for k, v in f["parameters"]["properties"].items())
                p += f"- {f['name']}({params}): {f['description']}\n"
        for name in PROJECT_NOTES:
            path = os.path.join(self.cwd, name)
            if os.path.isfile(path):
                with open(path, encoding="utf-8", errors="replace") as fh:
                    p += f"\n\n# Project notes ({name})\n" + fh.read()[:8000]
                break
        return p

    def est_tokens(self) -> int:
        return len(json.dumps(self.messages, ensure_ascii=False)) // 4

    def resolve(self, path: str) -> str:
        return os.path.normpath(os.path.join(self.cwd, os.path.expanduser(str(path))))

    def rel(self, path: str) -> str:
        r = os.path.relpath(path, self.cwd)
        return path if r.startswith("..") else r

    # ── main loop ──
    def turn(self, text: str) -> None:
        text = self.expand_mentions(text)
        if self.pending_ctx:
            text, self.pending_ctx = self.pending_ctx + "\n\n" + text, ""
        if self.est_tokens() > self.ctx_limit * 0.8:
            self.compact()
        self.messages.append({"role": "user", "content": text})
        self.stop_turn = False
        try:
            for _ in range(MAX_STEPS):
                if not self.step() or self.stop_turn:
                    break
            else:
                w(yellow(f"  (stopped after {MAX_STEPS} steps — say 'continue' to keep going)\n"))
        except KeyboardInterrupt:
            self.spinner.stop()
            self.repair_history()
            ensure_nl()
            w(red("  ⎿  Interrupted") + dim(" · tell nanocoder what to do instead\n"))
        except LLMError as e:
            self.spinner.stop()
            self.repair_history()
            ensure_nl()
            w(red(f"  ✗ {e}\n"))

    def step(self) -> bool:
        """One model call + tool execution. Returns True if tools were called."""
        stream = Stream(self)
        self.spinner.start(random.choice(VERBS))
        try:
            content, calls = self.llm.chat(self.messages, None if self.text_tools else TOOLS, stream)
        except LLMError as e:
            if not self.text_tools and e.code in (400, 422, 500) and "tool" in e.body.lower():
                self.spinner.stop()
                self.text_tools = True
                self.messages[0]["content"] = self.system_prompt()
                w(dim("  (model has no native tool calling → switching to text tool calls)\n"))
                return self.step()
            raise
        finally:
            self.spinner.stop()
            stream.end()

        text_calls = parse_text_calls(content)
        clean = THINK_RE.sub("", content).strip()
        if self.text_tools:
            calls = text_calls
            self.messages.append({"role": "assistant", "content": clean})
        else:
            calls = calls or text_calls  # some models leak calls into plain text
            msg = {"role": "assistant", "content": TOOL_CALL_RE.sub("", clean).strip()}
            if calls:
                msg["tool_calls"] = [{"id": c["id"], "type": "function", "function": {
                    "name": c["name"], "arguments": c["arguments"]}} for c in calls]
            self.messages.append(msg)

        results = []
        for c in calls:
            out = "Skipped: the user stopped this turn." if self.stop_turn else self.run_tool(c)
            if self.text_tools:
                results.append(f'<tool_result name="{c["name"]}">\n{out}\n</tool_result>')
            else:
                self.messages.append({"role": "tool", "tool_call_id": c["id"], "content": out})
        if results:
            self.messages.append({"role": "user", "content": "\n\n".join(results)})
        return bool(calls)

    def repair_history(self) -> None:
        """Keep the transcript valid after an interrupt or error."""
        last = self.messages[-1]
        if last["role"] == "user" and self.llm.partial:
            self.messages.append({"role": "assistant",
                                  "content": THINK_RE.sub("", self.llm.partial) + "\n[interrupted]"})
            self.llm.partial = ""
        for i in range(len(self.messages) - 1, 0, -1):
            m = self.messages[i]
            if m["role"] == "assistant":
                done = {x.get("tool_call_id") for x in self.messages[i + 1:]}
                for tc in m.get("tool_calls") or []:
                    if tc["id"] not in done:
                        self.messages.append({"role": "tool", "tool_call_id": tc["id"],
                                              "content": "Interrupted by the user."})
                if self.text_tools and m is self.messages[-1] and parse_text_calls(m["content"]):
                    self.messages.append({"role": "user", "content": "[interrupted by the user]"})
                break

    def run_tool(self, call: dict) -> str:
        name, raw = call["name"], call["arguments"]
        try:
            args = json.loads(raw) if isinstance(raw, str) else raw
            if not isinstance(args, dict):
                raise ValueError("arguments must be a JSON object")
        except ValueError as e:
            ensure_nl()
            w(f"{red('●')} {bold(name or '?')}\n  ⎿  {red('invalid tool arguments')}\n")
            return f"Error: could not parse tool arguments as a JSON object ({e}). Retry with valid JSON."
        ensure_nl()
        w(f"{green('●')} {bold(DISPLAY.get(name, name))}{dim('(' + self.arg_summary(name, args) + ')')}\n")
        fn = self.tools.get(name)
        if not fn:
            w(f"  ⎿  {red('unknown tool')}\n")
            return f"Error: unknown tool '{name}'. Available: {', '.join(self.tools)}."
        try:
            out = fn(**args)
        except ToolDenied as e:
            said = str(e).partition("and said: ")[2]
            w(f"  ⎿  {yellow('Denied' + (': ' + clip(said, term_width() - 20) if said else ''))}\n")
            return str(e)
        except ToolError as e:
            w(f"  ⎿  {red(clip(str(e).splitlines()[0], term_width() - 8))}\n")
            return f"Error: {e}"
        except TypeError as e:
            w(f"  ⎿  {red('bad arguments: ' + str(e))}\n")
            return f"Error: bad arguments for {name}: {e}"
        except (OSError, re.error) as e:
            w(f"  ⎿  {red(str(e))}\n")
            return f"Error: {e}"
        model_out, shown = out if isinstance(out, tuple) else (out, None)
        self.show_result(shown if shown is not None else model_out)
        return truncate(model_out)

    @staticmethod
    def arg_summary(name: str, a: dict) -> str:
        key = {"bash": "command", "glob": "pattern", "grep": "pattern"}.get(name, "path")
        s = str(a.get(key, ""))
        if name == "grep" and a.get("path"):
            s += f" in {a['path']}"
        return clip(s.replace("\n", " ⏎ "), term_width() - 20)

    @staticmethod
    def show_result(text: str, max_lines: int = 4) -> None:
        lines = text.rstrip("\n").splitlines() or ["(no output)"]
        width = term_width() - 8
        for i, line in enumerate(lines[:max_lines]):
            w(("  ⎿  " if i == 0 else "     ") + dim(clip(line, width)) + "\n")
        if len(lines) > max_lines:
            w("     " + dim(f"… +{len(lines) - max_lines} lines") + "\n")

    # ── permissions ──
    def ask(self, kind: str, title: str, preview: list) -> None:
        bar = orange("  │ ")
        if self.yolo or kind in self.always:
            for line in preview[:20]:
                w(bar + line + "\n")
            return
        if not sys.stdin.isatty():
            raise ToolDenied("Permission denied: not interactive (run nanocoder with --yolo to auto-approve).")
        w(orange("  ╭─ ") + bold(title) + "\n")
        for line in preview:
            w(bar + line + "\n")
        w(orange("  ╰─ ") + "Allow? " + bold("[y]") + "es · " + bold("[a]") + f"lways for {kind} · "
          + bold("[n]") + "o · or type what to do instead\n")
        try:
            ans = input(prompt_str("  ❯ ")).strip()
        except (KeyboardInterrupt, EOFError):
            w("\n")
            raise KeyboardInterrupt from None
        a = ans.lower()
        if a in ("", "y", "yes", "e", "evet"):
            return
        if a in ("a", "always"):
            self.always.add(kind)
            return
        if a in ("n", "no", "h", "hayir", "hayır"):
            self.stop_turn = True
            raise ToolDenied("The user denied this action. Stop and wait for their instructions.")
        raise ToolDenied(f"The user denied this action and said: {ans}")

    # ── tool implementations ──
    def t_read_file(self, path, offset=1, limit=500):
        p = self.resolve(path)
        if os.path.isdir(p):
            raise ToolError(f"{path} is a directory; use glob to list it.")
        if not os.path.isfile(p):
            raise ToolError(f"File not found: {path}")
        with open(p, "rb") as fh:
            data = fh.read()
        if b"\0" in data[:8192]:
            raise ToolError(f"{path} looks like a binary file.")
        lines = data.decode("utf-8", "replace").splitlines()
        offset, limit = max(1, int(offset or 1)), max(1, int(limit or 500))
        chunk = lines[offset - 1: offset - 1 + limit]
        out = "\n".join(f"{i:>5}\t{line[:2000]}" for i, line in enumerate(chunk, offset)) or "(empty file)"
        if offset - 1 + limit < len(lines):
            out += f"\n... ({len(lines)} lines total, use offset={offset + limit} to read more)"
        return out, f"Read {len(chunk)} lines"

    def t_write_file(self, path, content):
        p = self.resolve(path)
        exists = os.path.isfile(p)
        if exists:
            with open(p, encoding="utf-8", errors="replace") as fh:
                preview = diff_lines(fh.read(), content) or [dim("(no changes)")]
        else:
            lines = content.splitlines()
            preview = [green("+ " + clip(line, term_width() - 10)) for line in lines[:20]]
            if len(lines) > 20:
                preview.append(dim(f"… +{len(lines) - 20} more lines"))
        self.ask("Write", f"{'Overwrite' if exists else 'Create'} {self.rel(p)}", preview)
        os.makedirs(os.path.dirname(p) or ".", exist_ok=True)
        with open(p, "w", encoding="utf-8") as fh:
            fh.write(content)
        n = len(content.splitlines())
        return f"Wrote {n} lines to {self.rel(p)}"

    def t_edit_file(self, path, old_string, new_string, replace_all=False):
        p = self.resolve(path)
        if not os.path.isfile(p):
            raise ToolError(f"File not found: {path} (use write_file to create it).")
        with open(p, encoding="utf-8", errors="replace") as fh:
            text = fh.read()
        if not old_string:
            raise ToolError("old_string is empty. Use write_file to create or rewrite files.")
        if old_string == new_string:
            raise ToolError("old_string and new_string are identical.")
        count = text.count(old_string)
        if count == 0:
            fixed = self._loose_find(text, old_string)
            if fixed is None:
                raise ToolError("old_string was not found in the file." + self._closest(text, old_string))
            old_string, count = fixed, text.count(fixed)
        if count > 1 and not replace_all:
            raise ToolError(f"old_string matches {count} places. Include more surrounding lines to make it "
                            "unique, or set replace_all=true.")
        new_text = text.replace(old_string, new_string) if replace_all else text.replace(old_string, new_string, 1)
        self.ask("Edit", f"Edit {self.rel(p)}", diff_lines(text, new_text))
        with open(p, "w", encoding="utf-8") as fh:
            fh.write(new_text)
        n = count if replace_all else 1
        return f"Edited {self.rel(p)} ({n} replacement{'s' if n > 1 else ''})"

    @staticmethod
    def _loose_find(text: str, old: str):
        """Find old_string ignoring trailing whitespace / CRLF differences. Returns the real snippet."""
        tl, ol = text.splitlines(keepends=True), old.strip("\n").splitlines()
        if not ol:
            return None
        target = [x.rstrip() for x in ol]
        hits = [i for i in range(len(tl) - len(ol) + 1)
                if [x.rstrip() for x in tl[i:i + len(ol)]] == target]
        if len(hits) != 1:
            return None
        block = "".join(tl[hits[0]:hits[0] + len(ol)])
        if not old.endswith("\n"):
            block = block[:-2] if block.endswith("\r\n") else block.rstrip("\n") if block.endswith("\n") else block
        return block

    @staticmethod
    def _closest(text: str, old: str) -> str:
        tl, n = text.splitlines(), max(1, len(old.strip("\n").splitlines()))
        if len(tl) > 5000:
            return ""
        best, best_i = 0.0, -1
        for i in range(max(1, len(tl) - n + 1)):
            r = difflib.SequenceMatcher(None, "\n".join(tl[i:i + n]), old).ratio()
            if r > best:
                best, best_i = r, i
        if best < 0.5:
            return " Read the file again and copy the exact text."
        snippet = "\n".join(tl[best_i:best_i + n])
        return f" Closest match (lines {best_i + 1}-{best_i + n}, {best:.0%} similar):\n{snippet}"

    def t_bash(self, command, timeout=120):
        self.ask("Bash", "Run command", [cyan(clip(line, term_width() - 8)) for line in command.splitlines()])
        self.spinner.start("Running")
        try:
            r = subprocess.run(command, shell=True, cwd=self.cwd, capture_output=True, text=True,
                               errors="replace", timeout=int(timeout or 120), stdin=subprocess.DEVNULL)
        except subprocess.TimeoutExpired:
            raise ToolError(f"Command timed out after {timeout}s.") from None
        finally:
            self.spinner.stop()
        out = (r.stdout or "") + (r.stderr or "")
        if r.returncode:
            out += f"\n[exit code {r.returncode}]"
        return out.strip() or "(no output)"

    def _walk(self, base: str):
        for root, dirs, files in os.walk(base):
            dirs[:] = sorted(d for d in dirs if d not in IGNORED_DIRS)
            for f in files:
                yield os.path.join(root, f)

    def t_glob(self, pattern, path="."):
        base = self.resolve(path or ".")
        pats = [pattern] + ([pattern[3:]] if pattern.startswith("**/") else [])
        hits = []
        for full in self._walk(base):
            rel = os.path.relpath(full, base).replace(os.sep, "/")
            target = rel if "/" in pattern else os.path.basename(rel)
            if any(fnmatch.fnmatch(target, pat) for pat in pats):
                hits.append(full)
        hits.sort(key=lambda f: os.path.getmtime(f), reverse=True)
        out = "\n".join(self.rel(h) for h in hits[:200])
        if len(hits) > 200:
            out += f"\n... ({len(hits)} files total, showing 200 newest)"
        return out or "No files found.", f"Found {len(hits)} file{'s' if len(hits) != 1 else ''}"

    def t_grep(self, pattern, path=".", glob=None, ignore_case=False):
        base = self.resolve(path or ".")
        rg = shutil.which("rg")
        if rg:
            cmd = [rg, "-n", "--no-heading", "--color=never", "--max-columns", "300"]
            cmd += (["-i"] if ignore_case else []) + (["-g", glob] if glob else []) + ["-e", pattern, base]
            r = subprocess.run(cmd, capture_output=True, text=True, errors="replace", cwd=self.cwd)
            if r.returncode > 1:
                raise ToolError(r.stderr.strip())
            lines = [self.rel(line) if line.startswith(base) else line for line in r.stdout.splitlines()]
            lines = [line.replace(self.cwd + os.sep, "") for line in lines]
        else:
            rx = re.compile(pattern, re.I if ignore_case else 0)
            files = [base] if os.path.isfile(base) else self._walk(base)
            lines = []
            for f in files:
                if glob and not fnmatch.fnmatch(os.path.basename(f), glob):
                    continue
                try:
                    if os.path.getsize(f) > 2_000_000:
                        continue
                    with open(f, encoding="utf-8") as fh:
                        for no, line in enumerate(fh, 1):
                            if rx.search(line):
                                lines.append(f"{self.rel(f)}:{no}:{line.rstrip()[:300]}")
                except (UnicodeDecodeError, OSError):
                    continue
        out = "\n".join(lines[:200])
        if len(lines) > 200:
            out += f"\n... ({len(lines)} matches total, showing 200; narrow the search)"
        return out or "No matches.", f"Found {len(lines)} match{'es' if len(lines) != 1 else ''}"

    # ── extras ──
    def expand_mentions(self, text: str) -> str:
        files = []
        for m in re.finditer(r"(?<!\S)@([\w./~\-]+)", text):
            p = self.resolve(m.group(1))
            if os.path.isfile(p) and os.path.getsize(p) < 200_000:
                with open(p, encoding="utf-8", errors="replace") as fh:
                    files.append(f'<file path="{self.rel(p)}">\n{fh.read()}\n</file>')
                w(dim(f"  + attached {self.rel(p)}\n"))
        return text + ("\n\n" + "\n\n".join(files) if files else "")

    def shell(self, cmd: str) -> None:
        r = subprocess.run(cmd, shell=True, cwd=self.cwd, capture_output=True, text=True, errors="replace")
        out = ((r.stdout or "") + (r.stderr or "")).rstrip()
        if out:
            w(out + "\n")
        if r.returncode:
            w(red(f"[exit code {r.returncode}]\n"))
        self.pending_ctx += f"I ran `{cmd}` in the shell:\n```\n{truncate(out, 8000)}\n```\n"

    def compact(self) -> None:
        if len(self.messages) < 3:
            return
        lines = []
        for m in self.messages[1:]:
            body = m.get("content") or ""
            for tc in m.get("tool_calls") or []:
                body += f"\n[tool call {tc['function']['name']} {tc['function']['arguments'][:300]}]"
            lines.append(f"{m['role'].upper()}: {body[:2000]}")
        req = [{"role": "system", "content": "You compress coding sessions into concise notes."},
               {"role": "user", "content": "Summarize this session so the work can continue seamlessly: "
                "the user's goals, key decisions, files touched, current state and next steps. "
                "Be concise, use bullet points.\n\n" + "\n\n".join(lines)}]
        self.spinner.start("Compacting conversation")
        try:
            summary, _ = self.llm.chat(req, None, lambda *_: None)
        finally:
            self.spinner.stop()
        summary = THINK_RE.sub("", summary).strip()
        self.messages = [{"role": "system", "content": self.system_prompt()},
                         {"role": "user", "content": "[Summary of our conversation so far]\n" + summary},
                         {"role": "assistant", "content": "Got it, I'll continue from there."}]
        w(dim(f"✻ Conversation compacted (~{self.est_tokens()} tokens)\n"))


# ───────────────────────────────── REPL ────────────────────────────────────

HELP = """\
  /help          show this help
  /clear         start a fresh conversation
  /compact       summarize the conversation to free up context
  /init          analyze the project and write NANOCODER.md notes
  /model [name]  show or switch the model
  /models        list models available on the server
  /yolo          toggle auto-approve for edits and commands
  /think         toggle showing the model's reasoning
  /tokens        show context usage
  /exit          quit (or ctrl+d)
  !<command>     run a shell command; its output goes into context
  @path          attach a file to your message
  line ending with \\  continue on the next line"""
COMMANDS = ["/help", "/clear", "/compact", "/init", "/model", "/models", "/yolo", "/think", "/tokens", "/exit"]
INIT_PROMPT = ("Explore this project (structure, languages, build/test commands, conventions) and write a "
               "concise NANOCODER.md at the project root that will help a coding agent work here. "
               "If it already exists, improve it.")


def prompt_str(s: str) -> str:
    if not COLOR:
        return s
    if readline:  # \001 \002 tell readline these bytes take no width
        return f"\001{ACCENT}\002{s}\001\x1b[0m\002"
    return f"{ACCENT}{s}\x1b[0m"


def banner(agent: Agent) -> None:
    rows = [f"{orange('✻')} {bold('nanocoder')} {dim('v' + VERSION)}", "",
            f"  {dim('model')}  {agent.llm.model}", f"  {dim('server')} {agent.llm.base_url}",
            f"  {dim('cwd')}    {agent.cwd}", "",
            dim("  /help for commands · @file to attach · !cmd for shell")]
    inner = min(term_width() - 4, max(vlen(r) for r in rows) + 2)
    w(orange("╭" + "─" * (inner + 2) + "╮") + "\n")
    for r in rows:
        if vlen(r) > inner:
            r = clip(ANSI_RE.sub("", r), inner)
        w(orange("│ ") + r + " " * (inner - vlen(r)) + orange(" │") + "\n")
    w(orange("╰" + "─" * (inner + 2) + "╯") + "\n")


def check_server(agent: Agent) -> None:
    try:
        names = agent.llm.models()
    except LLMError as e:
        w(yellow(f"  ! {e}\n") + dim("    Is your model server running? For Ollama: `ollama serve` then "
                                     f"`ollama pull {agent.llm.model}`\n"))
        return
    m = agent.llm.model
    if names and m not in names and f"{m}:latest" not in names:
        w(yellow(f"  ! model '{m}' not found on the server.") + dim(f" Available: {', '.join(names[:8]) or '-'}\n"))


def read_input(agent: Agent) -> str:
    status = f" {agent.llm.model} · ~{agent.est_tokens() / 1000:.1f}k/{agent.ctx_limit // 1000}k tokens"
    status += " · yolo " if agent.yolo else " "
    ensure_nl()
    w("\n" + dim("──" + status + "─" * max(0, term_width() - len(status) - 2)) + "\n")
    lines = [input(prompt_str("❯ "))]
    while lines[-1].endswith("\\"):
        lines[-1] = lines[-1][:-1]
        lines.append(input(prompt_str("… ")))
    return "\n".join(lines)


def command(agent: Agent, line: str) -> bool:
    """Handle a slash command. Returns False to quit."""
    cmd, _, arg = line.strip().partition(" ")
    arg = arg.strip()
    if cmd in ("/exit", "/quit", "/q"):
        return False
    if cmd == "/help":
        w(HELP + "\n")
    elif cmd == "/clear":
        agent.reset()
        w(dim("✻ Fresh start. Previous conversation forgotten.\n"))
    elif cmd == "/compact":
        try:
            agent.compact()
        except (LLMError, KeyboardInterrupt) as e:
            agent.spinner.stop()
            w(red(f"  ✗ compaction failed: {e}\n"))
    elif cmd == "/init":
        agent.turn(INIT_PROMPT)
    elif cmd == "/model":
        if arg:
            agent.llm.model = arg
            w(dim(f"✻ Model set to {arg}\n"))
        else:
            w(f"  {agent.llm.model}\n")
    elif cmd == "/models":
        try:
            for n in agent.llm.models():
                w(("  ● " if n == agent.llm.model else "    ") + n + "\n")
        except LLMError as e:
            w(red(f"  ✗ {e}\n"))
    elif cmd == "/yolo":
        agent.yolo = not agent.yolo
        w(dim(f"✻ Auto-approve {'ON — no questions asked, good luck' if agent.yolo else 'OFF'}\n"))
    elif cmd == "/think":
        agent.show_think = not agent.show_think
        w(dim(f"✻ Showing reasoning: {'on' if agent.show_think else 'off'}\n"))
    elif cmd == "/tokens":
        u = agent.llm.usage or {}
        w(f"  ~{agent.est_tokens()} tokens in context (limit {agent.ctx_limit}), "
          f"{len(agent.messages)} messages")
        w(dim(f" · last call: {u.get('prompt_tokens', '?')} in / {u.get('completion_tokens', '?')} out\n"))
    else:
        w(yellow(f"  unknown command {cmd} — try /help\n"))
    return True


def setup_readline() -> None:
    if not readline:
        return
    hist = os.path.expanduser("~/.nanocoder_history")
    try:
        readline.read_history_file(hist)
    except OSError:
        pass
    readline.set_history_length(1000)
    atexit.register(lambda: _save_history(hist))

    def complete(text, state):
        opts = [c for c in COMMANDS if c.startswith(text)] if text.startswith("/") else []
        return opts[state] if state < len(opts) else None

    readline.set_completer(complete)
    readline.set_completer_delims(" \t\n")
    if "libedit" in (readline.__doc__ or ""):
        readline.parse_and_bind("bind ^I rl_complete")
    else:
        readline.parse_and_bind("tab: complete")


def _save_history(path: str) -> None:
    try:
        readline.write_history_file(path)
    except OSError:
        pass


def main(argv=None) -> int:
    env = os.environ.get
    ap = argparse.ArgumentParser(prog="nanocoder", description="A tiny local coding agent for your terminal.")
    ap.add_argument("prompt", nargs="*", help="run one prompt non-interactively and exit")
    ap.add_argument("-m", "--model", default=env("NANOCODER_MODEL", DEFAULT_MODEL))
    ap.add_argument("--base-url", default=env("NANOCODER_BASE_URL", DEFAULT_BASE_URL),
                    help="OpenAI-compatible API base URL (default: Ollama)")
    ap.add_argument("--api-key", default=env("NANOCODER_API_KEY", ""))
    ap.add_argument("-y", "--yolo", action="store_true", help="auto-approve edits and commands")
    ap.add_argument("--text-tools", action="store_true",
                    help="describe tools in the prompt (for models without native tool calling)")
    ap.add_argument("--ctx", type=int, default=int(env("NANOCODER_CTX", "32768")),
                    help="context window size in tokens, used for auto-compaction")
    ap.add_argument("-C", "--cwd", default=os.getcwd(), help="project directory")
    ap.add_argument("--version", action="version", version=f"nanocoder {VERSION}")
    args = ap.parse_args(argv)

    agent = Agent(LLM(args.base_url, args.model, args.api_key), args.cwd,
                  yolo=args.yolo, text_tools=args.text_tools, ctx_limit=args.ctx)

    prompt = " ".join(args.prompt)
    if not prompt and not sys.stdin.isatty():
        prompt = sys.stdin.read().strip()
    if prompt:
        agent.turn(prompt)
        return 0

    setup_readline()
    banner(agent)
    check_server(agent)
    while True:
        try:
            line = read_input(agent)
        except EOFError:
            w(dim("\nbye 👋\n"))
            return 0
        except KeyboardInterrupt:
            w(dim("\n  (ctrl+d or /exit to quit)\n"))
            continue
        if not line.strip():
            continue
        if line.startswith("/"):
            if not command(agent, line):
                w(dim("bye 👋\n"))
                return 0
        elif line.startswith("!"):
            agent.shell(line[1:].strip())
        else:
            agent.turn(line)


if __name__ == "__main__":
    sys.exit(main())
