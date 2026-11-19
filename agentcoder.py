#!/usr/bin/env python3
"""AgentCoder - a tiny, local, terminal coding agent.

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
# local servers probed by --base-url auto (name, url); override with AGENTCODER_BACKENDS="name=url,..."
KNOWN_BACKENDS = [("agentcoder", "http://127.0.0.1:8090/v1"), ("ollama", "http://127.0.0.1:11434/v1"), ("llama.cpp", "http://127.0.0.1:8080/v1"),
                  ("llama.cpp", "http://127.0.0.1:8081/v1"), ("llama.cpp", "http://127.0.0.1:8082/v1"),
                  ("llama.cpp", "http://127.0.0.1:8083/v1"), ("vllm", "http://127.0.0.1:8000/v1"),
                  ("sglang", "http://127.0.0.1:30000/v1"), ("lmstudio", "http://127.0.0.1:1234/v1")]
BENCH_CACHE = os.path.expanduser("~/.agentcoder/bench.json")
IGNORED_DIRS = {".git", "node_modules", "__pycache__", ".venv", "venv", "env", "dist", "build",
                ".next", ".mypy_cache", ".pytest_cache", ".idea", ".vscode", "target", ".tox"}
PROJECT_NOTES = ("AGENTCODER.md", "AGENTS.md", "CLAUDE.md")
MAX_TOOL_OUTPUT = 20_000
MAX_STEPS = 40
SUB_MAX_STEPS = 20
DEFAULT_WORKERS = "small=qwen2.5-coder:0.5b,medium=qwen2.5-coder:1.5b,large=qwen2.5-coder:3b"
READONLY_TIERS = {"small"}
DEFAULT_REFINER = "qwen3.5:9b"
REFINER_PROMPT = """You are the prompt refiner in a team of coding agents. You never write code and never talk to \
the user. You turn the user's raw message into a clear brief for the orchestrator agent that will do the work.

Write the brief in English with exactly these sections, even when the user writes in another language (the \
orchestrator and its sub-agents work in English):
REQUEST (English): the user's message translated faithfully into English. Keep code, identifiers, file names, \
paths, quoted strings and error messages exactly as written. If the user already wrote English, copy it.
GOAL: one or two sentences: what the user actually wants.
DETAILS: bullet list of concrete requirements, file names and constraints. Mark your guesses with "(assumed)".
DO NOT: bullet list of things the orchestrator must avoid for THIS request, e.g. scope creep, touching unrelated \
files, deleting data, adding dependencies, changing public APIs, rewriting working code, asking needless questions.
DONE WHEN: how to tell the task is finished (tests pass, command output, file exists...).
REPLY LANGUAGE: the language the user wrote in (e.g. Turkish). The orchestrator answers the user in it.

Rules:
- Keep the user's intent. Clarify, never invent new requirements.
- If the message is a short follow-up, confirmation or small talk ("yes", "continue", "thanks"), output exactly: PASS
- Under 220 words. No preamble."""
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
        self.max_tokens = None
        self.backend = ""     # which local server was picked (auto mode)
        self.pinned = False   # True when the user gave an explicit URL for this model

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
        if self.max_tokens:
            body["max_tokens"] = self.max_tokens
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
           "bash": "Bash", "glob": "Glob", "grep": "Grep", "task": "Task"}
READ_ONLY_TOOLS = {"read_file", "glob", "grep"}


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
        if a.role:  # sub-agent: no streaming text, just spinner updates
            if mode == "tool":
                a.spinner.label = f"{a.role} agent · preparing {DISPLAY.get(text, text)}"
            elif mode == "think":
                a.spinner.label = f"{a.role} agent · thinking"
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
    def __init__(self, llm: LLM, cwd: str, yolo=False, text_tools=False, ctx_limit=32768,
                 workers=None, role=None, parent=None, refiner=None):
        self.llm, self.cwd = llm, os.path.abspath(cwd)
        self.yolo, self.text_tools, self.ctx_limit = yolo, text_tools, ctx_limit
        self.role = role                      # None = orchestrator, else the worker tier name
        self.root = parent.root if parent else self
        self.depth = parent.depth + 1 if parent else 0
        self.ind = "    " * self.depth
        self.workers = {} if role else dict(workers or {})   # tier -> LLM, orchestrator only
        self.refiner = None if role else refiner              # LLM that rewrites user prompts
        self.show_think = False
        self.always: set = parent.root.always if parent else set()
        self.spinner = parent.spinner if parent else Spinner()
        self.pending_ctx = ""
        self.stop_turn = False
        self.tools = {"read_file": self.t_read_file, "write_file": self.t_write_file,
                      "edit_file": self.t_edit_file, "bash": self.t_bash,
                      "glob": self.t_glob, "grep": self.t_grep}
        if role in READONLY_TIERS:
            self.tools = {k: v for k, v in self.tools.items() if k in READ_ONLY_TOOLS}
        if self.workers:
            self.tools["task"] = self.t_task
        self.reset()

    def schemas(self) -> list:
        out = [t for t in TOOLS if t["function"]["name"] in self.tools]
        if "task" in self.tools:
            team = "; ".join(f"{t} = {llm.model}{' (read-only)' if t in READONLY_TIERS else ''}"
                             for t, llm in self.workers.items())
            out.append(_tool(
                "task", "Delegate a self-contained subtask to a smaller, faster sub-agent model. It cannot "
                "see this conversation: give complete, specific instructions (file paths, goal, what to "
                f"report). Returns the sub-agent's final report. Team: {team}.",
                {"agent": {"type": "string", "enum": list(self.workers), "description": "Which sub-agent."},
                 "task": _s("Full instructions for the sub-agent.")}, ["agent", "task"]))
        return out

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
        p = f"""You are AgentCoder, a small but sharp coding agent running in the user's terminal. \
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
        if self.workers:
            team = "\n".join(f"- {t} ({llm.model}): " + (
                "read-only. Codebase searches, finding definitions/usages, reading and summarizing files."
                if t in READONLY_TIERS else "can read, edit and run commands. Simple well-specified edits, "
                "writing tests, running tests and reporting results.") for t, llm in self.workers.items())
            p += f"""

# Your team
You are the orchestrator. With the `task` tool you can delegate subtasks to smaller, faster models:
{team}
- Sub-agents do NOT see this conversation. Give each one complete, specific instructions and say what to report.
- They are weaker than you. Keep hard reasoning and tricky edits for yourself; check their reports before relying on them.
- Delegate when it saves work (broad searches, reading many files, routine edits). For one small file, just do it yourself.
- Write `task` descriptions in English, whatever language the user speaks: the sub-agents were trained on English. \
Keep file names, identifiers and quoted text exactly as the user wrote them."""
        if self.refiner:
            p += """

# Briefs
User messages may end with a <brief> written by a refiner model that clarified the request and translated it into \
English. Follow its DO NOT rules and DONE WHEN criteria, and reply to the user in its REPLY LANGUAGE. If the brief \
conflicts with the user's own words, the user wins."""
        if self.role:
            p += f"""

# Your role
You are a sub-agent ("{self.role}") working for an orchestrator model. It cannot see your tool calls or their output, \
only your final reply. Do the task with your tools, then reply with a concise report: what you found or changed, \
with file paths and line numbers.{" You are read-only: you cannot edit files or run commands." if self.role in READONLY_TIERS else ""}"""
        if self.text_tools:
            p += "\n\n# Tools\nCall a tool by writing exactly:\n<tool_call>\n" \
                 '{"name": "tool_name", "arguments": {"arg": "value"}}\n</tool_call>\n' \
                 "You may emit several tool calls in one reply. Then stop and wait: results come back " \
                 "as <tool_result> messages. When the task is done, reply normally without tool calls.\n" \
                 "Available tools:\n"
            for t in self.schemas():
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
        raw = text
        text = self.expand_mentions(text)
        if self.pending_ctx:
            text, self.pending_ctx = self.pending_ctx + "\n\n" + text, ""
        try:
            if self.est_tokens() > self.ctx_limit * 0.8:
                self.compact()
            brief = self.refine(raw)
            if brief:
                text += f"\n\n<brief>\n{brief}\n</brief>"
            self.messages.append({"role": "user", "content": text})
            if not self.run_loop(MAX_STEPS):
                w(yellow(f"  (stopped after {MAX_STEPS} steps — say 'continue' to keep going)\n"))
        except KeyboardInterrupt:
            self.spinner.stop()
            self.repair_history()
            ensure_nl()
            w(red("  ⎿  Interrupted") + dim(" · tell AgentCoder what to do instead\n"))
        except LLMError as e:
            self.spinner.stop()
            self.repair_history()
            ensure_nl()
            w(red(f"  ✗ {e}\n"))

    def refine(self, raw: str) -> str:
        """Ask the refiner model to turn a raw user message into a brief. '' = no brief."""
        if not self.refiner or len(raw.strip()) < 12:
            return ""
        recent = []
        for m in self.messages[1:]:
            if m["role"] == "user" and not m["content"].startswith("<tool_result"):
                recent.append("USER: " + m["content"].split("\n\n<brief>")[0][:600])
            elif m["role"] == "assistant" and m.get("content"):
                recent.append("AGENT: " + TOOL_CALL_RE.sub("", m["content"])[:600])
        entries = ", ".join(sorted(e for e in os.listdir(self.cwd) if e not in IGNORED_DIRS)[:40])
        ctx = f"Project: {self.cwd}\nTop-level entries: {entries or '(empty)'}\n"
        if recent:
            ctx += "Recent conversation:\n" + "\n".join(recent[-6:]) + "\n"
        req = [{"role": "system", "content": REFINER_PROMPT},
               {"role": "user", "content": f"{ctx}\nUser's new message:\n{raw}"}]
        self.spinner.start("Refining prompt")
        try:
            out, _ = self.refiner.chat(req, None, lambda *_: None)
        except LLMError as e:
            self.spinner.stop()
            w(yellow(f"  ! refiner failed, continuing without brief: {clip(str(e), 100)}\n"))
            return ""
        finally:
            self.spinner.stop()
        brief = THINK_RE.sub("", out).strip()
        if not brief or brief.upper().startswith("PASS"):
            return ""
        ensure_nl()
        w(f"{cyan('●')} {bold('Brief')}{dim('(' + self.refiner.model + ')')}\n")
        lines = brief.splitlines()
        for i, line in enumerate(lines[:24]):
            w(("  ⎿  " if i == 0 else "     ") + dim(clip(line, term_width() - 8)) + "\n")
        if len(lines) > 24:
            w("     " + dim(f"… +{len(lines) - 24} lines") + "\n")
        return brief

    def run_loop(self, max_steps: int) -> bool:
        """Run model/tool steps until the model stops calling tools. False if max_steps was hit."""
        self.stop_turn = False
        for _ in range(max_steps):
            if not self.step() or self.stop_turn:
                return True
        return False

    def step(self) -> bool:
        """One model call + tool execution. Returns True if tools were called."""
        stream = Stream(self)
        verb = random.choice(VERBS)
        self.spinner.start(f"{self.role} agent · {verb.lower()}" if self.role else verb)
        try:
            content, calls = self.llm.chat(self.messages, None if self.text_tools else self.schemas(), stream)
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
            w(f"{self.ind}{red('●')} {bold(name or '?')}\n  ⎿  {red('invalid tool arguments')}\n")
            return f"Error: could not parse tool arguments as a JSON object ({e}). Retry with valid JSON."
        ensure_nl()
        w(f"{self.ind}{green('●')} {bold(DISPLAY.get(name, name))}{dim('(' + self.arg_summary(name, args) + ')')}\n")
        fn = self.tools.get(name)
        if not fn:
            w(f"{self.ind}  ⎿  {red('unknown tool')}\n")
            return f"Error: unknown tool '{name}'. Available: {', '.join(self.tools)}."
        try:
            out = fn(**args)
        except ToolDenied as e:
            said = str(e).partition("and said: ")[2]
            w(f"{self.ind}  ⎿  {yellow('Denied' + (': ' + clip(said, term_width() - 20) if said else ''))}\n")
            return str(e)
        except ToolError as e:
            w(f"{self.ind}  ⎿  {red(clip(str(e).splitlines()[0], term_width() - 8))}\n")
            return f"Error: {e}"
        except TypeError as e:
            w(f"{self.ind}  ⎿  {red('bad arguments: ' + str(e))}\n")
            return f"Error: bad arguments for {name}: {e}"
        except (OSError, re.error) as e:
            w(f"{self.ind}  ⎿  {red(str(e))}\n")
            return f"Error: {e}"
        model_out, shown = out if isinstance(out, tuple) else (out, None)
        self.show_result(shown if shown is not None else model_out)
        return truncate(model_out)

    @staticmethod
    def arg_summary(name: str, a: dict) -> str:
        key = {"bash": "command", "glob": "pattern", "grep": "pattern", "task": "task"}.get(name, "path")
        s = str(a.get(key, ""))
        if name == "task":
            s = f"{a.get('agent', '?')}: {s}"
        if name == "grep" and a.get("path"):
            s += f" in {a['path']}"
        return clip(s.replace("\n", " ⏎ "), term_width() - 20)

    def show_result(self, text: str, max_lines: int = 4) -> None:
        lines = text.rstrip("\n").splitlines() or ["(no output)"]
        width = term_width() - 8
        for i, line in enumerate(lines[:max_lines]):
            w(self.ind + ("  ⎿  " if i == 0 else "     ") + dim(clip(line, width)) + "\n")
        if len(lines) > max_lines:
            w(self.ind + "     " + dim(f"… +{len(lines) - max_lines} lines") + "\n")

    # ── permissions ──
    def ask(self, kind: str, title: str, preview: list) -> None:
        bar = self.ind + orange("  │ ")
        if self.root.yolo or kind in self.always:
            for line in preview[:20]:
                w(bar + line + "\n")
            return
        if not sys.stdin.isatty():
            raise ToolDenied("Permission denied: not interactive (run agentcoder with --yolo to auto-approve).")
        if self.role:
            title = f"[{self.role} agent] {title}"
        w(self.ind + orange("  ╭─ ") + bold(title) + "\n")
        for line in preview:
            w(bar + line + "\n")
        w(self.ind + orange("  ╰─ ") + "Allow? " + bold("[y]") + "es · " + bold("[a]") + f"lways for {kind} · "
          + bold("[n]") + "o · or type what to do instead\n")
        try:
            ans = input(prompt_str(self.ind + "  ❯ ")).strip()
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
            self.stop_turn = self.root.stop_turn = True
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

    def t_task(self, agent, task):
        llm = self.workers.get(agent)
        if not llm:
            raise ToolError(f"No sub-agent '{agent}'. Available: {', '.join(self.workers) or 'none'}.")
        sub = Agent(llm, self.cwd, ctx_limit=self.ctx_limit, role=agent, parent=self)
        sub.messages.append({"role": "user", "content": task})
        t0 = time.time()
        try:
            finished = sub.run_loop(SUB_MAX_STEPS)
        except LLMError as e:
            raise ToolError(f"{agent} agent failed: {e}") from None
        report = next((m.get("content") or "" for m in reversed(sub.messages) if m["role"] == "assistant"), "")
        report = TOOL_CALL_RE.sub("", report).strip() or "(the sub-agent returned no report)"
        n_calls = sum(len(m.get("tool_calls") or []) for m in sub.messages) or \
            sum(len(parse_text_calls(m.get("content") or "")) for m in sub.messages if m["role"] == "assistant")
        if not finished:
            report += f"\n\n[sub-agent hit its {SUB_MAX_STEPS}-step limit; the work may be incomplete]"
        if sub.stop_turn:
            report += "\n\n[the user stopped this sub-agent]"
        return report, f"{agent} agent ({llm.model}) done · {n_calls} tool call{"s" if n_calls != 1 else ""} · {int(time.time() - t0)}s"

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


# ─────────────────────────── backend auto-selection ─────────────────────────

def canon(name: str) -> str:
    """Normalize model ids across servers: 'qwen2.5-coder:7b', 'Qwen/Qwen2.5-Coder-7B-Instruct' and
    'qwen2.5-coder-7b-instruct-q4_k_m.gguf' all become 'qwen2.5-coder-7b'."""
    s = name.lower().strip().rsplit("/", 1)[-1].replace(":", "-").replace("_", "-")
    s = re.sub(r"\.gguf$", "", s)
    s = re.sub(r"-\d{5}-of-\d{5}$", "", s)
    s = re.sub(r"-(i?q\d[\w-]*|f16|bf16|fp16|fp8|int4|int8|awq|gptq|\d+bit)$", "", s)
    s = re.sub(r"-(instruct|chat|it|latest)(?=-|$)", "", s)
    return s


def discover(timeout: float = 0.5) -> list:
    """Probe local OpenAI-compatible servers in parallel. Returns [{name, url, models}]."""
    raw = os.environ.get("AGENTCODER_BACKENDS")
    specs = [tuple(x.strip().split("=", 1)) for x in raw.split(",") if "=" in x] if raw else KNOWN_BACKENDS
    found = [None] * len(specs)

    def probe(i, name, url):
        try:
            with LLM(url, "")._req("/models", timeout=timeout) as r:
                data = json.load(r).get("data", [])
        except (LLMError, ValueError, OSError):
            return
        owner = next((m.get("owned_by") for m in data if m.get("owned_by")), "")
        name = {"llamacpp": "llama.cpp", "vllm": "vllm", "sglang": "sglang",
                "agentcoder": "agentcoder-server"}.get(owner, name)
        found[i] = {"name": name, "url": url, "models": [m.get("id", "") for m in data if m.get("id")]}

    threads = [threading.Thread(target=probe, args=(i, n, u), daemon=True) for i, (n, u) in enumerate(specs)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout + 1)
    return [f for f in found if f]


BENCH_FILE = "\n".join(f"{i:>5}\tdef handler_{i}(request, retries={i % 5}):\n      \treturn route(request, {i})"
                       for i in range(1, 45))


def bench(url: str, model: str) -> dict:
    """Time a realistic agent exchange, since that is what agentcoder sends all day.

    A cold request with a ~1.2k-token system prompt, then the next turn: the same conversation plus a
    ~700-token tool result. Servers that reuse the prompt prefix (KV cache) shine on the second one.
    score = cold first-token / 4 + next-turn first-token + time for 64 tokens.
    """
    llm = LLM(url, model)
    llm.max_tokens = 1
    llm.chat([{"role": "user", "content": "hi"}], None, lambda *_: None)  # load the model first
    llm.max_tokens = 48
    system = (f"[run {random.random():.6f}] You are a careful coding agent. Follow the project's conventions, "
              "read before you edit, keep diffs small and run the tests. " * 40)
    msgs = [{"role": "system", "content": system},
            {"role": "user", "content": "Find where retries are handled in handlers.py and summarize it."}]

    def timed(messages):
        t0, first, chars = time.time(), [0.0], [0]

        def sink(mode, text):
            first[0] = first[0] or time.time()
            chars[0] += len(text)
        content, _ = llm.chat(messages, None, sink)
        end = time.time()
        start = first[0] or end
        toks = (llm.usage or {}).get("completion_tokens") or max(1, chars[0] // 4)
        return start - t0, toks / max(1e-3, end - start), content

    cold, _, reply = timed(msgs)
    msgs += [{"role": "assistant", "content": reply or "Let me read the file."},
             {"role": "user", "content": "Tool result (read_file handlers.py):\n" + BENCH_FILE + "\nNow summarize."}]
    warm, rate, _ = timed(msgs)
    return {"score": cold / 4 + warm + 64 / rate, "rate": rate, "ttft": warm, "cold": cold, "at": time.time()}


def _load_cache() -> dict:
    try:
        with open(BENCH_CACHE) as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return {}


def _save_cache(cache: dict) -> None:
    try:
        os.makedirs(os.path.dirname(BENCH_CACHE), exist_ok=True)
        with open(BENCH_CACHE, "w") as fh:
            json.dump(cache, fh, indent=1)
    except OSError:
        pass


def pick_backend(model: str, backends: list, cache: dict, rebench: bool = False):
    """Find every server that has `model`; if several, benchmark (cached a week) and pick the fastest."""
    want = canon(model)
    cands = [(b, mid) for b in backends for mid in b["models"] if mid == model or canon(mid) == want]
    if not cands:
        return None
    if len(cands) == 1:
        return cands[0][0], cands[0][1], ""
    scored = []
    for b, mid in cands:
        key = f"{b['url']}|{mid}"
        c = cache.get(key)
        if rebench or not c or "cold" not in c or time.time() - c.get("at", 0) > 7 * 86400:
            w(dim(f"  ✻ benchmarking {model} on {b['name']}…\n"))
            try:
                c = cache[key] = bench(b["url"], mid)
            except LLMError as e:
                w(dim(f"    skipped: {clip(str(e), 80)}\n"))
                continue
        scored.append((c["score"], b, mid, c))
    if not scored:
        return cands[0][0], cands[0][1], ""
    _, b, mid, c = min(scored, key=lambda x: x[0])
    return b, mid, f"fastest of {len(scored)}, {c['rate']:.0f} tok/s"


def auto_setup(agent: Agent, rebench: bool = False) -> None:
    """--base-url auto: route every role (refiner, orchestrator, workers) to the fastest local server."""
    backends = discover()
    if not backends:
        w(yellow("  ! no local model server found.") + dim(" Start one, e.g. `ollama serve` or "
                                                         "`llama-server -m model.gguf --jinja`\n"))
        return
    cache = _load_cache()

    def route(llm: LLM) -> bool:
        if llm.pinned:
            return True
        hit = pick_backend(llm.model, backends, cache, rebench)
        if not hit:
            return False
        b, mid, why = hit
        llm.base_url, llm.model = b["url"], mid
        llm.backend = b["name"] + (f" · {why}" if why else "")
        return True

    if not route(agent.llm):
        have = sorted({m for b in backends for m in b["models"]})
        w(yellow(f"  ! model '{agent.llm.model}' is not on any local server.")
          + dim(f" Found: {', '.join(have[:8]) or '-'}\n"))
        agent.llm.base_url = backends[0]["url"]
    if agent.refiner and not route(agent.refiner):
        w(dim(f"  ! prompt refiner disabled (not on any server): {agent.refiner.model}\n"))
        agent.refiner = None
    gone = [t for t, llm in agent.workers.items() if not route(llm)]
    if gone:
        w(dim("  ! sub-agents disabled (not on any server): "
              + ", ".join(f"{t}={agent.workers[t].model}" for t in gone) + "\n"))
    agent.workers = {t: llm for t, llm in agent.workers.items() if t not in gone}
    set_workers(agent, agent.workers)
    _save_cache(cache)


# ───────────────────────────────── REPL ────────────────────────────────────

HELP = """\
  /help          show this help
  /clear         start a fresh conversation
  /compact       summarize the conversation to free up context
  /init          analyze the project and write AGENTCODER.md notes
  /model [name]  show or switch the model
  /models        list models available on the server
  /yolo          toggle auto-approve for edits and commands
  /think         toggle showing the model's reasoning
  /tokens        show context usage
  /team          show the sub-agent team
  /refine        toggle the prompt refiner
  /backends      list local model servers and benchmark results
  /exit          quit (or ctrl+d)
  !<command>     run a shell command; its output goes into context
  @path          attach a file to your message
  line ending with \\  continue on the next line"""
COMMANDS = ["/help", "/clear", "/compact", "/init", "/model", "/models", "/yolo", "/think", "/tokens", "/team", "/refine", "/backends", "/exit"]
INIT_PROMPT = ("Explore this project (structure, languages, build/test commands, conventions) and write a "
               "concise AGENTCODER.md at the project root that will help a coding agent work here. "
               "If it already exists, improve it.")


def prompt_str(s: str) -> str:
    if not COLOR:
        return s
    if readline:  # \001 \002 tell readline these bytes take no width
        return f"\001{ACCENT}\002{s}\001\x1b[0m\002"
    return f"{ACCENT}{s}\x1b[0m"


def banner(agent: Agent) -> None:
    rows = [f"{orange('✻')} {bold('AgentCoder')} {dim('v' + VERSION)}", "",
            f"  {dim('model')}  {agent.llm.model.rsplit('/', 1)[-1]}" + (dim(f"  [{agent.llm.backend}]") if agent.llm.backend else ""),
            f"  {dim('server')} {agent.llm.base_url}",
            f"  {dim('cwd')}    {agent.cwd}"]
    if agent.refiner:
        rows.append(f"  {dim('brief')}  {agent.refiner.model}"
                    + (dim(f"  [{agent.refiner.backend.split(' ·')[0]}]") if agent.refiner.backend else ""))
    if agent.workers:
        rows.append(f"  {dim('team')}   " + " · ".join(
            f"{t} {llm.model}" + (dim(f" [{llm.backend.split(' ·')[0]}]") if llm.backend else "")
            for t, llm in agent.workers.items()))
    rows += ["",
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
    def missing(llm: LLM) -> bool:
        return bool(names) and llm.model not in names and f"{llm.model}:latest" not in names

    if missing(agent.llm):
        w(yellow(f"  ! model '{agent.llm.model}' not found on the server.")
          + dim(f" Available: {', '.join(names[:8]) or '-'}\n"))
    if agent.refiner and agent.refiner.base_url == agent.llm.base_url and missing(agent.refiner):
        w(dim(f"  ! prompt refiner disabled (model not pulled): {agent.refiner.model}\n"))
        agent.refiner = None
        agent.messages[0]["content"] = agent.system_prompt()
    gone = [t for t, llm in agent.workers.items() if llm.base_url == agent.llm.base_url and missing(llm)]
    if gone:
        w(dim("  ! sub-agents disabled (model not pulled): "
              + ", ".join(f"{t}={agent.workers[t].model}" for t in gone) + "\n"))
        set_workers(agent, {t: llm for t, llm in agent.workers.items() if t not in gone})


def set_workers(agent: Agent, workers: dict) -> None:
    agent.workers = workers
    agent.tools.pop("task", None)
    if workers:
        agent.tools["task"] = agent.t_task
    agent.messages[0]["content"] = agent.system_prompt()


def parse_workers(specs: list, base_url: str, api_key: str) -> dict:
    """'small=qwen2.5-coder:0.5b' or 'big=some-model@http://host:1234/v1' -> {tier: LLM}."""
    out = {}
    for spec in specs:
        for item in filter(None, (x.strip() for x in spec.split(","))):
            tier, eq, model = item.partition("=")
            if not eq or not tier.strip() or not model.strip():
                raise SystemExit(f"agentcoder: bad worker spec '{item}', expected TIER=MODEL[@BASE_URL]")
            model, _, url = model.strip().partition("@")
            out[tier.strip()] = LLM(url or base_url, model, api_key)
            out[tier.strip()].pinned = bool(url)
    return out


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
    elif cmd == "/backends":
        cache = _load_cache()
        for b in discover():
            w(f"  {bold(b['name'])} {dim(b['url'])}\n")
            for m in b["models"]:
                c = cache.get(f"{b['url']}|{m}")
                w(f"    {m}" + (dim(f"  {c['rate']:.0f} tok/s, first token {c['ttft']:.1f}s") if c else "") + "\n")
        w(dim("  (rerun agentcoder with --rebench to measure again)\n"))
    elif cmd == "/refine":
        if agent.refiner:
            agent.parked_refiner, agent.refiner = agent.refiner, None
        else:
            agent.refiner = getattr(agent, "parked_refiner", None)
        agent.messages[0]["content"] = agent.system_prompt()
        w(dim(f"✻ Prompt refiner: {agent.refiner.model if agent.refiner else 'off'}\n"))
    elif cmd == "/team":
        if agent.refiner:
            w(f"  {bold('refiner'):<13}  {agent.refiner.model}\n")
        w(f"  {bold('orchestrator')}  {agent.llm.model}\n")
        for t, llm in agent.workers.items():
            w(f"  {t:<13} {llm.model}{dim(' (read-only)') if t in READONLY_TIERS else ''}\n")
        if not agent.workers:
            w(dim("  no sub-agents (configure with --worker TIER=MODEL)\n"))
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
    hist = os.path.expanduser("~/.agentcoder_history")
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
    ap = argparse.ArgumentParser(prog="agentcoder", description="A tiny local coding agent for your terminal.")
    ap.add_argument("prompt", nargs="*", help="run one prompt non-interactively and exit")
    ap.add_argument("-m", "--model", default=env("AGENTCODER_MODEL", DEFAULT_MODEL))
    ap.add_argument("--base-url", default=env("AGENTCODER_BASE_URL", "auto"),
                    help="OpenAI-compatible API base URL, or 'auto' (default) to find local servers "
                         "and use the fastest one for each model")
    ap.add_argument("--rebench", action="store_true", help="re-measure backend speed (auto mode)")
    ap.add_argument("--api-key", default=env("AGENTCODER_API_KEY", ""))
    ap.add_argument("-y", "--yolo", action="store_true", help="auto-approve edits and commands")
    ap.add_argument("--text-tools", action="store_true",
                    help="describe tools in the prompt (for models without native tool calling)")
    ap.add_argument("--ctx", type=int, default=int(env("AGENTCODER_CTX", "32768")),
                    help="context window size in tokens, used for auto-compaction")
    ap.add_argument("-w", "--worker", action="append", metavar="TIER=MODEL",
                    help="sub-agent model, repeatable, e.g. -w small=qwen2.5-coder:0.5b "
                         f"(default: {DEFAULT_WORKERS}; tier 'small' is read-only)")
    ap.add_argument("--no-workers", action="store_true", help="single agent, no sub-agents")
    ap.add_argument("-r", "--refiner", default=env("AGENTCODER_REFINER", DEFAULT_REFINER), metavar="MODEL",
                    help="model that rewrites your prompt into a brief for the orchestrator "
                         "(MODEL or MODEL@BASE_URL)")
    ap.add_argument("--no-refiner", action="store_true", help="send prompts to the orchestrator as-is")
    ap.add_argument("-C", "--cwd", default=os.getcwd(), help="project directory")
    ap.add_argument("--version", action="version", version=f"agentcoder {VERSION}")
    args = ap.parse_args(argv)

    auto = args.base_url == "auto"
    base = DEFAULT_BASE_URL if auto else args.base_url
    specs = [] if args.no_workers else (args.worker or [env("AGENTCODER_WORKERS", DEFAULT_WORKERS)])
    workers = parse_workers(specs, base, args.api_key)
    refiner = None
    if not args.no_refiner and args.refiner:
        model, _, url = args.refiner.partition("@")
        refiner = LLM(url or base, model, args.api_key, temperature=0.1)
        refiner.pinned = bool(url)
    chef = LLM(base, args.model, args.api_key)
    chef.pinned = not auto
    agent = Agent(chef, args.cwd, yolo=args.yolo,
                  text_tools=args.text_tools, ctx_limit=args.ctx, workers=workers, refiner=refiner)

    prompt = " ".join(args.prompt)
    if not prompt and not sys.stdin.isatty():
        prompt = sys.stdin.read().strip()
    def connect():
        auto_setup(agent, args.rebench) if auto else check_server(agent)

    if prompt:
        connect()
        agent.turn(prompt)
        return 0

    setup_readline()
    if auto:
        connect()
        banner(agent)
    else:
        banner(agent)
        connect()
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
