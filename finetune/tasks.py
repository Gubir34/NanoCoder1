"""Synthetic mini-repos + sub-agent tasks used to fine-tune and evaluate nanocoder worker models.

Every task knows its ground truth, so it comes with:
  - a scripted *teacher* that solves it with nanocoder's real tools (used to make training data)
  - a *check* that verifies the repo state / final report (used for data validation and eval)
"""
from __future__ import annotations

import hashlib
import os
import random
import re
import subprocess
import sys

# edits happen within the same second as test runs; stale .pyc files would hide them
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"

# ───────────────────────────── function templates ─────────────────────────────
# body is the function body (unindented); bug = (good fragment, buggy fragment) inside body.
TEMPLATES = [
    dict(key="add", names=["add", "add_numbers", "plus", "sum_two"], args="a, b", body="return a + b",
         bug=("a + b", "a - b"), desc="returns the sum of a and b", tests=[("2, 3", "5"), ("-1, 1", "0")]),
    dict(key="mul", names=["multiply", "mul", "times", "product_of"], args="a, b", body="return a * b",
         bug=("a * b", "a + b"), desc="returns a multiplied by b", tests=[("3, 4", "12"), ("0, 5", "0")]),
    dict(key="sub", names=["subtract", "minus", "diff"], args="a, b", body="return a - b",
         bug=("a - b", "b - a"), desc="returns a minus b", tests=[("5, 3", "2")]),
    dict(key="avg", names=["average", "mean", "avg"], args="xs",
         body="if not xs:\n    return 0\nreturn sum(xs) / len(xs)", bug=("sum(xs) / len(xs)", "sum(xs) / (len(xs) + 1)"),
         desc="returns the arithmetic mean of a list (0 for an empty list)", tests=[("[2, 4, 6]", "4.0"), ("[]", "0")]),
    dict(key="max", names=["maximum", "largest", "max_value"], args="xs",
         body="best = xs[0]\nfor x in xs[1:]:\n    if x > best:\n        best = x\nreturn best",
         bug=("if x > best:", "if x < best:"), desc="returns the largest item of a non-empty list",
         tests=[("[3, 9, 2]", "9"), ("[-5, -1]", "-1")]),
    dict(key="even", names=["is_even", "check_even"], args="n", body="return n % 2 == 0",
         bug=("n % 2 == 0", "n % 2 == 1"), desc="tells whether n is even", tests=[("4", "True"), ("7", "False")]),
    dict(key="rev", names=["reverse_string", "reverse_text", "flip"], args="s", body="return s[::-1]",
         bug=("s[::-1]", "s[::1]"), desc="returns the string reversed", tests=[('"abc"', '"cba"')]),
    dict(key="vowels", names=["count_vowels", "vowel_count"], args="s",
         body='return sum(1 for ch in s.lower() if ch in "aeiou")', bug=('"aeiou"', '"ae"'),
         desc="counts the vowels in a string", tests=[('"hello"', "2"), ('"AEIOU"', "5")]),
    dict(key="clamp", names=["clamp", "bound", "limit_value"], args="x, lo, hi", body="return max(lo, min(x, hi))",
         bug=("max(lo, min(x, hi))", "min(lo, max(x, hi))"), desc="limits x to the range [lo, hi]",
         tests=[("5, 0, 10", "5"), ("-3, 0, 10", "0"), ("15, 0, 10", "10")]),
    dict(key="fact", names=["factorial", "fact"], args="n",
         body="result = 1\nfor i in range(2, n + 1):\n    result *= i\nreturn result",
         bug=("range(2, n + 1)", "range(2, n)"), desc="returns n factorial", tests=[("5", "120"), ("0", "1")]),
    dict(key="c2f", names=["celsius_to_fahrenheit", "c_to_f", "to_fahrenheit"], args="c", body="return c * 9 / 5 + 32",
         bug=("c * 9 / 5 + 32", "c * 5 / 9 + 32"), desc="converts Celsius to Fahrenheit",
         tests=[("100", "212.0"), ("0", "32.0")]),
    dict(key="words", names=["word_count", "count_words"], args="text", body="return len(text.split())",
         bug=("len(text.split())", "len(text)"), desc="counts the words in a text", tests=[('"a b c"', "3")]),
    dict(key="title", names=["title_case", "capitalize_words"], args="s",
         body='return " ".join(w.capitalize() for w in s.split())', bug=("w.capitalize()", "w.upper()"),
         desc="capitalizes the first letter of every word", tests=[('"hello world"', '"Hello World"')]),
    dict(key="pct", names=["percent", "percentage", "pct"], args="part, total", body="return part / total * 100",
         bug=("part / total * 100", "total / part * 100"), desc="returns part as a percentage of total",
         tests=[("1, 4", "25.0")]),
    dict(key="slug", names=["slugify", "make_slug", "to_slug"], args="s", body='return "-".join(s.lower().split())',
         bug=("s.lower().split()", "s.split()"), desc="turns a title into a lowercase dash-separated slug",
         tests=[('"Hello World"', '"hello-world"')]),
    dict(key="uniq", names=["unique", "dedupe", "remove_duplicates"], args="xs",
         body="seen = []\nfor x in xs:\n    if x not in seen:\n        seen.append(x)\nreturn seen",
         bug=("if x not in seen:", "if x in seen:"), desc="removes duplicates from a list, keeping order",
         tests=[("[1, 2, 1, 3]", "[1, 2, 3]")]),
    dict(key="square", names=["square", "squared"], args="x", body="return x * x", bug=("x * x", "x + x"),
         desc="returns x squared", tests=[("3", "9"), ("-4", "16")]),
    dict(key="pal", names=["is_palindrome", "palindrome"], args="s", body="s = s.lower()\nreturn s == s[::-1]",
         bug=("s == s[::-1]", "s == s[::1]"), desc="tells whether a string reads the same backwards",
         tests=[('"Level"', "True"), ('"abc"', "False")]),
]

MODULES = ["utils", "mathx", "helpers", "text_tools", "numutil", "stats", "convert", "strings", "calc", "ops",
           "misc", "shared", "funcs", "common"]
PACKAGES = ["app", "mylib", "toolkit", "proj", "engine", "service", "pkg", "lib"]
APP_WORDS = ["falcon", "orbit", "pixel", "maple", "quartz", "ember", "nimbus", "cobalt", "willow", "atlas",
             "juniper", "vertex", "lumen", "sable", "tundra", "zephyr"]
CONSTS = {"PORT": ("the server port", lambda r: r.randrange(3000, 9000)),
          "TIMEOUT": ("the request timeout in seconds", lambda r: r.choice([5, 10, 15, 30, 45, 60, 90, 120])),
          "MAX_RETRIES": ("the maximum number of retries", lambda r: r.randrange(1, 10)),
          "CACHE_SIZE": ("the cache size", lambda r: r.choice([64, 128, 256, 512, 1024, 2048])),
          "WORKERS": ("the number of worker threads", lambda r: r.randrange(2, 17))}
TEST_CMD = "python3 -m unittest discover -s tests"


def indent(code: str, n: int = 4) -> str:
    return "\n".join((" " * n + line) if line else line for line in code.split("\n"))


class Func:
    def __init__(self, t: dict, name: str, module: str):
        self.t, self.name, self.module = t, name, module
        self.args, self.body, self.bug, self.desc, self.tests = t["args"], t["body"], t["bug"], t["desc"], t["tests"]
        self.buggy = False

    def render(self) -> str:
        body = self.body.replace(self.bug[0], self.bug[1], 1) if self.buggy else self.body
        return f'def {self.name}({self.args}):\n    """{self.desc[0].upper() + self.desc[1:]}."""\n{indent(body)}\n'

    def bad_line(self) -> str:
        body = self.body.replace(self.bug[0], self.bug[1], 1)
        return next("    " + line for line in body.split("\n") if self.bug[1] in line)

    def good_line(self) -> str:
        return next("    " + line for line in self.body.split("\n") if self.bug[0] in line)


class Repo:
    """A tiny random Python project with modules, config, main.py and unittest tests."""

    def __init__(self, root: str, rng: random.Random, n_bugs: int = 0, exclude_keys=()):
        self.root, self.rng = root, rng
        self.app = rng.choice(APP_WORDS)
        self.pkg = rng.choice(PACKAGES) if rng.random() < 0.5 else None
        mods = rng.sample(MODULES, rng.randint(2, 3))
        pool = [t for t in TEMPLATES if t["key"] not in exclude_keys]
        temps = rng.sample(pool, rng.randint(len(mods) * 2, len(mods) * 3))
        self.funcs, used = [], set()
        for i, t in enumerate(temps):
            name = rng.choice([n for n in t["names"] if n not in used])
            used.add(name)
            self.funcs.append(Func(t, name, mods[i % len(mods)]))
        self.modules = mods
        for f in rng.sample(self.funcs, n_bugs):
            f.buggy = True
        keys = rng.sample(list(CONSTS), 3)
        self.consts = {k: CONSTS[k][1](rng) for k in keys}
        self.debug = rng.choice([True, False])
        self.used_in_main = rng.sample(self.funcs, min(len(self.funcs), rng.randint(2, 4)))
        self.write()

    # paths
    def mod_path(self, module: str) -> str:
        return f"{self.pkg}/{module}.py" if self.pkg else f"{module}.py"

    def imp(self, module: str) -> str:
        return f"{self.pkg}.{module}" if self.pkg else module

    def test_path(self, module: str) -> str:
        return f"tests/test_{module}.py"

    @property
    def config_path(self) -> str:
        return self.mod_path("config")

    def funcs_in(self, module: str) -> list:
        return [f for f in self.funcs if f.module == module]

    # rendering
    def render_module(self, module: str) -> str:
        head = f'"""{module.replace("_", " ").capitalize()} helpers for {self.app}."""\n\n\n'
        return head + "\n\n".join(f.render() for f in self.funcs_in(module))

    def render_test(self, module: str) -> str:
        fs = self.funcs_in(module)
        methods = []
        for f in fs:
            lines = [f"        self.assertEqual({f.name}({a}), {e})" for a, e in f.tests]
            methods.append(f"    def test_{f.name}(self):\n" + "\n".join(lines) + "\n")
        cls = "".join(p.capitalize() for p in module.split("_"))
        return ("import os\nimport sys\nimport unittest\n\n"
                "sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))\n\n"
                f"from {self.imp(module)} import {', '.join(f.name for f in fs)}\n\n\n"
                f"class Test{cls}(unittest.TestCase):\n" + "\n".join(methods) +
                '\n\nif __name__ == "__main__":\n    unittest.main()\n')

    def render_config(self) -> str:
        lines = [f'"""Configuration for {self.app}."""\n', f'APP_NAME = "{self.app}"', f"DEBUG = {self.debug}"]
        lines += [f"{k} = {v}" for k, v in self.consts.items()]
        return "\n".join(lines) + "\n"

    def render_main(self) -> str:
        by_mod = {}
        for f in self.used_in_main:
            by_mod.setdefault(f.module, []).append(f.name)
        imports = [f"from {self.imp('config')} import APP_NAME, DEBUG"]
        imports += [f"from {self.imp(m)} import {', '.join(sorted(ns))}" for m, ns in sorted(by_mod.items())]
        calls = [f"    print({f.name}({f.tests[0][0]}))" for f in self.used_in_main]
        return (f'"""Entry point for {self.app}."""\n' + "\n".join(imports) + "\n\n\ndef main():\n"
                "    if DEBUG:\n        print(\"debug mode\")\n    print(APP_NAME)\n" + "\n".join(calls) +
                '\n\n\nif __name__ == "__main__":\n    main()\n')

    def files(self) -> dict:
        out = {self.mod_path(m): self.render_module(m) for m in self.modules}
        out.update({self.test_path(m): self.render_test(m) for m in self.modules})
        out[self.config_path] = self.render_config()
        out["main.py"] = self.render_main()
        out["README.md"] = f"# {self.app}\n\nA small utility project.\n\nRun the tests:\n\n    {TEST_CMD}\n"
        if self.pkg:
            out[f"{self.pkg}/__init__.py"] = ""
        return out

    def write(self) -> None:
        for rel, content in self.files().items():
            path = os.path.join(self.root, rel)
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "w") as fh:
                fh.write(content)

    # facts
    def read(self, rel: str) -> str:
        with open(os.path.join(self.root, rel)) as fh:
            return fh.read()

    def def_line(self, f: Func) -> int:
        lines = self.read(self.mod_path(f.module)).split("\n")
        return next(i for i, line in enumerate(lines, 1) if line.startswith(f"def {f.name}("))

    def line_of(self, rel: str, needle: str) -> int:
        return next(i for i, line in enumerate(self.read(rel).split("\n"), 1) if needle in line)


def run_tests(root: str):
    r = subprocess.run(TEST_CMD, shell=True, cwd=root, capture_output=True, text=True, timeout=60)
    return r.returncode == 0, r.stdout + r.stderr


def snapshot(root: str) -> dict:
    out = {}
    for dp, dns, fns in os.walk(root):
        dns[:] = [d for d in dns if d != "__pycache__"]
        for fn in fns:
            p = os.path.join(dp, fn)
            with open(p, "rb") as fh:
                out[os.path.relpath(p, root)] = hashlib.md5(fh.read()).hexdigest()
    return out


def changed(before: dict, root: str) -> set:
    after = snapshot(root)
    return {k for k in set(before) | set(after) if before.get(k) != after.get(k)}


# ───────────────────────────────── tasks ─────────────────────────────────

class Task:
    def __init__(self, kind, tier, prompt, teacher, check):
        self.kind, self.tier, self.prompt, self.teacher, self.check = kind, tier, prompt, teacher, check


def pick(rng, *options):
    return rng.choice(options)


def task_find_def(rng, root, tier):
    repo = Repo(root, rng)
    f = rng.choice(repo.funcs)
    path, line = repo.mod_path(f.module), repo.def_line(f)
    explain = rng.random() < 0.4
    prompt = pick(rng, "Find where the function `{n}` is defined. Report the file and line number.",
                  "Where is `{n}` implemented? Give me the file path and line.",
                  "Locate the definition of `{n}` in this codebase and report its location.",
                  "Find the definition of the `{n}` function.").format(n=f.name)
    if explain:
        prompt += pick(rng, " Also explain briefly what it does.", " Also tell me what it does.")

    def teacher(env):
        args = {"pattern": f"def {f.name}\\b"}
        if rng.random() < 0.3:
            args["glob"] = "*.py"
        out = env.call(pick(rng, "", "", "Let me search for it."), "grep", args)
        assert f"{path}:{line}:" in out, out
        if explain:
            env.call("", "read_file", {"path": path, "offset": line, "limit": 12})
            env.final(f"`{f.name}({f.args})` is defined in `{path}` at line {line}. It {f.desc}.")
        else:
            env.final(pick(rng, f"`{f.name}` is defined in `{path}` at line {line}.",
                           f"Found it: `{path}:{line}` — `def {f.name}({f.args})`."))

    def check(root, report):
        ok = path in report and re.search(rf"\b{line}\b", report) is not None
        return ok, "" if ok else f"expected {path}:{line}"
    return Task("find_def", tier, prompt, teacher, check)


def task_usages(rng, root, tier):
    repo = Repo(root, rng)
    f = rng.choice(repo.used_in_main)
    files = sorted({"main.py", repo.test_path(f.module)})
    prompt = pick(rng, "Find every place where `{n}` is called (not where it is defined). List file and line for each.",
                  "Which files call `{n}`? Report each call site with its line number, excluding the definition.",
                  "List all call sites of `{n}` in the project.").format(n=f.name)

    def teacher(env):
        out = env.call(pick(rng, "", "Searching for call sites."), "grep", {"pattern": f"{f.name}\\("})
        sites = [ln.split(":")[:2] for ln in out.splitlines()
                 if ln.count(":") >= 2 and f"def {f.name}(" not in ln]
        assert {s[0] for s in sites} == set(files), (sites, files)
        body = "\n".join(f"- `{p}:{n}`" for p, n in sites)
        env.final(f"`{f.name}` is called in {len(sites)} place{'s' if len(sites) != 1 else ''}:\n{body}")

    def check(root, report):
        miss = [p for p in files if p not in report]
        return not miss, f"missing {miss}" if miss else ""
    return Task("usages", tier, prompt, teacher, check)


def task_summarize(rng, root, tier):
    repo = Repo(root, rng)
    mod = rng.choice(repo.modules)
    path, fs = repo.mod_path(mod), repo.funcs_in(mod)
    by_path = rng.random() < 0.6
    prompt = (pick(rng, "Read `{p}` and summarize it: list each function with a one-line description.",
                   "What does `{p}` contain? List its functions and what each one does.").format(p=path) if by_path
              else pick(rng, "Summarize the `{m}` module: list its functions and what they do.",
                        "Give me an overview of the {m} module's functions.").format(m=mod))

    def teacher(env):
        if not by_path:
            out = env.call("", "glob", {"pattern": f"**/{mod}.py"})
            assert path in out, out
        env.call("", "read_file", {"path": path})
        items = "\n".join(f"- `{f.name}({f.args})`: {f.desc}" for f in fs)
        env.final(f"`{path}` defines {len(fs)} functions:\n{items}")

    def check(root, report):
        miss = [f.name for f in fs if f.name not in report]
        return not miss, f"missing {miss}" if miss else ""
    return Task("summarize", tier, prompt, teacher, check)


def task_config(rng, root, tier):
    repo = Repo(root, rng)
    const = rng.choice(list(repo.consts))
    val, cfg = repo.consts[const], repo.config_path
    prompt = pick(rng, "What is the value of `{c}` in this project, and where is it set?",
                  "Find out what {d} is configured to. Report the value and the file.",
                  "Look up the `{c}` setting and report its value.").format(c=const, d=CONSTS[const][0])

    def teacher(env):
        out = env.call("", "grep", {"pattern": f"{const}\\s*="})
        line = repo.line_of(cfg, f"{const} =")
        assert f"{cfg}:{line}:" in out, out
        env.final(f"`{const}` is set to `{val}` in `{cfg}` (line {line}).")

    def check(root, report):
        ok = re.search(rf"\b{val}\b", report) is not None
        return ok, "" if ok else f"expected {val}"
    return Task("config", tier, prompt, teacher, check)


def _fail_names(out: str) -> list:
    return re.findall(r"^(?:FAIL|ERROR): (test_\w+)", out, re.M)


def task_run_tests(rng, root, tier):
    repo = Repo(root, rng, n_bugs=rng.choice([0, 1, 1, 2]))
    bad = sorted(f"test_{f.name}" for f in repo.funcs if f.buggy)
    prompt = pick(rng, "Run the test suite and report the results. Don't change any code.",
                  "Run the tests and tell me which ones fail, if any.",
                  "Check whether the tests pass. Report failures with their error messages.")

    def teacher(env):
        if rng.random() < 0.3:
            env.call("", "read_file", {"path": "README.md"})
        out = env.call(pick(rng, "", "Running the tests."), "bash", {"command": TEST_CMD})
        fails = sorted(_fail_names(out))
        assert fails == bad, (fails, bad, out)
        total = re.search(r"Ran (\d+) test", out).group(1)
        if not fails:
            env.final(f"All {total} tests pass.")
            return
        errs = re.findall(r"AssertionError: (.*)", out)
        lines = [f"- `{n}`" + (f": AssertionError: {e}" if i < len(errs) else "") for i, (n, e) in
                 enumerate(zip(fails, errs + [""] * len(fails)))]
        env.final(f"Ran {total} tests, {len(fails)} failed:\n" + "\n".join(lines))

    def check(root, report):
        if not bad:
            ok = "pass" in report.lower() and "fail" not in report.lower().replace("0 fail", "")
            return ok, "" if ok else "expected all-pass report"
        miss = [b for b in bad if b not in report]
        return not miss, f"missing {miss}" if miss else ""
    return Task("run_tests", tier, prompt, teacher, check)


def task_fix_bug(rng, root, tier):
    repo = Repo(root, rng, n_bugs=1)
    f = next(x for x in repo.funcs if x.buggy)
    path = repo.mod_path(f.module)
    a, e = f.tests[0]
    vague = rng.random() < 0.4
    prompt = (pick(rng, "The test suite is failing. Find the bug, fix it, and make sure all tests pass.",
                   "Some tests fail. Track down the cause, fix it with a minimal change, and rerun the tests.")
              if vague else
              pick(rng, "In `{p}`, `{n}` returns the wrong result: `{n}({a})` should return {e}. Fix it and run the tests.",
                   "`{n}` in `{p}` is broken (`{n}({a})` should be {e}). Fix the bug and confirm the tests pass.")
              .format(p=path, n=f.name, a=a, e=e))
    before = snapshot(root)

    def teacher(env):
        if vague:
            out = env.call(pick(rng, "", "Let me run the tests first."), "bash", {"command": TEST_CMD})
            assert f"test_{f.name}" in out, out
            out = env.call("", "grep", {"pattern": f"def {f.name}\\b"})
            assert path in out
        env.call("", "read_file", {"path": path})
        bad, good = f.bad_line(), f.good_line()
        text = repo.read(path)
        old, new = bad, good
        if text.count(old) != 1:
            old = f"def {f.name}({f.args}):" + text.split(f"def {f.name}({f.args}):")[1].split(bad)[0] + bad
            new = old[: -len(bad)] + good
        if rng.random() < 0.25:  # recovery example: sloppy old_string, then fix
            sloppy = bad.strip().replace(" ", "", 2)
            if sloppy not in text:
                out = env.call("", "edit_file", {"path": path, "old_string": sloppy, "new_string": good.strip()})
                assert out.startswith("Error"), out
        env.call(pick(rng, "", f"`{f.name}` has the bug. Fixing it."), "edit_file",
                 {"path": path, "old_string": old, "new_string": new})
        out = env.call("", "bash", {"command": TEST_CMD})
        assert "OK" in out and "FAILED" not in out, out
        total = re.search(r"Ran (\d+) test", out).group(1)
        env.final(f"Fixed `{f.name}` in `{path}`: changed `{bad.strip()}` to `{good.strip()}`. All {total} tests pass.")

    def check(root, report):
        ok, _ = run_tests(root)
        extra = changed(before, root) - {path}
        if not ok:
            return False, "tests fail"
        return not extra, f"touched other files {sorted(extra)}" if extra else ""
    return Task("fix_bug", tier, prompt, teacher, check)


def task_rename(rng, root, tier):
    repo = Repo(root, rng)
    f = rng.choice(repo.funcs)
    old = f.name
    new = rng.choice([n for n in f.t["names"] if n != old] + [f"compute_{f.t['key']}", f"{old}_value", f"do_{old}"])
    while any(new == g.name for g in repo.funcs):
        new = f"{new}_fn"
    prompt = pick(rng, "Rename the function `{o}` to `{n}` everywhere (definition, imports and call sites), then run the tests.",
                  "Rename `{o}` to `{n}` across the whole project and make sure the tests still pass.").format(o=old, n=new)

    def teacher(env):
        out = env.call(pick(rng, "", f"Finding all uses of `{old}`."), "grep", {"pattern": f"\\b{old}\\b"})
        files = sorted({ln.split(":")[0] for ln in out.splitlines() if ln.count(":") >= 2})
        for p in files:
            if rng.random() < 0.5:
                env.call("", "read_file", {"path": p})
            env.call("", "edit_file", {"path": p, "old_string": old, "new_string": new, "replace_all": True})
        out = env.call("", "grep", {"pattern": f"\\b{old}\\b"})
        assert out.startswith("No matches"), out
        out = env.call("", "bash", {"command": TEST_CMD})
        assert "FAILED" not in out, out
        env.final(f"Renamed `{old}` to `{new}` in {len(files)} files ({', '.join(f'`{p}`' for p in files)}). "
                  "No references to the old name remain and all tests pass.")

    def check(root, report):
        for p in snapshot(root):
            if p.endswith(".py") and re.search(rf"\b{old}\b", open(os.path.join(root, p)).read()):
                return False, f"old name left in {p}"
        ok, _ = run_tests(root)
        return ok, "" if ok else "tests fail"
    return Task("rename", tier, prompt, teacher, check)


def task_add_function(rng, root, tier):
    repo = Repo(root, rng)
    used = {f.t["key"] for f in repo.funcs}
    t = rng.choice([x for x in TEMPLATES if x["key"] not in used])
    mod = rng.choice(repo.modules)
    nf = Func(t, rng.choice(t["names"]), mod)
    path, tpath = repo.mod_path(mod), repo.test_path(mod)
    prompt = pick(rng, "Add a function `{n}({a})` to `{p}` that {d}. Add a unit test for it in `{t}` and run the tests.",
                  "In `{p}`, implement `{n}({a})`: it {d}. Then add a test to `{t}` and make sure everything passes.") \
        .format(n=nf.name, a=nf.args, p=path, d=nf.desc, t=tpath)

    def teacher(env):
        env.call("", "read_file", {"path": path})
        text = repo.read(path).rstrip("\n")
        lines = text.split("\n")
        k = 2
        while text.count("\n".join(lines[-k:])) != 1:
            k += 1
        anchor = "\n".join(lines[-k:])
        env.call(pick(rng, "", f"Adding `{nf.name}` at the end of the module."), "edit_file",
                 {"path": path, "old_string": anchor, "new_string": anchor + "\n\n\n" + nf.render().rstrip("\n")})
        env.call("", "read_file", {"path": tpath})
        ttext = repo.read(tpath)
        imp = next(ln for ln in ttext.split("\n") if ln.startswith("from "))
        env.call("", "edit_file", {"path": tpath, "old_string": imp, "new_string": imp + f", {nf.name}"})
        test = f"    def test_{nf.name}(self):\n" + "\n".join(
            f"        self.assertEqual({nf.name}({a}), {e})" for a, e in nf.tests) + "\n"
        tail = '\n\nif __name__ == "__main__":'
        env.call("", "edit_file", {"path": tpath, "old_string": tail, "new_string": "\n" + test + tail})
        out = env.call("", "bash", {"command": TEST_CMD})
        assert "FAILED" not in out and "Error" not in out.split("Ran")[0], out
        env.final(f"Added `{nf.name}({nf.args})` to `{path}` and a `test_{nf.name}` test in `{tpath}`. All tests pass.")

    def check(root, report):
        code = (f"import sys; sys.path.insert(0, {root!r}); from {repo.imp(mod)} import {nf.name}; "
                + "; ".join(f"assert {nf.name}({a}) == {e}" for a, e in nf.tests))
        r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=30)
        if r.returncode:
            return False, "function missing or wrong: " + r.stderr.strip()[-200:]
        ok, _ = run_tests(root)
        if not ok:
            return False, "tests fail"
        has_test = f"test_{nf.name}" in open(os.path.join(root, tpath)).read()
        return has_test, "" if has_test else "no test added"
    return Task("add_function", tier, prompt, teacher, check)


def task_change_const(rng, root, tier):
    repo = Repo(root, rng)
    const = rng.choice(list(repo.consts))
    old = repo.consts[const]
    new = CONSTS[const][1](rng)
    while new == old:
        new = CONSTS[const][1](rng)
    cfg = repo.config_path
    prompt = pick(rng, "Change `{c}` from {o} to {n}.", "Set {d} to {n}.",
                  "Update the `{c}` setting to {n}. Don't change anything else.").format(
        c=const, o=old, n=new, d=CONSTS[const][0])
    before = snapshot(root)

    def teacher(env):
        out = env.call("", "grep", {"pattern": f"{const}\\s*="})
        assert cfg in out
        if rng.random() < 0.5:
            env.call("", "read_file", {"path": cfg})
        env.call("", "edit_file", {"path": cfg, "old_string": f"{const} = {old}", "new_string": f"{const} = {new}"})
        env.final(f"Changed `{const}` in `{cfg}` from {old} to {new}.")

    def check(root, report):
        ok = f"{const} = {new}\n" in open(os.path.join(root, cfg)).read()
        extra = changed(before, root) - {cfg}
        if not ok:
            return False, "value not changed"
        return not extra, f"touched other files {sorted(extra)}" if extra else ""
    return Task("change_const", tier, prompt, teacher, check)


def task_readonly_fix(rng, root, tier):
    repo = Repo(root, rng, n_bugs=1)
    f = next(x for x in repo.funcs if x.buggy)
    path = repo.mod_path(f.module)
    a, e = f.tests[0]
    prompt = pick(rng, "Fix the bug in `{n}` in `{p}`: `{n}({a})` should return {e}.",
                  "`{n}` gives wrong results (`{n}({a})` should be {e}). Please fix it.").format(n=f.name, p=path, a=a, e=e)
    before = snapshot(root)

    def teacher(env):
        if rng.random() < 0.5:
            env.call("", "grep", {"pattern": f"def {f.name}\\b"})
        env.call("", "read_file", {"path": path})
        line = repo.line_of(path, f.bad_line().strip())
        env.final(f"I'm read-only, so I didn't change any files. The bug is in `{path}` line {line}: "
                  f"`{f.bad_line().strip()}` should be `{f.good_line().strip()}`.")

    def check(root, report):
        if changed(before, root):
            return False, "files changed"
        ok = f.bug[0] in report
        return ok, "" if ok else "fix not reported"
    return Task("readonly_fix", tier, prompt, teacher, check)


READ_TASKS = {"find_def": task_find_def, "usages": task_usages, "summarize": task_summarize, "config": task_config}
MIX = {
    "small": {"find_def": 25, "usages": 20, "summarize": 20, "config": 15, "readonly_fix": 20},
    "medium": {"find_def": 8, "usages": 7, "summarize": 7, "config": 5, "run_tests": 13, "fix_bug": 22,
               "rename": 13, "add_function": 15, "change_const": 10},
}
BUILDERS = dict(READ_TASKS, run_tests=task_run_tests, fix_bug=task_fix_bug, rename=task_rename,
                add_function=task_add_function, change_const=task_change_const, readonly_fix=task_readonly_fix)


def sample_task(rng: random.Random, root: str, tier: str, kind: str = None) -> Task:
    mix = MIX[tier]
    kind = kind or rng.choices(list(mix), weights=list(mix.values()))[0]
    return BUILDERS[kind](rng, root, tier)
