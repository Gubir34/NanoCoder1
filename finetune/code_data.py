"""Code-writing samples from MBPP (CC-BY-4.0, google-research-datasets/mbpp).

Two formats per problem, both verified by running the problem's tests:
  code_chat   plain request -> a clean code block (no tools)
  code_agent  "implement it in this repo": a medium-tier worker reads the module, adds the function,
              adds tests and runs them, using agentcoder's real tools (like gen_data.py)

Reference solutions are normalized with ast.unparse (MBPP's originals have CRLFs and odd spacing).
Training uses the train/validation/prompt splits; code_eval.py uses the held-out test split.

    pip install pyarrow
    python finetune/code_data.py --mbpp /path/to/mbpp --out finetune/data/code.jsonl
    # /path/to/mbpp holds {train,validation,prompt,test}.parquet from
    # https://huggingface.co/api/datasets/google-research-datasets/mbpp/parquet/full
"""
import argparse
import ast
import json
import os
import random
import re
import shutil
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from gen_data import Env  # noqa: E402
from tasks import TEST_CMD  # noqa: E402

CHAT_SYSTEM = "You are a precise Python coding assistant. Answer with correct, clean code."
TRAIN_SPLITS = ("train", "validation", "prompt")
MODULES = ["utils", "helpers", "solutions", "algo", "tools", "core", "lib", "funcs"]
# MBPP is downloaded data: only keep problems whose code imports these pure-computation modules
SAFE_IMPORTS = {"re", "math", "cmath", "collections", "heapq", "itertools", "operator", "array", "bisect",
                "datetime", "copy", "functools", "string", "statistics", "fractions", "decimal"}
IMPORT_RE = re.compile(r"^\s*(?:import|from)\s+([\w.]+)", re.M)
RISKY_RE = re.compile(r"__import__|\b(?:eval|exec|open|compile|getattr|globals|locals)\s*\(")
NAME_RE = re.compile(r"assert\s+(?:not\s+)?\(?\s*(?:set\(|math\.isclose\(|round\()?\s*([A-Za-z_]\w*)\s*\(")


def load(mbpp_dir, splits):
    import pyarrow.parquet as pq
    rows = []
    for s in splits:
        rows += pq.read_table(os.path.join(mbpp_dir, f"{s}.parquet")).to_pylist()
    return rows


def clean(code):
    return ast.unparse(ast.parse(code.replace("\r", ""))).strip()


def safe(code):
    mods = {m.split(".")[0] for m in IMPORT_RE.findall(code)}
    return mods <= SAFE_IMPORTS | {"sys"} and not RISKY_RE.search(code) and \
        all(a in ("maxsize", "getsizeof") for a in re.findall(r"\bsys\.(\w+)", code))


def passes(code, tests, timeout=10):
    """Run code + asserts in an isolated interpreter (-I), in an empty temp dir, with a timeout."""
    blob = code + "\n\n" + "\n".join(tests)
    if not safe(blob):
        return False
    tmp = tempfile.mkdtemp(prefix="ncrun_")
    try:
        r = subprocess.run([sys.executable, "-I", "-c", blob], cwd=tmp, capture_output=True, text=True,
                           timeout=timeout, stdin=subprocess.DEVNULL)
        return r.returncode == 0
    except subprocess.TimeoutExpired:
        return False
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def func_name(row):
    m = NAME_RE.search(row["test_list"][0])
    return m.group(1) if m else None


def usable(row):
    """Clean the reference and keep the problem only if it still passes and has a clear entry point."""
    if row["test_setup_code"].strip():
        return None
    try:
        code = clean(row["code"])
    except SyntaxError:
        return None
    name = func_name(row)
    if not name or not re.search(rf"^def {re.escape(name)}\(", code, re.M):
        return None
    if not passes(code, row["test_list"]):
        return None
    return {"id": row["task_id"], "text": row["text"].strip(), "code": code, "name": name,
            "tests": row["test_list"]}


def chat_prompt(p, rng=None):
    tests = "\n".join(p["tests"])
    if rng is None or rng.random() < 0.6:
        return f"{p['text']}\nYour code should pass these tests:\n\n{tests}"
    return rng.choice([
        f"{p['text']} Call the function `{p['name']}`.",
        f"Python: {p['text'][0].lower() + p['text'][1:]} Name it `{p['name']}`. Example: `{p['tests'][0][7:]}`",
        f"{p['text']}\nFor example:\n{p['tests'][0]}",
    ])


def chat_sample(p, rng):
    return {"tier": "chat", "kind": "code_chat", "tools": None, "messages": [
        {"role": "system", "content": CHAT_SYSTEM},
        {"role": "user", "content": chat_prompt(p, rng)},
        {"role": "assistant", "content": f"```python\n{p['code']}\n```"}]}


def to_unittest(line):
    """'assert f(x) == y' -> 'self.assertEqual(f(x), y)'; anything else -> assertTrue."""
    expr = ast.parse(line).body[0].test
    if isinstance(expr, ast.Compare) and len(expr.ops) == 1 and isinstance(expr.ops[0], ast.Eq):
        return f"self.assertEqual({ast.unparse(expr.left)}, {ast.unparse(expr.comparators[0])})"
    return f"self.assertTrue({ast.unparse(expr)})"


def split_imports(code):
    tree = ast.parse(code)
    imps = [n for n in tree.body if isinstance(n, (ast.Import, ast.ImportFrom))]
    rest = [n for n in tree.body if not isinstance(n, (ast.Import, ast.ImportFrom))]
    return [ast.unparse(n) for n in imps], "\n\n\n".join(ast.unparse(n) for n in rest)


def write_repo(root, mod, existing):
    """A small module with a few already-solved problems and their unittest file."""
    imps, bodies = [], []
    for p in existing:
        i, b = split_imports(p["code"])
        imps += [x for x in i if x not in imps]
        bodies.append(b)
    head = '"""Assorted helpers."""\n' + ("\n" + "\n".join(imps) + "\n" if imps else "")
    with open(os.path.join(root, f"{mod}.py"), "w") as fh:
        fh.write(head + "\n\n" + "\n\n\n".join(bodies) + "\n")
    os.makedirs(os.path.join(root, "tests"), exist_ok=True)
    names = ", ".join(p["name"] for p in existing)
    methods = "\n\n".join(f"    def test_{p['name']}(self):\n" + "\n".join(f"        {to_unittest(t)}" for t in p["tests"])
                          for p in existing)
    with open(os.path.join(root, "tests", f"test_{mod}.py"), "w") as fh:
        fh.write(f"import unittest\n\nfrom {mod} import {names}\n\n\nclass Test{mod.title()}(unittest.TestCase):\n"
                 f"{methods}\n\n\nif __name__ == \"__main__\":\n    unittest.main()\n")
    open(os.path.join(root, "tests", "__init__.py"), "w").close()


def agent_sample(p, pool, rng):
    root = tempfile.mkdtemp(prefix="ncc_")
    try:
        mod = rng.choice(MODULES)
        existing = rng.sample([q for q in pool if q["name"] != p["name"]], rng.randint(1, 3))
        write_repo(root, mod, existing)
        path, tpath = f"{mod}.py", f"tests/test_{mod}.py"
        shown = "\n".join(p["tests"][:2])
        prompt = rng.choice([
            f"In `{path}`, add a function `{p['name']}`: {p['text'][0].lower() + p['text'][1:]} "
            f"It must satisfy:\n{shown}\nAdd these as a test in `{tpath}` and run the tests.",
            f"{p['text']} Put it in `{path}` as `{p['name']}`, add tests to `{tpath}` "
            f"(e.g. `{p['tests'][0]}`) and make sure the suite passes.",
        ])
        env = Env(root, "medium", prompt)
        imps, body = split_imports(p["code"])
        src = env.call("", "read_file", {"path": path})
        text = open(os.path.join(root, path)).read()
        new_imps = [i for i in imps if i not in text]
        if new_imps:
            env.call(rng.choice(["", "It needs an import first."]), "edit_file",
                     {"path": path, "old_string": '"""Assorted helpers."""',
                      "new_string": '"""Assorted helpers."""\n' + "\n".join(new_imps)})
            text = open(os.path.join(root, path)).read()
        lines = text.rstrip("\n").split("\n")
        k = 2
        while text.count("\n".join(lines[-k:])) != 1:
            k += 1
        anchor = "\n".join(lines[-k:])
        env.call(rng.choice(["", f"Adding `{p['name']}` at the end of the module."]), "edit_file",
                 {"path": path, "old_string": anchor, "new_string": anchor + "\n\n\n" + body})
        env.call("", "read_file", {"path": tpath})
        ttext = open(os.path.join(root, tpath)).read()
        imp = next(ln for ln in ttext.split("\n") if ln.startswith(f"from {mod} import"))
        method = f"    def test_{p['name']}(self):\n" + "\n".join(f"        {to_unittest(t)}" for t in p["tests"]) + "\n"
        tail = '\n\nif __name__ == "__main__":'
        env.call("", "edit_file", {"path": tpath, "old_string": imp, "new_string": imp + f", {p['name']}"})
        env.call("", "edit_file", {"path": tpath, "old_string": tail, "new_string": "\n" + method + tail})
        out = env.call("", "bash", {"command": TEST_CMD})
        if not out.rstrip().endswith("OK") and "\nOK" not in out:
            raise AssertionError(out[-300:])
        n = out.split("Ran ")[1].split(" ")[0] if "Ran " in out else "all"
        env.final(f"Added `{p['name']}` to `{path}` and `test_{p['name']}` to `{tpath}`. All {n} tests pass.")
        assert src  # the module was read before editing
        blob = json.dumps({"tier": "medium", "kind": "code_agent", "messages": env.messages,
                           "tools": env.agent.schemas()})
        return json.loads(blob.replace(root, "/home/user/project"))
    finally:
        shutil.rmtree(root, ignore_errors=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mbpp", required=True, help="directory with MBPP parquet files")
    ap.add_argument("--out", default=os.path.join(HERE, "data", "code.jsonl"))
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--agent-frac", type=float, default=0.6, help="share of problems that also get a code_agent sample")
    args = ap.parse_args()
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    rng = random.Random(args.seed)
    probs = [p for p in map(usable, load(args.mbpp, TRAIN_SPLITS)) if p]
    print(f"{len(probs)} usable MBPP problems")
    n_chat = n_agent = bad = 0
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as fh:
        for p in probs:
            fh.write(json.dumps(chat_sample(p, rng)) + "\n")
            n_chat += 1
            if rng.random() < args.agent_frac:
                try:
                    fh.write(json.dumps(agent_sample(p, probs, rng)) + "\n")
                    n_agent += 1
                except (AssertionError, SyntaxError, StopIteration) as e:
                    bad += 1
                    print(f"skip {p['id']}: {str(e)[:100]}", file=sys.stderr)
    print(f"code_chat {n_chat}, code_agent {n_agent}, skipped {bad}")


if __name__ == "__main__":
    main()
