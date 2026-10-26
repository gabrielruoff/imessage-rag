"""Contract tests for the entry CLIs: `python3 <entry>.py` must actually run.

A file run as a script is the module `__main__`, a second module object beside
the one its children import by name. A parent that re-exports child names from
a bottom from-import therefore has two script-only failures:
  CLASS 1 -- a guard above the bottom import calls the local `main`, which reads
    a name that is not bound yet: NameError.
  CLASS 2 -- `__main__` runs the bottom import itself and re-enters a
    half-initialized child: ImportError.

Shape rule: every module-level `if __name__ == "__main__":` guard is the last
module-level statement, or its body is exactly `run_cli(<name>)` /
`db.run_cli(<name>)`, which re-reads `main` off the fully-initialized real
module. EntrypointShapeTest guards CLASS 1 (statically, plus a runtime proof of
that re-entry); EntrypointStartupTest (`--help`) guards CLASS 2.

Limits: a CLI that is not a git-tracked root-level file or listed in
PACKAGE_ENTRYPOINTS is not covered, and `--help` exits inside argparse, so it
proves initialization, not main()'s body.
"""

import ast
import os
import subprocess
import sys
import tempfile
import textwrap
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Packages run as `python3 -m <pkg>`; their __main__.py is a real entrypoint
# that a root-directory scan cannot see. `make eval` invokes the first of these.
PACKAGE_ENTRYPOINTS = ("eval.evalkit",)


def _iter_entry_files():
    """(display name, absolute path) for every git-tracked root-level module,
    plus each package `__main__.py`. Tracked files only, so the `--help` leg
    never executes an untracked scratch script."""
    try:
        out = subprocess.run(["git", "-C", ROOT, "ls-files", "--", "*.py"],
                             capture_output=True, text=True, check=True).stdout
    except (OSError, subprocess.CalledProcessError) as exc:
        raise unittest.SkipTest(f"entrypoint discovery needs a git checkout: {exc}")
    for name in sorted(out.splitlines()):
        if "/" not in name:
            yield name, os.path.join(ROOT, name)
    for pkg in PACKAGE_ENTRYPOINTS:
        path = os.path.join(ROOT, *pkg.split("."), "__main__.py")
        if os.path.exists(path):
            yield pkg, path


def _is_main_guard(node):
    """A module-level `if __name__ == "__main__"`, and only that. The `==` is
    load-bearing: an `if __name__ != "__main__"` block is dead on the script
    path, so it is not a guard."""
    if not isinstance(node, ast.If) or not isinstance(node.test, ast.Compare):
        return False
    if len(node.test.ops) != 1 or not isinstance(node.test.ops[0], ast.Eq):
        return False
    sides = (node.test.left, node.test.comparators[0])
    return (any(isinstance(s, ast.Name) and s.id == "__name__" for s in sides)
            and any(isinstance(s, ast.Constant) and s.value == "__main__" for s in sides))


def _entrypoints():
    """(display name, tree) for every module with a `__main__` guard."""
    found = []
    for name, path in _iter_entry_files():
        with open(path, encoding="utf-8") as fh:
            src = fh.read()
        try:
            tree = ast.parse(src)
        except SyntaxError:
            continue
        if any(_is_main_guard(n) for n in tree.body):
            found.append((name, tree))
    return found


def _imports(tree, module, name):
    """True if the module body binds `name` from `module` by an un-aliased
    `from module import name` (or `import module` when `name` is None)."""
    for node in tree.body:
        if name is None and isinstance(node, ast.Import):
            if any(a.name == module and a.asname is None for a in node.names):
                return True
        if (name is not None and isinstance(node, ast.ImportFrom)
                and node.module == module and node.level == 0
                and any(a.name == name and a.asname is None for a in node.names)):
            return True
    return False


def _is_run_cli_handoff(guard, tree):
    """The guard body is exactly one `run_cli(<Name>)` / `db.run_cli(<Name>)`,
    with run_cli really db's. Exact on purpose: anything else in the guard runs
    before the handoff, so it is not a handoff."""
    if len(guard.body) != 1 or not isinstance(guard.body[0], ast.Expr):
        return False
    call = guard.body[0].value
    if not (isinstance(call, ast.Call) and len(call.args) == 1 and not call.keywords
            and isinstance(call.args[0], ast.Name)):
        return False
    func = call.func
    if isinstance(func, ast.Name) and func.id == "run_cli":
        shadowed = any(isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
                       and n.name == "run_cli" for n in tree.body)
        return _imports(tree, "db", "run_cli") and not shadowed
    return (isinstance(func, ast.Attribute) and func.attr == "run_cli"
            and isinstance(func.value, ast.Name) and func.value.id == "db"
            and _imports(tree, "db", None))


def guard_violations(tree):
    """Line of every `__main__` guard that is neither the last module-level
    statement nor a run_cli handoff. Every guard, not just the last: a stale one
    above a good one still runs first."""
    return [n.lineno for n in tree.body
            if _is_main_guard(n) and n is not tree.body[-1]
            and not _is_run_cli_handoff(n, tree)]


# Planted sources for the self-test. LATE_NAME is bound below the guard.
_LAYOUT = """\
{header}

def main():
    return LATE_NAME

if __name__ == "__main__":
{body}

from somewhere import LATE_NAME
"""


def _planted(body, header="from db import run_cli"):
    return _LAYOUT.format(header=header, body=textwrap.indent(body, "    "))


class EntrypointShapeTest(unittest.TestCase):
    """CLASS 1: no guard may run before a later module-level binding."""

    def test_entrypoints_are_discovered(self):
        # A collapse to zero would make every other assertion here vacuous.
        names = [name for name, _ in _entrypoints()]
        self.assertGreaterEqual(len(names), 10, names)
        for expected in ("annotate.py", "conflict_detect.py", "chat.py",
                         "ingest.py", "eval.evalkit"):
            self.assertIn(expected, names)

    def test_no_entrypoint_can_run_before_its_bindings(self):
        for name, tree in _entrypoints():
            with self.subTest(entrypoint=name):
                bad = guard_violations(tree)
                self.assertEqual(
                    bad, [],
                    f"{name}: the guard at line(s) {bad} runs before later "
                    "module-level statements; make it the last statement or a "
                    "bare `run_cli(main)`.")

    def test_shape_rule_on_planted_sources(self):
        # Asserts the exact verdict the real test checks, so the rule cannot rot
        # into an always-empty result and stay green.
        guard_last = textwrap.dedent("""\
            def main():
                return LATE_NAME

            from somewhere import LATE_NAME

            if __name__ == "__main__":
                main()
        """)
        stale_above_good = _planted("main()").replace(
            "\nfrom somewhere",
            '\nif __name__ == "__main__":\n    run_cli(main)\n\nfrom somewhere')
        cases = [
            ("main() above the bottom import", _planted("main()"), [6]),
            ("main() as the last statement", guard_last, []),
            ("run_cli(main) above the bottom import", _planted("run_cli(main)"), []),
            ("db.run_cli(main) above the bottom import",
             _planted("db.run_cli(main)", header="import db"), []),
            ("cosmetic self-import before main()",
             _planted("import planted\nprint(planted.__name__)\nmain()"), [6]),
            ("run_cli(main) plus a second statement",
             _planted("run_cli(main)\nprint('done')"), [6]),
            ("run_cli is a local def, not db's",
             _planted("run_cli(main)", header="def run_cli(f): return f()"), [6]),
            ("db.run_cli without `import db`",
             _planted("db.run_cli(main)", header="from db import run_cli"), [6]),
            ("stale main() guard above a good one", stale_above_good, [6]),
            ("an != guard is not a guard",
             _planted("main()").replace("==", "!="), []),
        ]
        for label, src, expected in cases:
            with self.subTest(label):
                self.assertEqual(guard_violations(ast.parse(src)), expected, src)


class RunCliReentryTest(unittest.TestCase):
    """The shape rule's premise: `run_cli(main)` above a bottom import works
    because db.run_cli re-reads `main` off the fully-initialized real module."""

    PARENT = textwrap.dedent("""\
        from db import run_cli


        def main():
            print(child_value())
            return 0


        if __name__ == "__main__":
            {guard}

        from child import child_value  # noqa: E402
    """)
    CHILD = 'import parent  # noqa: F401\n\n\ndef child_value():\n    return "ok"\n'

    def _run(self, guard, argv):
        with tempfile.TemporaryDirectory() as tmp:
            with open(os.path.join(tmp, "parent.py"), "w", encoding="utf-8") as fh:
                fh.write(self.PARENT.format(guard=guard))
            with open(os.path.join(tmp, "child.py"), "w", encoding="utf-8") as fh:
                fh.write(self.CHILD)
            env = dict(os.environ, PYTHONPATH=ROOT)
            return subprocess.run([sys.executable, *argv], cwd=tmp, env=env,
                                  capture_output=True, text=True,
                                  stdin=subprocess.DEVNULL, timeout=60)

    def test_run_cli_reenters_through_the_real_module(self):
        for argv in (["parent.py"], ["-m", "parent"]):
            with self.subTest(argv=argv):
                result = self._run("run_cli(main)", argv)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn("ok", result.stdout)

    def test_a_bare_main_call_in_the_same_shape_raises_nameerror(self):
        # Negative control: without it the positive case could pass for a
        # reason other than the re-entry.
        result = self._run("main()", ["parent.py"])
        self.assertNotEqual(result.returncode, 0, result.stdout)
        self.assertIn("NameError", result.stderr)


class EntrypointStartupTest(unittest.TestCase):
    """CLASS 2: every entry CLI must survive module initialization."""

    def test_every_entrypoint_survives_help(self):
        # stdin is closed so a future non-argparse script fails fast instead of
        # hanging.
        for name, _ in _entrypoints():
            with self.subTest(entrypoint=name):
                argv = ([sys.executable, "-m", name] if not name.endswith(".py")
                        else [sys.executable, name])
                result = subprocess.run(
                    argv + ["--help"], cwd=ROOT, capture_output=True, text=True,
                    stdin=subprocess.DEVNULL, timeout=60,
                )
                # Only an actual ImportError is the cycle; anything else here is
                # a broken parser or a module that fails to initialize, and
                # saying "import cycle" would send the reader to the wrong file.
                hint = ("An ImportError here means the __main__ copy of "
                        f"{name} cannot resolve its own import cycle."
                        if "ImportError" in result.stderr else
                        f"{name} either failed to initialize or its parser "
                        "rejected --help.")
                self.assertEqual(
                    result.returncode, 0,
                    f"`{' '.join(argv[1:])} --help` exited "
                    f"{result.returncode}. {hint}\n{result.stderr}")


if __name__ == "__main__":
    unittest.main()
