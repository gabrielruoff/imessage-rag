"""Contract tests for the entry CLIs: `python3 <entry>.py` must actually run.

THE governing fact: a file run as a script is loaded as the module `__main__`,
which is a DIFFERENT module object from the one its own siblings import by
name. Several parents here (annotate, conflict_detect) were split into child
modules that `import <parent>` at module level to read the parent's knobs late,
and the parent re-exports the children's names from a from-import at the BOTTOM
of the file. That shape has two script-only failure modes, and NEITHER is
visible to the rest of the suite, to ruff, or to a `--help` smoke test:

  CLASS 1 -- name unbound on the script path. If `if __name__ == "__main__"`
    sits ABOVE the bottom imports, the guard calls main() before those imports
    bind, and main() dies on NameError. Every other caller is fine: `import
    annotate` runs the whole module first, so the name IS bound -- which is why
    the test suite passed while `make pipeline` did not. ruff sees a
    module-level binding and reports nothing. `--help` exits inside argparse
    before main()'s body, so it passes too.

  CLASS 2 -- the __main__ double-module cycle. If `__main__` executes the
    bottom from-import itself, Python loads a second copy of the parent, which
    re-enters the child while the child is still on its own `import <parent>`
    line -- so `from <child> import NAME` raises ImportError on a still-empty
    module. This one DOES break `--help`, because it fails at import.

Both parents resolve both classes the same way: the guard sits ABOVE the bottom
imports and delegates through the real module name (`import annotate;
run_cli(annotate.main)`), so the cycle resolves against one fully-initialized
copy and `main` is read off THAT copy. annotate.py would also work with a plain
guard moved below its bottom import, because its top-level `import
annotate_distill` already forces the full load -- but only by accident, so it
uses the shim too.

`test_no_entrypoint_calls_an_unbound_name` is the CLASS 1 guard; it is static,
so it needs no DB and costs nothing. `test_every_entrypoint_survives_help` is
the CLASS 2 guard and spends one subprocess per CLI.

Scope notes, all deliberate. The static leg reasons about module-level names
only: a name bound inside a module-level `try`/`except` or `if` counts as bound
at its statement's line (a `del`, or an optional import that fails at runtime,
would fool it), and a function reached only through a dict/registry rather than
a direct name reference is not walked. Free-variable analysis is delegated to
`symtable` rather than hand-rolled over the AST, so nested defs, lambdas and
comprehension scopes are handled correctly. `test_static_leg_detects_a_planted_
break` keeps the whole analysis honest: the real entrypoints are all expected
to pass, which would make a broken analyzer indistinguishable from a clean
tree.
"""

import ast
import os
import subprocess
import sys
import symtable
import textwrap
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Packages run as `python3 -m <pkg>`; their __main__.py is a real entrypoint
# that a root-directory scan cannot see. `make eval` invokes the first of these.
PACKAGE_ENTRYPOINTS = ("eval.evalkit",)


def _iter_entry_files():
    """(display name, absolute path) for every root-level module with a
    module-level `__main__` guard, plus each package `__main__.py`. Discovered
    rather than listed, so a new entry CLI is covered the day it is added."""
    for name in sorted(os.listdir(ROOT)):
        if name.endswith(".py"):
            yield name, os.path.join(ROOT, name)
    for pkg in PACKAGE_ENTRYPOINTS:
        path = os.path.join(ROOT, *pkg.split("."), "__main__.py")
        if os.path.exists(path):
            yield pkg, path


def _entrypoints():
    """(display name, source, tree, guard node) for everything with a
    module-level `if __name__ == "__main__"`."""
    found = []
    for name, path in _iter_entry_files():
        with open(path, encoding="utf-8") as fh:
            src = fh.read()
        try:
            tree = ast.parse(src)
        except SyntaxError:
            continue
        guards = [n for n in tree.body
                  if isinstance(n, ast.If) and "__name__" in ast.dump(n.test)]
        if guards:
            found.append((name, src, tree, guards[-1]))
    return found


def _bound_names(body):
    """{name: lineno} for every binding in a module-level statement list,
    keeping the EARLIEST line -- a name rebound later is bound from its first
    statement. Descends into `if`/`try` so a conditionally-imported name still
    counts; the line recorded is the inner statement's."""
    bound = {}

    def record(name, lineno):
        if name and bound.get(name, lineno + 1) > lineno:
            bound[name] = lineno

    def walk(stmts):
        for node in stmts:
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                for alias in node.names:
                    record((alias.asname or alias.name).split(".")[0], node.lineno)
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                record(node.name, node.lineno)
            elif isinstance(node, (ast.Assign, ast.AugAssign, ast.AnnAssign)):
                targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                for target in targets:
                    for sub in ast.walk(target):
                        if isinstance(sub, ast.Name):
                            record(sub.id, node.lineno)
            elif isinstance(node, (ast.If, ast.Try)):
                for attr in ("body", "orelse", "finalbody"):
                    walk(getattr(node, attr, []) or [])
                for handler in getattr(node, "handlers", []) or []:
                    walk(handler.body)
    walk(body)
    return bound


def _module_scope_reads(table):
    """Names a function symbol table reads from module scope, including from
    every nested scope inside it. symtable does the scope resolution, so a name
    shadowed inside a nested def, lambda or comprehension is correctly NOT
    reported as a module-level read -- the trap a flat AST walk falls into."""
    names = set()
    for sym in table.get_symbols():
        if sym.is_global() and sym.is_referenced():
            names.add(sym.get_name())
    for child in table.get_children():
        names |= _module_scope_reads(child)
    return names


def _function_tables(src, filename):
    """{name: symbol table} for every function defined at module level."""
    top = symtable.symtable(src, filename, "exec")
    return {child.get_name(): child for child in top.get_children()
            if child.get_type() == "function"}


def _delegating_stem(guard, stem):
    """True when the guard both imports its OWN module and calls through it.

    Importing alone is not enough: a guard that does `import conflict_detect`
    but then calls the LOCAL `main` is still broken -- `main` is __main__'s
    copy, whose globals never got the bottom imports -- while the self-import
    silently resolves the cycle, so `--help` passes too. Requiring the call to
    go through the module attribute is what makes the exemption sound."""
    imported = False
    for node in ast.walk(guard):
        if isinstance(node, ast.Import):
            if any(alias.name.split(".")[0] == stem and alias.asname is None
                   for alias in node.names):
                imported = True
    if not imported:
        return False
    for node in ast.walk(guard):
        if (isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name)
                and node.value.id == stem):
            return True
    return False


def unbound_uses(name, src, tree, guard):
    """{used name: (function, binding lineno)} for every module-level name the
    __main__ path reaches that is not yet bound when the guard fires. Empty
    means the entrypoint is sound. Kept module-level so the self-test can drive
    it over a planted break."""
    stem = name.split(".")[0] if name.endswith(".py") is False else name[:-3]
    if _delegating_stem(guard, stem):
        return {}
    funcs = _function_tables(src, name)
    if "main" not in funcs:
        return {}
    bound = _bound_names(tree.body)

    reachable, stack = set(), ["main"]
    while stack:
        fn = stack.pop()
        if fn in reachable or fn not in funcs:
            continue
        reachable.add(fn)
        stack.extend(_module_scope_reads(funcs[fn]) & set(funcs))

    late = {}
    for fn in sorted(reachable):
        for used in sorted(_module_scope_reads(funcs[fn])):
            lineno = bound.get(used)
            if lineno is not None and lineno > guard.lineno:
                late.setdefault(used, (fn, lineno))
    return late


class EntrypointBindingTest(unittest.TestCase):
    """CLASS 1: nothing main() reaches may be bound after the guard fires."""

    def test_entrypoints_are_discovered(self):
        # A collapse to zero would make every other assertion here vacuous.
        names = [name for name, _, _, _ in _entrypoints()]
        self.assertGreaterEqual(len(names), 10, names)
        for expected in ("annotate.py", "conflict_detect.py", "chat.py",
                         "ingest.py", "eval.evalkit"):
            self.assertIn(expected, names)

    def test_no_entrypoint_calls_an_unbound_name(self):
        for name, src, tree, guard in _entrypoints():
            with self.subTest(entrypoint=name):
                late = unbound_uses(name, src, tree, guard)
                stem = name[:-3] if name.endswith(".py") else name
                self.assertEqual(
                    late, {},
                    f"{name}: run as a script, the `__main__` guard at line "
                    f"{guard.lineno} fires before these module-level names are "
                    f"bound, so main() raises NameError: "
                    + ", ".join(f"{n} (used by {fn}(), bound at line {ln})"
                                for n, (fn, ln) in sorted(late.items()))
                    + ". Move the guard below those bindings, or make it "
                    f"`import {stem}` and call `{stem}.main`.")

    def test_static_leg_detects_a_planted_break(self):
        # Every real entrypoint passes, so without this the analysis could rot
        # into a no-op and stay green forever. Each case is the shape of the
        # bug that actually shipped.
        planted = textwrap.dedent("""\
            def main():
                return helper()

            def helper():
                return LATE_NAME

            if __name__ == "__main__":
                main()

            from somewhere import LATE_NAME
        """)
        tree = ast.parse(planted)
        guard = [n for n in tree.body if isinstance(n, ast.If)][-1]
        self.assertEqual(unbound_uses("planted.py", planted, tree, guard),
                         {"LATE_NAME": ("helper", 10)})

        # A self-import that does NOT call through the module is still broken,
        # so the delegation exemption must not fire on it.
        half = planted.replace("    main()", "    import planted\n    main()")
        tree = ast.parse(half)
        guard = [n for n in tree.body if isinstance(n, ast.If)][-1]
        self.assertEqual(list(unbound_uses("planted.py", half, tree, guard)),
                         ["LATE_NAME"])

        # The real shim -- import AND call through it -- is exempt.
        shim = planted.replace("    main()", "    import planted\n    planted.main()")
        tree = ast.parse(shim)
        guard = [n for n in tree.body if isinstance(n, ast.If)][-1]
        self.assertEqual(unbound_uses("planted.py", shim, tree, guard), {})

        # A name shadowed inside a nested scope is not a module-level read;
        # a flat AST walk would wrongly drop the sibling use and pass.
        nested = textwrap.dedent("""\
            def main():
                def inner():
                    LATE_NAME = 1
                    return LATE_NAME
                return inner() + LATE_NAME

            if __name__ == "__main__":
                main()

            from somewhere import LATE_NAME
        """)
        tree = ast.parse(nested)
        guard = [n for n in tree.body if isinstance(n, ast.If)][-1]
        self.assertEqual(list(unbound_uses("planted.py", nested, tree, guard)),
                         ["LATE_NAME"])


class EntrypointStartupTest(unittest.TestCase):
    """CLASS 2: every entry CLI must survive module initialization."""

    def test_every_entrypoint_survives_help(self):
        # --help cannot catch CLASS 1 (argparse exits before main()'s body),
        # but it is the cheapest invocation that proves the module -- and every
        # import cycle it participates in -- initializes at all. stdin is closed
        # so a future non-argparse script fails fast instead of hanging.
        for name, _, _, _ in _entrypoints():
            with self.subTest(entrypoint=name):
                argv = ([sys.executable, "-m", name] if not name.endswith(".py")
                        else [sys.executable, name])
                result = subprocess.run(
                    argv + ["--help"], cwd=ROOT, capture_output=True, text=True,
                    stdin=subprocess.DEVNULL, timeout=60,
                )
                self.assertEqual(
                    result.returncode, 0,
                    f"`{' '.join(argv[1:])} --help` exited {result.returncode}. "
                    f"An ImportError here means the __main__ copy of {name} "
                    f"cannot resolve its own import cycle.\n{result.stderr}")


if __name__ == "__main__":
    unittest.main()
