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

Where the walk starts and where "too late" begins are both READ OFF THE GUARD
rather than assumed. The entry point is whatever module-level callable the guard
names -- `main()`, `run_cli(main)`, `sys.exit(main(...))`, or a future `cli()` --
because assuming the name `main` turns a renamed entry point into a silent pass
rather than a failure. The cutoff is the line that hands control, not the guard's
`if` line, so a name bound inside the guard ABOVE the handoff (the shim's own
`import <self>`) counts as bound, which it is. Every module-level `__name__ ==
"__main__"` guard is analyzed, not just the last: they all run, in order, and a
stale broken one above an appended shim still decides what the script does.

Nothing is exempt from the walk on the strength of a self-import. A guard that
names a local callable is analyzed even when it also reaches through its own
module, because the local name is the thing that runs; `_delegates_to_own_module`
survives only to tell a sound empty result (the shim, which hands off through the
fully-initialized copy) from a vacuous one.

Scope notes, all deliberate -- the honest edges of the static leg:
  - Reachability follows direct name references from the guard, through module
    level functions, class bodies (a named class reaches its methods) and
    decorator lists. A callable reached ONLY through a dict/registry, a string,
    or getattr is not walked.
  - Binding recognition covers import / def / class / assignment, descending
    into module-level `if` and `try`. It does NOT model `del`, an optional
    import that fails at runtime, a module-level walrus, `global` rebinding from
    inside a function, or bindings in a module-level `for`/`while`/`with` body.
    An unrecognized binding makes a name invisible rather than late, so the leg
    under-reports rather than false-alarms.
  - Free-variable analysis is delegated to `symtable`, not hand-rolled over the
    AST, so nested defs, lambdas and comprehension scopes resolve correctly.
  - Discovery sees root-level `*.py` plus the packages listed in
    PACKAGE_ENTRYPOINTS. A CLI in a subdirectory, or a console_scripts entry
    point, is not covered until it is listed.
  - The `--help` leg proves initialization only as far as the guard: argparse
    exits there, so module-level code BELOW the guard never runs on that path.

Three tests exist to keep the analysis from rotting into a permanently green
no-op, which is the standing risk when every real entrypoint is expected to
pass: `test_static_leg_detects_a_planted_break` drives it over synthetic breaks,
`test_guard_selection_takes_every_real_main_guard` pins which `if` blocks count,
and `test_every_entrypoint_exposes_an_entry_function` refuses the one input shape
the analyzer cannot say anything about.
"""

import ast
import os
import subprocess
import symtable
import sys
import textwrap
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Packages run as `python3 -m <pkg>`; their __main__.py is a real entrypoint
# that a root-directory scan cannot see. `make eval` invokes the first of these.
PACKAGE_ENTRYPOINTS = ("eval.evalkit",)


def _module_name(display):
    """The dotted name a human would `import`: `annotate.py` -> `annotate`, and
    a package entrypoint (`eval.evalkit`) is already one."""
    return display[:-3] if display.endswith(".py") else display


def _root_package(display):
    """First component of the import name -- the only part `import a.b` binds,
    and so the part a self-import in the guard has to match."""
    return _module_name(display).split(".")[0]


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


def _is_main_guard(node):
    """A module-level `if __name__ == "__main__"`, and only that.

    The `==` is load-bearing. A trailing `if __name__ != "__main__"` block is
    dead on the script path, so treating it as the guard would push the cutoff
    past the real handoff and hide every binding in between."""
    if not isinstance(node, ast.If) or not isinstance(node.test, ast.Compare):
        return False
    if len(node.test.ops) != 1 or not isinstance(node.test.ops[0], ast.Eq):
        return False
    sides = (node.test.left, node.test.comparators[0])
    return (any(isinstance(s, ast.Name) and s.id == "__name__" for s in sides)
            and any(isinstance(s, ast.Constant) and s.value == "__main__" for s in sides))


def _entrypoints():
    """(display name, source, tree, [guard nodes]) for everything with at least
    one module-level `if __name__ == "__main__"`.

    EVERY such guard is returned, not just the last. They all run, in order, and
    the first one to hand control decides what the script does -- so appending a
    delegating shim below a stale broken guard has to stay a failure, not become
    a pass because the analyzer only looked at the bottom one."""
    found = []
    for name, path in _iter_entry_files():
        with open(path, encoding="utf-8") as fh:
            src = fh.read()
        try:
            tree = ast.parse(src)
        except SyntaxError:
            continue
        guards = [n for n in tree.body if _is_main_guard(n)]
        if guards:
            found.append((name, src, tree, guards))
    return found


def _bound_names(body):
    """{name: lineno} for every binding in a module-level statement list,
    keeping the EARLIEST line -- a name rebound later is bound from its first
    statement. Descends into `if`/`try` so a conditionally-imported name still
    counts; the line recorded is the inner statement's. That descent covers the
    `__main__` guard itself, which is why the caller compares against the
    handoff line rather than the guard's."""
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


def _callable_tables(src, filename):
    """{name: symbol table} for every function AND class defined at module level.

    Classes are in here because naming one reaches every method it defines, and
    a class table is a child of the MODULE table -- never of a function's -- so
    dropping it made class bodies a region the walk could not enter from any
    entry function. `_module_scope_reads` already recurses through a table's
    children, so a class table yields its methods' module-scope reads for free.
    A function whose late-bound read sits in a method rather than in a plain
    function is the same bug with a different enclosing scope."""
    top = symtable.symtable(src, filename, "exec")
    return {child.get_name(): child for child in top.get_children()
            if child.get_type() in ("function", "class")}


def _decorator_names(tree):
    """{decorated name: {names in its decorator list}}.

    A decorator wraps the call, so the wrapper's body runs on the entry path --
    but symtable evaluates the decorator expression in MODULE scope, outside the
    decorated function's own table, so the BFS would never reach it."""
    out = {}
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names = {sub.id for dec in node.decorator_list
                     for sub in ast.walk(dec) if isinstance(sub, ast.Name)}
            if names:
                out.setdefault(node.name, set()).update(names)
    return out


def _delegates_to_own_module(guard, display):
    """True when the guard imports its OWN module and reaches through it.

    This is NOT an exemption from analysis -- it used to be, and that made it a
    total bypass: a guard whose self-import and attribute access were cosmetic
    (`import parent; print(parent.__name__); main()`) called the LOCAL `main`,
    whose globals never got the bottom imports, and every leg still reported
    clean. Analysis is now gated on whether the guard names a local callable, so
    this answers a narrower question: when the guard names NO local callable, is
    that because it delegates (sound -- `main` is read off the fully-initialized
    copy) or because we simply cannot see the entry point (vacuous)?

    Matching is on the ROOT package, since that is the only name `import a.b`
    binds and `a.b.main()` nests the root as the innermost Name."""
    root = _root_package(display)
    imported = any(
        isinstance(node, ast.Import)
        and any(alias.name.split(".")[0] == root and alias.asname is None
                for alias in node.names)
        for node in ast.walk(guard))
    if not imported:
        return False
    return any(isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name)
               and node.value.id == root
               for node in ast.walk(guard))


def entry_functions(guard, funcs):
    """Module-level callables the `__main__` guard hands control to.

    Read off the guard rather than assumed to be named `main`. A guard may call
    it (`main()`), hand it to a runner as a VALUE (`run_cli(main)`,
    `db.run_cli(main)`) or wrap it (`sys.exit(main(sys.argv[1:]))`), so the test
    is "does the guard name a module-level callable", not "does it call one".
    Naming one is handing control to it, and over-inclusion only widens the
    walk -- anything the guard names really does run. A guard that names one at
    all is analyzed, even if it ALSO self-imports: the local name is the thing
    that runs, and a self-import beside it does not make it safe."""
    return {node.id for node in ast.walk(guard)
            if isinstance(node, ast.Name) and node.id in funcs}


def handoff_lineno(guard, entries):
    """The line at which the guard hands control -- the cutoff for "bound yet?".

    Anchoring on the guard's own `if` line instead would call a name bound
    INSIDE the guard above the handoff (a self-import, a `logging.basicConfig()`
    call) late when it is already bound by the time control transfers, and would
    miss one bound below the handoff."""
    lines = [node.lineno for node in ast.walk(guard)
             if isinstance(node, ast.Name) and node.id in entries]
    return min(lines) if lines else guard.lineno


def unbound_uses(name, src, tree, guard):
    """{used name: (function, binding lineno)} for every module-level name the
    __main__ path reaches that is not yet bound when the guard hands control.
    Empty means the entrypoint is sound. Kept module-level so the self-test can
    drive it over a planted break.

    Returns empty only when the guard names no module-level callable at all --
    either because it delegates through its own module (sound) or because the
    entry point is invisible to us (vacuous).
    `test_every_entrypoint_exposes_an_entry_function` is what tells those two
    apart; nothing here is exempt on the strength of a self-import alone."""
    funcs = _callable_tables(src, name)
    entries = entry_functions(guard, funcs)
    if not entries:
        return {}
    bound = _bound_names(tree.body)
    cutoff = handoff_lineno(guard, entries)
    decorators = _decorator_names(tree)

    reachable, stack = set(), sorted(entries)
    while stack:
        fn = stack.pop()
        if fn in reachable or fn not in funcs:
            continue
        reachable.add(fn)
        reached = _module_scope_reads(funcs[fn]) | decorators.get(fn, set())
        stack.extend(reached & set(funcs))

    late = {}
    for fn in sorted(reachable):
        for used in sorted(_module_scope_reads(funcs[fn])):
            lineno = bound.get(used)
            if lineno is not None and lineno > cutoff:
                late.setdefault(used, (fn, lineno))
    return late


class EntrypointBindingTest(unittest.TestCase):
    """CLASS 1: nothing the guard reaches may be bound after it hands control."""

    def test_entrypoints_are_discovered(self):
        # A collapse to zero would make every other assertion here vacuous.
        names = [name for name, _, _, _ in _entrypoints()]
        self.assertGreaterEqual(len(names), 10, names)
        for expected in ("annotate.py", "conflict_detect.py", "chat.py",
                         "ingest.py", "eval.evalkit"):
            self.assertIn(expected, names)

    def test_every_entrypoint_exposes_an_entry_function(self):
        # Discovery being non-empty is not enough: unbound_uses() returns {}
        # when it finds nothing to walk from, which reads exactly like "clean".
        # So every guard must either name a module-level callable -- which is
        # then really analyzed -- or delegate through its own module, where
        # binding order is provably irrelevant. A guard that does neither hands
        # control somewhere we cannot follow, and this is where that becomes
        # visible instead of silent.
        for name, src, tree, guards in _entrypoints():
            for guard in guards:
                with self.subTest(entrypoint=name, guard_line=guard.lineno):
                    if entry_functions(guard, _callable_tables(src, name)):
                        continue
                    module = _module_name(name)
                    self.assertTrue(
                        _delegates_to_own_module(guard, name),
                        f"{name}: the `__main__` guard at line {guard.lineno} "
                        f"names no module-level callable and does not delegate "
                        f"through `{module}`, so the CLASS 1 walk starts nowhere "
                        f"and this guard passes vacuously. Call the entry point "
                        f"by name in the guard, or delegate with `import "
                        f"{module}` and `{module}.<entry>`.")

    def test_no_entrypoint_calls_an_unbound_name(self):
        for name, src, tree, guards in _entrypoints():
            for guard in guards:
                with self.subTest(entrypoint=name, guard_line=guard.lineno):
                    late = unbound_uses(name, src, tree, guard)
                    module = _module_name(name)
                    entries = entry_functions(guard, _callable_tables(src, name))
                    self.assertEqual(
                        late, {},
                        f"{name}: run as a script, the `__main__` guard hands "
                        f"control at line {handoff_lineno(guard, entries)}, "
                        f"before these module-level names are bound, so "
                        f"{'/'.join(sorted(entries))} raises NameError: "
                        + ", ".join(f"{n} (used by {fn}(), bound at line {ln})"
                                    for n, (fn, ln) in sorted(late.items()))
                        + ". Move the guard below those bindings, or make it "
                        f"`import {module}` and call `{module}.main`.")

    def test_static_leg_detects_a_planted_break(self):
        # Every real entrypoint passes, so without this the analysis could rot
        # into a no-op and stay green forever. Each case is a shape the analyzer
        # has to get right, starting with the bug that actually shipped.
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

        # The entry function need not be called `main`. Hardcoding that name
        # made a renamed entry point a silent pass rather than a failure.
        renamed = planted.replace("def main():", "def cli():").replace(
            "    main()", "    cli()")
        tree = ast.parse(renamed)
        guard = [n for n in tree.body if isinstance(n, ast.If)][-1]
        self.assertEqual(unbound_uses("planted.py", renamed, tree, guard),
                         {"LATE_NAME": ("helper", 10)})

        # The runner indirection most real guards here use: `main` is an
        # ARGUMENT, never a call target, so "what does the guard call" is the
        # wrong question -- "what does it name" is the right one.
        runner = planted.replace("    main()", "    run_cli(main)")
        tree = ast.parse(runner)
        guard = [n for n in tree.body if isinstance(n, ast.If)][-1]
        self.assertEqual(unbound_uses("planted.py", runner, tree, guard),
                         {"LATE_NAME": ("helper", 10)})

        # A self-import that does NOT call through the module is still broken,
        # so naming a local callable has to win over any self-import beside it.
        half = planted.replace("    main()", "    import planted\n    main()")
        tree = ast.parse(half)
        guard = [n for n in tree.body if isinstance(n, ast.If)][-1]
        self.assertEqual(list(unbound_uses("planted.py", half, tree, guard)),
                         ["LATE_NAME"])

        # Nor may a COSMETIC reach through the self-module buy an exemption.
        # This shape satisfies _delegates_to_own_module in full -- self-import
        # plus an attribute access -- while the thing that actually runs is the
        # local `main`. It used to bypass the walk entirely.
        cosmetic = planted.replace(
            "    main()",
            '    import planted\n    print("banner", planted.__name__)\n    main()')
        tree = ast.parse(cosmetic)
        guard = [n for n in tree.body if isinstance(n, ast.If)][-1]
        self.assertTrue(_delegates_to_own_module(guard, "planted.py"))
        self.assertEqual(list(unbound_uses("planted.py", cosmetic, tree, guard)),
                         ["LATE_NAME"])

        # The real shim -- names no local callable, reaches through the module --
        # has nothing to walk, and the vacuity test accepts it on that basis.
        shim = planted.replace("    main()", "    import planted\n    planted.main()")
        tree = ast.parse(shim)
        guard = [n for n in tree.body if isinstance(n, ast.If)][-1]
        self.assertEqual(unbound_uses("planted.py", shim, tree, guard), {})
        self.assertTrue(_delegates_to_own_module(guard, "planted.py"))

        # Bindings inside the guard straddle the handoff: EARLY is bound by the
        # time control transfers, LATE is not. Anchoring on the guard's `if`
        # line called both late; anchoring on the guard's last line would call
        # both bound.
        straddle = textwrap.dedent("""\
            def main():
                return EARLY + LATE

            if __name__ == "__main__":
                EARLY = 1
                main()
                LATE = 2
        """)
        tree = ast.parse(straddle)
        guard = [n for n in tree.body if isinstance(n, ast.If)][-1]
        self.assertEqual(unbound_uses("planted.py", straddle, tree, guard),
                         {"LATE": ("main", 7)})

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

        # A class body is not a place the late read can hide. A module-level
        # class table hangs off the MODULE table, never off a function's, so
        # keeping only `function` children made every method unreachable from
        # any entry point -- while chat.py really does read module globals from
        # inside its producer classes.
        in_class = textwrap.dedent("""\
            class Runner:
                def run(self):
                    return LATE_NAME

            def main():
                return Runner().run()

            if __name__ == "__main__":
                main()

            from somewhere import LATE_NAME
        """)
        tree = ast.parse(in_class)
        guard = [n for n in tree.body if isinstance(n, ast.If)][-1]
        self.assertEqual(list(unbound_uses("planted.py", in_class, tree, guard)),
                         ["LATE_NAME"])

        # Same for a decorator: symtable evaluates the decorator expression in
        # module scope, outside main()'s own table, so the wrapper's body is
        # only reachable by reading the decorator list off the AST.
        decorated = textwrap.dedent("""\
            def trace(fn):
                def wrapper(*a):
                    return LATE_NAME(fn(*a))
                return wrapper

            @trace
            def main():
                return 0

            if __name__ == "__main__":
                main()

            from somewhere import LATE_NAME
        """)
        tree = ast.parse(decorated)
        guard = [n for n in tree.body if isinstance(n, ast.If)][-1]
        self.assertEqual(list(unbound_uses("planted.py", decorated, tree, guard)),
                         ["LATE_NAME"])

    def test_guard_selection_takes_every_real_main_guard(self):
        # Appending a shim without deleting the stale guard above it is the
        # plausible way to regress this, and looking only at the LAST guard made
        # it pass: the stale one still fires first and still dies.
        stale = textwrap.dedent("""\
            def main():
                return LATE_NAME

            if __name__ == "__main__":
                main()

            from somewhere import LATE_NAME

            if __name__ == "__main__":
                import planted
                planted.main()
        """)
        tree = ast.parse(stale)
        guards = [n for n in tree.body if _is_main_guard(n)]
        self.assertEqual([g.lineno for g in guards], [4, 9])
        self.assertEqual(list(unbound_uses("planted.py", stale, tree, guards[0])),
                         ["LATE_NAME"])
        self.assertEqual(unbound_uses("planted.py", stale, tree, guards[1]), {})

        # `!=` is not a guard: its body is dead on the script path, and treating
        # it as one would push the cutoff past the real handoff.
        inverted = textwrap.dedent("""\
            def main():
                return LATE_NAME

            if __name__ == "__main__":
                main()

            from somewhere import LATE_NAME

            if __name__ != "__main__":
                pass
        """)
        tree = ast.parse(inverted)
        guards = [n for n in tree.body if _is_main_guard(n)]
        self.assertEqual([g.lineno for g in guards], [4])
        self.assertEqual(list(unbound_uses("planted.py", inverted, tree, guards[0])),
                         ["LATE_NAME"])

    def test_vacuity_check_rejects_a_guard_that_names_no_local_function(self):
        # The companion to the planted-break test: prove the vacuity guard
        # fires, or it is itself a no-op. `unittest.main()` names nothing local,
        # so unbound_uses() reports clean on a file it never analyzed.
        opaque = textwrap.dedent("""\
            def main():
                return LATE_NAME

            if __name__ == "__main__":
                unittest.main()

            from somewhere import LATE_NAME
        """)
        tree = ast.parse(opaque)
        guard = [n for n in tree.body if isinstance(n, ast.If)][-1]
        funcs = _callable_tables(opaque, "planted.py")
        self.assertEqual(entry_functions(guard, funcs), set())
        self.assertEqual(unbound_uses("planted.py", opaque, tree, guard), {})


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
