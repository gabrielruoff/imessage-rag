# Packaging (macOS `.app` for non-technical users)

Goal: ship imessage-rag as a **signed, notarized double-click app** — no
`git clone`, no Homebrew, no `make install`, no visible Ollama — while the LLM
still runs locally.

## Step 1 — DE-RISKED: sqlite-vec survives freezing

The riskiest unknown was whether the sqlite-vec loadable extension works from a
frozen bundle, since python.org / system Python ship a `sqlite3` that **cannot**
load extensions (`enable_load_extension` disabled). Proven working:

```sh
# from repo root, in the .venv used for development
.venv/bin/python3 -m pip install pyinstaller
.venv/bin/python3 -m PyInstaller --noconfirm --clean --onedir \
    --collect-all sqlite_vec --name pkg_smoke packaging/pkg_smoke.py
env -i ./dist/pkg_smoke/pkg_smoke     # empty env: no venv, no python on PATH
```

Result: `frozen: True`, `vec_version: v0.1.9`, KNN query returns correct rows.

**Why it works:** PyInstaller bundles the *build* Python's `_sqlite3.so` **and
its `libsqlite3.dylib`** (Homebrew's, SQLite 3.53.3, extension-enabled, linked
via `@rpath`) — not the system one. So the Homebrew-Python requirement is a
*build-time* dependency only; end users get a self-contained SQLite that can
load `vec0.dylib`. `--collect-all sqlite_vec` places `vec0.dylib` next to the
package so `sqlite_vec.loadable_path()` (which uses `__file__`) resolves inside
the bundle. Bundle size: ~21 MB (Python + SQLite + vec, no models).

**Invariant to keep:** always build the app from **Homebrew Python** (the same
constraint `make install` already documents). Building from system/python.org
Python would freeze an extension-disabled `_sqlite3` and every vec query would
fail at runtime. A CI check should run `packaging/pkg_smoke.py` frozen.

## Remaining steps (not yet built)

2. **Embed + supervise the `ollama` binary** as a subprocess on a private port
   (ship the single Go binary inside the bundle; do NOT use the curl installer
   or `/Applications/Ollama.app` — unmanageable, per CLAUDE.md). Auto-detect
   Apple FM (macOS 26 + Apple Silicon + Apple Intelligence) and skip the model
   download entirely on those machines (`config.is_apple_model` already
   branches).
3. **First-run onboarding UI**: request Full Disk Access (needed to read
   `~/Library/Messages/chat.db`), then a model-download progress bar driven by
   `ollama pull`'s stream. Ship the smallest chat model that meets quality; let
   power users upgrade.
4. **Codesign (Developer ID) + notarize + staple**, wrap in a drag-to-Applications
   `.dmg`, add Sparkle for auto-updates.

## Notes

- `dist/` and `build/` are PyInstaller output — gitignored, never commit.
- The real app has several entry CLIs (`chat.py`, `ingest.py`, …); the shipping
  app needs a small launcher that runs ingest on first launch and opens the chat
  UI. `pkg_smoke.py` only proves the extension seam.
