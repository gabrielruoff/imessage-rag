#!/bin/bash
# Fast single-stream prompt sweep against chat.py.
#
# Usage:
#   eval/fast_eval.sh eval/prompts-smoke.txt [-m model] [-- extra chat.py args]
#
# Prompts file format: one prompt per line as `id|prompt text`; blank lines
# and lines starting with # are skipped.
#
# Prompts run SEQUENTIALLY, never in parallel: Ollama serializes chat
# requests, so parallel eval clients only add queueing latency and evict the
# KV prefix cache between conversations (measured: ~4x slower per prompt and
# worse total throughput -- see docs/eval_report_2026-07-16.md).
#
# Env defaults below shrink per-round-trip prefill for quick smoke sweeps;
# export different values (or a TARGET_DB pointing at the real analysis DB)
# before invoking to override. Transcripts land in
# eval/results/<UTC timestamp>/<id>.txt with duration and exit code.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PROMPTS_FILE="${1:?usage: fast_eval.sh <prompts-file> [-m model]}"
shift

export TOOL_RESULT_MAX_CHARS="${TOOL_RESULT_MAX_CHARS:-12000}"
export MAX_TOOL_ITERATIONS="${MAX_TOOL_ITERATIONS:-8}"
export CHAT_TIMEOUT="${CHAT_TIMEOUT:-600}"
export COLUMNS="${COLUMNS:-200}"

PYTHON="${PYTHON:-$REPO_ROOT/.venv/bin/python3}"
RESULTS_DIR="$REPO_ROOT/eval/results/$(date -u +%Y%m%dT%H%M%SZ)"
mkdir -p "$RESULTS_DIR"
echo "results -> $RESULTS_DIR"

while IFS='|' read -r id prompt; do
    [[ -z "$id" || "$id" == \#* ]] && continue
    out="$RESULTS_DIR/$id.txt"
    {
        echo "PROMPT_ID: $id"
        echo "PROMPT: $prompt"
        echo "START: $(date -u '+%H:%M:%SZ')"
        echo "---"
    } > "$out"
    start=$SECONDS
    rc=0
    "$PYTHON" "$REPO_ROOT/chat.py" -p "$prompt" "$@" >> "$out" 2>&1 || rc=$?
    {
        echo "---"
        echo "EXIT: $rc  DURATION_SECONDS: $((SECONDS - start))"
    } >> "$out"
    echo "$id: ${rc}rc $((SECONDS - start))s"
done < "$PROMPTS_FILE"
