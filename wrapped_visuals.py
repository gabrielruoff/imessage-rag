"""Feature visual builders for the Wrapped report (drift / persona /
wellbeing cards).

Split out of wrapped_html.py: the feature-specific visual builders. Re-exported
from wrapped_html so wrapped_html.render_voiceprint_card (etc.) -- used by
wrapped_sections and the tests -- keeps resolving. Reads the generic renderer
primitives from wrapped_html.
"""

import re


# --- feature: drift ---
def render_window_chips(rows, label_key="contact", days_key="days_until_dormant",
                        status_key="status"):
    """Open reconnection-window chips (RES-66 / D4): one chip per still-open
    cooling tie -- the contact and how many days remain before the drift model
    would count the tie dormant. Reuses the shared .chip styling (no new CSS,
    so this stays a clean EOF append); the window status rides in the tooltip.
    Returns '' when there is nothing to show, so the caller can drop the block."""
    chips = []
    for row in rows:
        name = str(row.get(label_key, "") or "")
        days = row.get(days_key)
        status = str(row.get(status_key, "") or "")
        if isinstance(days, (int, float)) and not isinstance(days, bool):
            unit = "day" if int(days) == 1 else "days"
            body = f"{name} · {int(days)} {unit} left"
        else:
            body = name
        tip = f"{status} reconnection window" if status else ""
        title = f' title="{esc(tip)}"' if tip else ""
        chips.append(f'<span class="chip"{title}>{esc(body)}</span>')
    if not chips:
        return ""
    return '<div class="chips">' + "".join(chips) + "</div>"
# --- end feature: drift ---
# --- feature: persona ---
def _voiceprint_phrases(entries, limit=3):
    """Signature phrases out of an openers/closers list ([{phrase, count}, ...],
    as get_voiceprint returns them): up to `limit` non-empty phrase strings, in
    order. Tolerant of a bare list of strings too."""
    out = []
    for entry in entries or []:
        phrase = entry.get("phrase") if isinstance(entry, dict) else entry
        phrase = str(phrase or "").strip()
        if phrase:
            out.append(phrase)
        if len(out) >= limit:
            break
    return out


def render_voiceprint_card(openers, closers, outgoing=True, reveal=None):
    """A voiceprint's signature phrases as iMessage bubbles: opening phrases then
    closing phrases, outgoing (send-blue) for the reader's own voice or incoming
    for a mystery 'guess who' card. `reveal`, when set, appends a collapsed answer
    -- the ONLY place a mystery texter's name is written, so the card carries no
    name until the reader opens it. Returns None when there are no phrases (so an
    empty voiceprint renders nothing rather than an empty shell)."""
    side = "out" if outgoing else "inc"
    bubbles = [
        f'<figure class="msg {side}"><div class="bubble">'
        f"{format_scalar(phrase)}</div></figure>"
        for phrase in _voiceprint_phrases(openers) + _voiceprint_phrases(closers)
    ]
    if not bubbles:
        return None
    body = '<div class="bubbles">' + "".join(bubbles) + "</div>"
    if reveal:
        body += ("<details><summary>Reveal</summary>"
                 f'<p class="scalar">It’s {esc(reveal)}.</p></details>')
    return body
# --- end feature: persona ---


# --- feature: wellbeing ---
# W4 (RES-68): the shaded-window trajectory chart + code-emitted support footer
# for the opt-in Wrapped wellbeing section. Fully self-contained -- the visuals
# carry their own inline <style> (referencing the page's existing theme tokens,
# so light/dark just work) rather than editing the shared _CSS literal, and the
# support footer is rendered BY CODE from wellbeing.support_block(); the narrator
# never touches it. No network: the only href is a bare help-line URL.

# Scoped styles for the wellbeing visuals. Emitted once, alongside the chart.
# All colors are the page's own CSS custom properties so both themes are handled
# by :root already; the calm neutral palette is deliberate (never alarm-red).
_WELLBEING_STYLE = """<style>
.wb-traj { margin: 0.25rem 0 0.5rem; }
.wb-legend {
  display: flex; flex-wrap: wrap; gap: 0.4rem 1rem;
  font-size: 0.78rem; color: var(--text-muted); margin: 0.1rem 0 0.9rem;
}
.wb-legend .key { display: inline-flex; align-items: center; gap: 0.4rem; }
.wb-legend .swatch { width: 10px; height: 10px; border-radius: 3px; flex: 0 0 10px; }
.wb-legend .swatch.base { background: var(--series-1); }
.wb-legend .swatch.win { background: var(--series-3); }
.wb-legend .swatch.band {
  background: var(--surface-2); border: 1px solid var(--border);
}
.wb-row {
  display: grid;
  grid-template-columns: minmax(6rem, 9rem) 1fr;
  align-items: center; gap: 0.75rem; margin: 0.35rem 0;
}
.wb-label {
  font-size: 0.82rem; color: var(--text-secondary); overflow-wrap: anywhere;
}
.wb-cells {
  display: flex; align-items: flex-end; gap: 2px;
  height: 3.2rem; overflow-x: auto; border-bottom: 1px solid var(--border);
}
.wb-col {
  flex: 1 1 0; min-width: 5px; height: 100%;
  display: flex; align-items: flex-end; justify-content: center;
}
.wb-col.in { background: var(--surface-2); border-radius: 3px 3px 0 0; }
.wb-bar {
  width: 100%; max-width: 12px; height: var(--h, 0%); min-height: 2px;
  border-radius: 3px 3px 0 0; background: var(--series-1);
}
.wb-col.in .wb-bar { background: var(--series-3); }
.wb-axis {
  display: flex; gap: 2px; margin-top: 0.3rem;
  font-size: 0.62rem; color: var(--text-muted);
}
.wb-tick { flex: 1 1 0; min-width: 5px; text-align: center; white-space: nowrap; }
.wb-support {
  margin: 1.1rem 0 0; padding: 0.9rem 1.1rem;
  background: var(--surface-2); border-radius: 12px;
  border-left: 3px solid var(--series-1);
  color: var(--text-secondary); font-size: 0.9rem; line-height: 1.55;
}
.wb-support a { color: var(--series-1); }
</style>"""

_WELLBEING_URL_RE = re.compile(r"(https?://\S+)")


def render_trajectory(markers, window):
    """A within-person trajectory chart with the flagged window shaded.

    `markers` is a list of {"label", "points": [[period 'YYYY-MM', value], ...]}
    rows -- one self-normalized bar strip per marker (bars are value/peak, so a
    marker's own scale never dominates another's). `window` is a
    (start_month, end_month) inclusive 'YYYY-MM' pair; every column whose period
    falls inside it is shaded and drawn in the highlight series color, so the
    departing stretch reads at a glance. Returns None when there is nothing to
    plot. Static heights via the --h custom property (never an inline height:),
    matching the report's animation/print convention; a <details> table view is
    the accessibility fallback."""
    rows = [m for m in (markers or []) if m.get("points")]
    if not rows:
        return None
    start, end = (window or (None, None))
    periods = sorted({p for m in rows for p, _ in m["points"]})
    if not periods:
        return None
    in_window = {p: (start is not None and end is not None and start <= p <= end)
                 for p in periods}
    sparse = len(periods) > 24

    row_html = []
    for m in rows:
        by_period = {p: v for p, v in m["points"]}
        peak = max((abs(float(v or 0)) for v in by_period.values()), default=0)
        cols = []
        for p in periods:
            v = float(by_period.get(p) or 0)
            pct = (abs(v) / peak * 100) if peak > 0 else 0
            tip = f"{display_scalar(p)} · {m['label']}: {format_scalar(by_period.get(p))}"
            cls = "wb-col in" if in_window[p] else "wb-col"
            cols.append(
                f'<span class="{cls}" title="{esc(tip)}">'
                f'<span class="wb-bar" style="--h:{pct:.1f}%"></span></span>'
            )
        row_html.append(
            f'<div class="wb-row"><span class="wb-label">{esc(m["label"])}</span>'
            f'<span class="wb-cells">{"".join(cols)}</span></div>'
        )

    ticks = []
    for p in periods:
        show = (not sparse) or str(p).endswith("-01")
        ticks.append(f'<span class="wb-tick">{esc(display_scalar(p)) if show else ""}</span>')
    axis = ('<div class="wb-row"><span class="wb-label"></span>'
            f'<div class="wb-axis">{"".join(ticks)}</div></div>')

    legend = (
        '<div class="wb-legend">'
        '<span class="key"><span class="swatch base"></span>Your usual range</span>'
        '<span class="key"><span class="swatch win"></span>The flagged stretch</span>'
        '<span class="key"><span class="swatch band"></span>Shaded window</span>'
        "</div>"
    )

    table_rows = [
        {"period": p, **{m["label"]: dict(m["points"]).get(p) for m in rows},
         "in flagged window": in_window[p]}
        for p in periods
    ]
    table = ("<details><summary>Table view</summary>"
             + _rows_table(table_rows, _columns(table_rows)) + "</details>")

    return ('<div class="wb-traj">' + legend
            + "".join(row_html) + axis + table + "</div>")


def render_support_block(text):
    """Render the fixed, code-emitted support footer into HTML. The text comes
    from wellbeing.support_block() verbatim -- this only escapes it, linkifies
    the bare help-line URL, and preserves its line breaks. The narrator never
    produces this block, so it can never be omitted or reworded."""
    if not text:
        return ""
    safe = esc(text)
    safe = _WELLBEING_URL_RE.sub(
        r'<a href="\1" rel="noopener noreferrer">\1</a>', safe)
    return ('<aside class="wb-support" role="note">'
            + safe.replace("\n", "<br>") + "</aside>")


def render_wellbeing(markers, window, support_text):
    """Assemble the wellbeing section body: scoped styles, the shaded-window
    trajectory chart, and the code-emitted support footer. The footer is
    appended unconditionally whenever this renders, so a flagged window always
    ships its support text."""
    chart = render_trajectory(markers, window) or ""
    return _WELLBEING_STYLE + chart + render_support_block(support_text)
# --- end feature: wellbeing ---

# Bottom import: this module is one half of a cycle -- wrapped_html re-exports the
# names below, so importing it from the TOP here makes wrapped_visuals unimportable on
# its own (the parent would run its own re-export against a still-empty
# wrapped_visuals). Down here every name above is already bound, so the cycle resolves
# whichever module a caller reaches first. Only used inside functions, so the
# late binding costs nothing.
from wrapped_html import (  # noqa: E402
    _columns,
    _rows_table,
    display_scalar,
    esc,
    format_scalar,
)
