"""Generic HTML renderers for the Wrapped report: value formatting, stat
tiles, tables, charts and the feature-agnostic visual builders.

Split out of wrapped_html.py, which keeps the page shell (inline CSS/JS,
render_page/render_hero) -- and re-exports everything here, so
`wrapped_html.render_value` and friends keep resolving for wrapped_sections,
wrapped_visuals and the tests.

The renderer is deliberately structural, not schema-aware: it dispatches on
the *shape* of a value (scalar / list-of-dicts / dict-of-dicts), so a new
analytics tool's stats render sensibly without touching this module. The two
schema-ish concessions are `_HIDDEN_KEYS` (database plumbing -- guids, row
ids -- that is noise in a keepsake report) and the period-series chart
detection.

Colors come from the validated default data-viz palette: categorical slots 1
(blue) and 2 (green) for the sent/received series, and the blue sequential
ramp for the heatmap. Dark mode is a separate set of steps from the same
ramps, not an automatic flip -- and the sequential ramp reverses direction so
"more" is always the step furthest from the surface.
"""

import bisect
import html
import math
import re

# Database plumbing: identifiers and flags that carry no meaning for a reader.
# Dropped from every table and key-value block at any depth.
_HIDDEN_KEYS = frozenset({
    "guid", "id", "chat_id", "chat_ids", "handle_id", "handle_ids",
    "contact_id", "contact_ids", "is_from_me", "has_attachment",
    "decode_failed", "item_type", "chunk_id", "is_me", "available",
})

# Word-level label fixups applied when humanizing a snake_case key.
_LABEL_WORDS = {
    "n": "count", "lsm": "LSM", "gini": "Gini", "pct": "%", "id": "ID",
    "ngram": "n-gram", "nth": "Nth", "utc": "UTC", "tz": "timezone",
}

# Whole-key label overrides (lowercased key -> display label) for keys whose
# word-by-word humanization reads as jargon in a keepsake report.
_LABEL_OVERRIDES = {
    "previous_rank": "Last year",
    "median_seconds": "Median",
    "p90_seconds": "90th percentile",
    "median_duration_seconds": "Typical length",
    "typical_length_seconds": "Typical length",
    "ended_by_me_share": "Ended by me",
    "we_per_100": "Says “we” per 100",
    "question_per_100": "Questions per 100",
    "laugh_per_100": "Laughs per 100",
    "laughs_per_100_my_messages": "Laughs per 100 of mine",
    "lsm_score": "Score (0–1)",
    "style_match": "Style match (LSM)",
    "balance_gini": "Balance (Gini, 0 = even)",
    "volume_series": "Messages by month",
    "average_tone": "Average tone (−1 to +1)",
    "share_of_messages": "Share of all messages",
    "messages_all_time": "All-time messages",
    "late_night_share": "Of all messages",
    "longest_sitting": "Longest sitting",
    "it_ran": "It ran",
    "with_whom": "With",
    "on": "On",
    "days_until": "Days away",
    "who_replies_to_whom": "Who replies to whom",
    "replies_to": "Replies to",
}

_MONTHS = ("Jan", "Feb", "Mar", "Apr", "May", "Jun",
           "Jul", "Aug", "Sep", "Oct", "Nov", "Dec")

# 'YYYY-MM-DD' or a full ISO timestamp (optionally tz-suffixed); group 1-3 are
# the date parts. Timestamps render date-only: clock detail is plumbing here.
_ISO_DATE_RE = re.compile(r"^(\d{4})-(\d{2})-(\d{2})([T ].*)?$")
_ISO_MONTH_RE = re.compile(r"^(\d{4})-(\d{2})$")

# Keys whose name is packaging rather than meaning: a heading reading "ROWS"
# above a table tells a reader nothing the table doesn't. Suppressed only when
# the value is a list -- see the unwrap rule in render_value.
_CONTAINER_KEYS = frozenset({"rows", "top", "series", "items"})

_WEEKDAYS = ("Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun")

# Sequential blue ramp, low -> high. Dark mode reverses it (see _CSS) so the
# hottest cell is always the step furthest from the surface.
_SEQ_STEPS = 7


def esc(value):
    return html.escape(str(value), quote=True)


def humanize(key):
    """'n_messages' -> 'Messages'; 'group_lsm' -> 'Group LSM'. Whole-key
    overrides win; a leading 'n_' and a trailing '_seconds' are dropped (the
    value formatter renders *_seconds as a duration, so the unit would lie)."""
    override = _LABEL_OVERRIDES.get(str(key).lower())
    if override is not None:
        return override
    words = [w for w in str(key).replace("-", " ").replace("_", " ").split() if w]
    if not words:
        return ""
    if words[0].lower() == "n" and len(words) > 1:
        words = words[1:]
    if words[-1].lower() == "seconds" and len(words) > 1:
        words = words[:-1]
    out = [_LABEL_WORDS.get(w.lower(), w) for w in words]
    first = out[0]
    out[0] = first if first[:1].isupper() else first[:1].upper() + first[1:]
    return " ".join(out)


def _is_scalar(value):
    return value is None or isinstance(value, (str, int, float, bool))


def _format_float(value, key=None):
    if math.isnan(value) or math.isinf(value):
        return "—"
    name = str(key or "")
    if "share" in name.lower() and 0 <= value <= 1:
        return f"{value * 100:.1f}%"
    if value and abs(value) < 0.01:
        return f"{value:.3g}"
    text = f"{value:,.2f}".rstrip("0").rstrip(".")
    return text or "0"


def format_hour(hour):
    """0 -> '12 AM', 13 -> '1 PM'."""
    hour = int(hour) % 24
    if hour == 0:
        return "12 AM"
    if hour < 12:
        return f"{hour} AM"
    if hour == 12:
        return "12 PM"
    return f"{hour - 12} PM"


def format_duration(seconds):
    """Seconds -> '42 s' / '27 min' / '5 h 38 min' / '1.4 days'."""
    try:
        seconds = float(seconds)
    except (TypeError, ValueError):
        return "—"
    if math.isnan(seconds) or math.isinf(seconds) or seconds < 0:
        return "—"
    if seconds < 60:
        return f"{seconds:.0f} s"
    if seconds < 3600:
        return f"{seconds / 60:.0f} min"
    if seconds < 86400:
        hours, mins = int(seconds // 3600), int(round(seconds % 3600 / 60))
        if mins == 60:
            hours, mins = hours + 1, 0
        return f"{hours} h {mins} min" if mins else f"{hours} h"
    days = f"{seconds / 86400:.1f}".rstrip("0").rstrip(".")
    return f"{days} days"


def display_scalar(value, key=None):
    """One scalar as plain display text (NOT escaped): thousands separators,
    *_seconds keys as durations, share floats as percentages, ISO dates and
    timestamps as 'Feb 3, 2025', 'YYYY-MM' as 'Feb 2025'. Shared by the HTML
    renderer and the narrator payload so both show identical values."""
    if value is None or value == "":
        return "—"
    if isinstance(value, bool):
        return "Yes" if value else "No"
    if isinstance(value, (int, float)) and str(key or "").endswith("seconds"):
        return format_duration(value)
    if isinstance(value, int):
        return f"{value:,}"
    if isinstance(value, float):
        return _format_float(value, key)
    # ￼ is the attachment placeholder chat.db leaves in message text.
    text = str(value).replace("￼", "").strip()
    m = _ISO_DATE_RE.match(text)
    if m and 1 <= int(m[2]) <= 12 and 1 <= int(m[3]) <= 31:
        return f"{_MONTHS[int(m[2]) - 1]} {int(m[3])}, {m[1]}"
    m = _ISO_MONTH_RE.match(text)
    if m and 1 <= int(m[2]) <= 12:
        return f"{_MONTHS[int(m[2]) - 1]} {m[1]}"
    return text or "—"


def format_scalar(value, key=None):
    """One scalar as display text (already escaped)."""
    return esc(display_scalar(value, key))


def _visible_items(mapping):
    return [(k, v) for k, v in mapping.items() if k not in _HIDDEN_KEYS]


def _unavailable_note(mapping):
    """A tool's {'available': False, 'note': ...} dict -> its note, else None."""
    if mapping.get("available") is False:
        return mapping.get("note") or "not available"
    return None


def _inline(value, key=None):
    """Compact one-line rendering for a value nested inside a table cell."""
    if _is_scalar(value):
        return format_scalar(value, key)
    if isinstance(value, dict):
        note = _unavailable_note(value)
        if note is not None:
            return esc(note)
        parts = [f"{humanize(k)} {_inline(v, k)}" for k, v in _visible_items(value)]
        return ", ".join(parts) or "—"
    if isinstance(value, list):
        if not value:
            return "—"
        return ", ".join(_inline(v) for v in value)
    return esc(value)


def _heading(depth, text):
    level = min(depth + 3, 6)
    return f'<h{level} class="block-title">{esc(text)}</h{level}>'


def _tiles(items):
    cells = []
    for key, value in items:
        cells.append(
            '<div class="tile">'
            f'<div class="tile-value">{format_scalar(value, key)}</div>'
            f'<div class="tile-label">{esc(humanize(key))}</div>'
            "</div>"
        )
    return '<div class="tiles">' + "".join(cells) + "</div>"


def _kv_table(items):
    rows = []
    for key, value in items:
        rows.append(
            f'<tr><th scope="row">{esc(humanize(key))}</th>'
            f"<td>{format_scalar(value, key)}</td></tr>"
        )
    return '<table class="kv"><tbody>' + "".join(rows) + "</tbody></table>"


def _columns(rows):
    """Ordered union of the visible keys across a list of dicts."""
    cols = []
    for row in rows:
        for key in row:
            if key not in _HIDDEN_KEYS and key not in cols:
                cols.append(key)
    return cols


def _numeric_series_keys(rows, cols):
    """The chartable columns of a period series: numeric in every row."""
    keys = []
    for col in cols:
        if col == "period":
            continue
        values = [r.get(col) for r in rows]
        if all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in values):
            keys.append(col)
    return keys


def _rows_table(rows, cols, css_class="grid"):
    head = "".join(f"<th>{esc(humanize(c))}</th>" for c in cols)
    body = []
    for row in rows:
        cells = "".join(f"<td>{_inline(row.get(c), c)}</td>" for c in cols)
        body.append(f"<tr>{cells}</tr>")
    return (
        f'<div class="table-wrap"><table class="{css_class}">'
        f"<thead><tr>{head}</tr></thead><tbody>{''.join(body)}</tbody></table></div>"
    )


def render_series_chart(rows, label=None):
    """Grouped bar chart over a [{period, <numeric>...}] series, with a
    legend, per-bar hover tooltips, and a collapsed table view. Returns None
    when the series has nothing chartable."""
    cols = _columns(rows)
    series_keys = _numeric_series_keys(rows, cols)[:2]
    if not series_keys or len(rows) < 2:
        return None
    peak = max((float(r.get(k) or 0) for r in rows for k in series_keys), default=0)
    if peak <= 0:
        return None
    # Past ~2 years of monthly bars the ticks collide; label only the
    # January columns so the axis stays a readable year scale.
    sparse_ticks = len(rows) > 24
    columns = []
    for row in rows:
        bars = []
        raw_period = str(row.get("period") or "")
        period = display_scalar(raw_period)
        for i, key in enumerate(series_keys, 1):
            value = float(row.get(key) or 0)
            pct = value / peak * 100
            tip = f"{period} · {humanize(key)}: {format_scalar(row.get(key), key)}"
            bars.append(
                f'<div class="bar s{i}" style="--h:{pct:.2f}%" title="{esc(tip)}"></div>'
            )
        tick = "" if sparse_ticks and not raw_period.endswith("-01") else esc(period)
        columns.append(
            f'<div class="bar-col"><div class="bar-group">{"".join(bars)}</div>'
            f'<div class="bar-tick">{tick}</div></div>'
        )
    legend = "".join(
        f'<span class="key"><span class="swatch s{i}"></span>{esc(humanize(k))}</span>'
        for i, k in enumerate(series_keys, 1)
    )
    caption = f'<figcaption>{esc(label)}</figcaption>' if label else ""
    return (
        '<figure class="chart">'
        + caption
        + f'<div class="legend">{legend}</div>'
        + f'<div class="bars">{"".join(columns)}</div>'
        + "<details><summary>Table view</summary>"
        + _rows_table(rows, cols)
        + "</details></figure>"
    )


def heatmap_grid(heatmap):
    """Pull the 7x24 weekday-major count grid out of a raw grid or a dict
    wrapping one. Returns None when no grid is recognizable."""
    grid = heatmap
    if isinstance(heatmap, dict):
        grid = next(
            (heatmap[k] for k in ("grid", "counts", "cells", "heatmap") if k in heatmap),
            None,
        )
    if not grid:
        return None
    try:
        max(v for row in grid for v in row)
    except (TypeError, ValueError):
        return None
    return grid


def _quantile_cuts(values):
    """Bin edges splitting the nonzero values into _SEQ_STEPS equal-count
    bands. Message counts per hour are heavily skewed -- one 3am outlier
    sets the peak and a linear ramp then paints every other cell the same
    mid step, hiding the pattern the heatmap exists to show."""
    ordered = sorted(values)
    return [ordered[len(ordered) * i // _SEQ_STEPS] for i in range(1, _SEQ_STEPS)]


def render_heatmap(heatmap):
    """A weekday x hour count grid as an HTML heatmap on the blue sequential
    ramp, shaded by quantile band, with per-cell hover tooltips carrying the
    exact count and a low->high legend. None when unrenderable."""
    grid = heatmap_grid(heatmap)
    if grid is None:
        return None
    nonzero = [v for row in grid for v in row if v]
    cuts = _quantile_cuts(nonzero) if nonzero else []
    tick_labels = {0: "12a", 6: "6a", 12: "12p", 18: "6p"}
    ticks = "".join(
        f'<span class="hm-tick">{tick_labels.get(h, "")}</span>' for h in range(24)
    )
    rows = []
    for day, row in zip(_WEEKDAYS, grid):
        cells = []
        for hour, value in enumerate(row):
            step = 0 if not value else min(_SEQ_STEPS, bisect.bisect_right(cuts, value) + 1)
            tip = f"{day} {format_hour(hour)} · {format_scalar(value)} messages"
            cells.append(f'<span class="hm-cell q{step}" title="{esc(tip)}"></span>')
        rows.append(
            f'<div class="hm-row"><span class="hm-day">{day}</span>'
            f'<span class="hm-cells">{"".join(cells)}</span></div>'
        )
    scale = "".join(f'<span class="hm-cell q{i}"></span>' for i in range(_SEQ_STEPS + 1))
    return (
        '<figure class="chart heatmap">'
        "<figcaption>Messages by weekday and hour — shaded by quantile band, "
        "so equal shading means equally busy relative to the rest of the week. "
        "Hover a cell for its count.</figcaption>"
        f'<div class="hm-row hm-head"><span class="hm-day"></span>'
        f'<span class="hm-cells">{ticks}</span></div>'
        + "".join(rows)
        + f'<div class="hm-legend">less {scale} more</div>'
        "</figure>"
    )


# --------------------------------------------------------------- visuals
# Chart builders for the section layer. All follow the same rules: thin
# marks with rounded data ends, series colors on marks only (text stays in
# ink tokens), hover tooltips on every mark, and a <details> table view as
# the accessibility fallback. Row keys starting with "_" are presentation
# hints (tooltips, badges) and are excluded from the table view.

_REACTION_EMOJI = {
    "love": "❤️", "like": "👍", "dislike": "👎", "laugh": "😂",
    "emphasize": "‼️", "question": "❓", "custom-emoji": "✨", "sticker": "🏷️",
}


def block(title, inner, depth=0):
    """One visual wrapped in the standard block chrome + heading."""
    return (f'<section class="block d{min(depth, 3)}">'
            + _heading(depth, title) + inner + "</section>")


def _table_view(rows):
    visible = [{k: v for k, v in r.items() if not str(k).startswith("_")}
               for r in rows]
    return ("<details><summary>Table view</summary>"
            + _rows_table(visible, _columns(visible)) + "</details>")


def render_hbars(rows, label_key, value_key, table=True):
    """Horizontal bar list: label | bar | value. The value label formats by
    key (counts, durations, shares); bar length is linear against the max.
    Optional per-row presentation hints: '_badge' (a chip after the label,
    '_badge_up' truthy tints it as a positive delta) and '_tip' (tooltip)."""
    vals = [float(r.get(value_key) or 0) for r in rows]
    peak = max(vals, default=0)
    if peak <= 0:
        return None
    out = []
    for i, (row, v) in enumerate(zip(rows, vals)):
        badge = ""
        if row.get("_badge"):
            cls = "hb-badge up" if row.get("_badge_up") else "hb-badge"
            badge = f' <span class="{cls}">{esc(row["_badge"])}</span>'
        tip = f' title="{esc(row["_tip"])}"' if row.get("_tip") else ""
        out.append(
            f'<div class="hb-row" style="--i:{i}"{tip}>'
            f'<span class="hb-label">{format_scalar(row.get(label_key), label_key)}'
            f"{badge}</span>"
            f'<span class="hb-track"><span class="hb-fill" style="--w:{v / peak * 100:.1f}%">'
            "</span></span>"
            f'<span class="hb-val">{format_scalar(row.get(value_key), value_key)}</span>'
            "</div>"
        )
    html_out = '<div class="hbars">' + "".join(out) + "</div>"
    if table:
        html_out += _table_view(rows)
    return html_out


def render_split_bars(rows, label_key, left_key, right_key,
                      left_label="Me", right_label="Them",
                      sqrt_scale=False, table=True):
    """Opposed two-series bars around a center label column (given|received,
    me|them). sqrt_scale compresses wide-ranging magnitudes (reply times span
    seconds to hours) while keeping order; exact values stay in the labels
    and tooltips."""
    def mag(value):
        value = float(value or 0)
        return math.sqrt(value) if sqrt_scale else value

    peak = max((mag(r.get(k)) for r in rows for k in (left_key, right_key)),
               default=0)
    if peak <= 0:
        return None
    body = []
    for i, row in enumerate(rows):
        label = str(row.get(label_key, ""))
        cells = [f'<div class="sb-row" style="--i:{i}">']
        for side, key, series in (("l", left_key, "s1"), ("r", right_key, "s2")):
            value = row.get(key)
            text = format_scalar(value, key)
            tip = esc(f"{label} · {left_label if side == 'l' else right_label}: "
                      f"{display_scalar(value, key)}")
            val = f'<span class="sb-val">{text}</span>'
            track = (f'<span class="sb-track {side}"><span class="sb-fill {series}" '
                     f'style="--w:{mag(value) / peak * 100:.1f}%" title="{tip}">'
                     "</span></span>")
            if side == "l":
                cells += [val, track,
                          f'<span class="sb-label">{esc(label)}</span>']
            else:
                cells += [track, val]
        cells.append("</div>")
        body.append("".join(cells))
    legend = (
        '<div class="legend legend-center">'
        f'<span class="key"><span class="swatch s1"></span>{esc(left_label)}</span>'
        f'<span class="key"><span class="swatch s2"></span>{esc(right_label)}</span>'
        "</div>"
    )
    html_out = legend + '<div class="sbars">' + "".join(body) + "</div>"
    if table:
        html_out += _table_view(rows)
    return html_out


def render_bubbles(rows):
    """Most-reacted messages as iMessage-style chat bubbles: incoming on the
    left, my messages on the right in the send-blue, with the reactions as a
    tapback pill on the bubble's shoulder."""
    out = []
    for row in rows:
        me = str(row.get("sender")) == "Me"
        pills = []
        for kind, n in (row.get("reactions") or {}).items():
            emoji = _REACTION_EMOJI.get(str(kind), "✨")
            try:
                count = int(n)
            except (TypeError, ValueError):
                count = 1
            pills.append(f"{emoji} ×{count}" if count > 1 else emoji)
        tap = (f'<span class="tapback">{esc("  ".join(pills))}</span>'
               if pills else "")
        caption = " · ".join(p for p in (
            str(row.get("sender") or ""),
            display_scalar(row.get("date"), "date"),
        ) if p)
        out.append(
            f'<figure class="msg {"out" if me else "inc"}">'
            f'<div class="bubble">{tap}{format_scalar(row.get("text"))}</div>'
            f"<figcaption>{esc(caption)}</figcaption></figure>"
        )
    return '<div class="bubbles">' + "".join(out) + "</div>"


def render_word_cloud(rows, word_key="word", count_key="count"):
    """Size-scaled word chips: font size follows sqrt(count/max) so the
    heavy hitters read first without drowning the rest. Counts live in the
    tooltip; the top three get primary ink and weight."""
    counts = [float(r.get(count_key) or 0) for r in rows]
    peak = max(counts, default=0)
    if peak <= 0:
        return None
    chips = []
    for i, (row, count) in enumerate(zip(rows, counts)):
        size = 0.9 + 0.95 * math.sqrt(count / peak)
        cls = "cw cw-top" if i < 3 else "cw"
        tip = f"{format_scalar(row.get(count_key))} uses"
        chips.append(
            f'<span class="{cls}" style="font-size:{size:.2f}rem" '
            f'title="{tip}">{esc(str(row.get(word_key, "")))}</span>'
        )
    return '<div class="cloud">' + "".join(chips) + "</div>"


def render_emoji_row(rows, emoji_key="emoji", count_key="count"):
    """Top emoji as large glyph tiles with counts."""
    tiles = []
    for row in rows:
        tiles.append(
            '<div class="emoji-tile">'
            f'<div class="emoji-glyph">{esc(str(row.get(emoji_key, "")))}</div>'
            f'<div class="emoji-count">{format_scalar(row.get(count_key))}</div>'
            "</div>"
        )
    return '<div class="emoji-grid">' + "".join(tiles) + "</div>"


def render_share_bar(rows, label_key="member", share_key="share"):
    """One 100% stacked bar of who-talks-how-much, categorical slots 1-4
    with everyone past four folded into a muted 'Other' segment. The legend
    carries name + percent as direct labels (the relief for the light-mode
    slots that sit under 3:1 contrast)."""
    shares = [(str(r.get(label_key, "")), float(r.get(share_key) or 0))
              for r in rows]
    total = sum(s for _, s in shares)
    if total <= 0:
        return None
    if len(shares) > 4:
        folded = sum(s for _, s in shares[4:])
        shares = shares[:4] + [("Other", folded)]
    segs, legend = [], []
    for i, (name, share) in enumerate(shares):
        cls = f"s{i + 1}" if i < 4 else "s-other"
        pct = share / total * 100
        label = f"{name} {pct:.1f}%"
        segs.append(f'<span class="seg {cls}" style="--w:{pct:.1f}%" '
                    f'title="{esc(label)}"></span>')
        legend.append(f'<span class="key"><span class="swatch {cls}"></span>'
                      f'{esc(name)} <b>{pct:.1f}%</b></span>')
    return ('<div class="share-bar">' + "".join(segs) + "</div>"
            + '<div class="legend legend-wrap">' + "".join(legend) + "</div>")


def render_joke_cards(rows):
    """Inside jokes as a card grid: the term big, usage count as a badge,
    provenance underneath, and the first-use message in the tooltip."""
    cards = []
    for row in rows:
        meta1 = " · ".join(p for p in (
            f"with {row['with']}" if row.get("with") else "",
            str(row.get("kind") or ""),
        ) if p)
        meta2 = " · ".join(p for p in (
            f"coined by {row['coined_by']}" if row.get("coined_by") else "",
            display_scalar(row.get("first_used"), "date")
            if row.get("first_used") else "",
        ) if p)
        quote = display_scalar(row.get("quote")) if row.get("quote") else ""
        tip = f' title="{esc("First use: “" + quote + "”")}"' if quote else ""
        count = (f'<span class="joke-count">{format_scalar(row.get("uses"))}×</span>'
                 if row.get("uses") else "")
        cards.append(
            f'<div class="joke"{tip}>'
            f'<div class="joke-term">{esc(str(row.get("term", "")))}{count}</div>'
            + (f'<div class="joke-meta">{esc(meta1)}</div>' if meta1 else "")
            + (f'<div class="joke-meta">{esc(meta2)}</div>' if meta2 else "")
            + "</div>"
        )
    return '<div class="jokes">' + "".join(cards) + "</div>"


# --- feature: nudges ---
def _open_loop_fallback(row):
    """A plain one-line description of an open loop when no LLM nudge_text was
    written -- so the board reads sensibly after a --no-llm detector run."""
    who = row.get("who") or "someone"
    if row.get("due"):
        return f"{row.get('kind') or 'A date'} with {who} is coming up"
    return f"{who}: {row.get('kind') or 'a loose end'}"


def render_open_loops(rows):
    """The Open Loops board (Wrapped N5): each drifting thread as an INCOMING
    iMessage bubble -- the loop 'texting you back' -- captioned with who it's
    about, why it's open, and, for a recurring occasion, the upcoming date as a
    tapback pill. Reuses the bubble chrome (.bubbles/.msg.inc/.bubble/.tapback),
    so it inherits light/dark and needs no network or new styles."""
    out = []
    for row in rows:
        text = row.get("nudge") or _open_loop_fallback(row)
        due = row.get("due")
        tap = (f'<span class="tapback">📅 {esc(display_scalar(due, "date"))}</span>'
               if due else "")
        cap_bits = [f"with {row['who']}" if row.get("who") else "",
                    str(row.get("kind") or "")]
        if not due and row.get("since"):
            cap_bits.append(f"since {display_scalar(row['since'], 'date')}")
        caption = " · ".join(p for p in cap_bits if p)
        out.append(
            f'<figure class="msg inc"><div class="bubble">{tap}'
            f'{format_scalar(text)}</div>'
            f'<figcaption>{esc(caption)}</figcaption></figure>'
        )
    return '<div class="bubbles">' + "".join(out) + "</div>"
# --- end feature: nudges ---


# Bucket bounds are tuned to where corpus-average valence actually lives
# (annotated chunk tones cluster within roughly ±0.15; a whole month past
# ±0.08 is genuinely rough/glowing).
_TONE_BUCKETS = (  # (upper bound, css class)
    (-0.08, "dv-neg2"), (-0.02, "dv-neg1"), (0.02, "dv-mid"),
    (0.08, "dv-pos1"), (float("inf"), "dv-pos2"),
)


def render_tone_strip(rows):
    """The emotional arc as one strip: a cell per period, filled on the
    blue↔red diverging ramp by average tone (gray = neutral) and labeled
    with the period's leading emotion. Exact numbers live in the tooltip."""
    cells = []
    for row in rows:
        tone = row.get("tone")
        cls = "dv-mid"
        if isinstance(tone, (int, float)):
            for bound, name in _TONE_BUCKETS:
                if tone <= bound:
                    cls = name
                    break
        period = str(row.get("period") or "")
        # Month cells get the month abbreviation (with the year on January /
        # the first cell); year cells just the year.
        m = _ISO_MONTH_RE.match(period)
        if m:
            label = _MONTHS[int(m[2]) - 1]
            if m[2] == "01" or row is rows[0]:
                label += f" ’{m[1][2:]}"
        else:
            label = period
        emotion = str(row.get("top_emotion") or "—")
        tip_bits = [display_scalar(period)]
        if isinstance(tone, (int, float)):
            tip_bits.append(f"tone {tone:+.2f}")
        if row.get("snippets"):
            tip_bits.append(f"{emotion} · {format_scalar(row['snippets'])} snippets")
        cells.append(
            f'<div class="tcell {cls}" title="{esc(" · ".join(tip_bits))}">'
            f'<span class="tcell-emo">{esc(emotion)}</span>'
            f'<span class="tcell-p">{esc(label)}</span></div>'
        )
    keys = "".join(
        f'<span class="key"><span class="swatch" style="background:var(--{var})">'
        f"</span>{label}</span>"
        for var, label in (("dv-neg2", "rough"), ("dv-mid", "neutral"),
                           ("dv-pos2", "glowing"))
    )
    legend = f'<div class="legend legend-center tstrip-legend">{keys}</div>'
    return '<div class="tstrip">' + "".join(cells) + "</div>" + legend


def _uniform_dict_table(items):
    """{'me': {...}, 'them': {...}} -> one table keyed by the outer name.
    Only when every value is a scalars-only dict over the same keys, so the
    rows really are comparable -- otherwise the generic path nests blocks."""
    if len(items) < 2:
        return None
    first_keys = None
    for _, value in items:
        if not isinstance(value, dict):
            return None
        keys = [k for k in value if k not in _HIDDEN_KEYS]
        if not keys or not all(_is_scalar(value[k]) for k in keys):
            return None
        if first_keys is None:
            first_keys = keys
        elif keys != first_keys:
            return None
    head = "".join(f"<th>{esc(humanize(c))}</th>" for c in first_keys)
    body = []
    for name, value in items:
        cells = "".join(f"<td>{format_scalar(value[c], c)}</td>" for c in first_keys)
        body.append(f'<tr><th scope="row">{esc(humanize(name))}</th>{cells}</tr>')
    return (
        '<div class="table-wrap"><table class="grid">'
        f"<thead><tr><th></th>{head}</tr></thead>"
        f"<tbody>{''.join(body)}</tbody></table></div>"
    )


def _render_list(value, depth, key=None):
    if not value:
        return '<p class="note">None.</p>'
    if all(isinstance(v, dict) for v in value):
        cols = _columns(value)
        if "period" in cols:
            # The enclosing block heading already names the series -- a
            # figcaption here would just repeat it.
            chart = render_series_chart(value)
            if chart:
                return chart
        return _rows_table(value, cols)
    chips = "".join(f'<span class="chip">{_inline(v)}</span>' for v in value)
    return f'<div class="chips">{chips}</div>'


def render_value(value, depth=0, key=None):
    """Render any stats value as HTML, dispatching on its shape."""
    if _is_scalar(value):
        return f'<p class="scalar">{format_scalar(value, key)}</p>'
    if isinstance(value, list):
        return _render_list(value, depth, key)
    if not isinstance(value, dict):
        return f'<p class="scalar">{esc(value)}</p>'

    note = _unavailable_note(value)
    if note is not None:
        return f'<p class="note">{esc(note)}</p>'
    items = _visible_items(value)
    if not items:
        return '<p class="note">None.</p>'
    scalars = [(k, v) for k, v in items if _is_scalar(v)]
    complex_items = [(k, v) for k, v in items if not _is_scalar(v)]
    # A lone list under one key ({"people": [...]}) is a container: its key
    # would only render a heading repeating the parent's, so render the payload
    # directly. A lone *dict* is not -- there the key is a label that carries
    # data (a group chat's name, a contact's name), and dropping it loses it.
    if not scalars and len(complex_items) == 1 and isinstance(complex_items[0][1], list):
        return render_value(complex_items[0][1], depth, complex_items[0][0])
    if not scalars:
        table = _uniform_dict_table(complex_items)
        if table:
            return table
    parts = []
    if scalars:
        parts.append(_tiles(scalars) if depth == 0 else _kv_table(scalars))
    for sub_key, sub_value in complex_items:
        bare = sub_key in _CONTAINER_KEYS and isinstance(sub_value, list)
        parts.append(
            f'<section class="block d{min(depth, 3)}">'
            + ("" if bare else _heading(depth, humanize(sub_key)))
            + render_value(sub_value, depth + 1, sub_key)
            + "</section>"
        )
    return "".join(parts)


def humanized(value, key=None):
    """A display-oriented copy of a stats value for the LLM narrator: hidden
    keys stripped, labels humanized, unavailable sub-results dropped, and
    every scalar formatted exactly as the page will render it -- so the
    narrator can only quote numbers in the same form the reader sees."""
    if _is_scalar(value):
        return display_scalar(value, key)
    if isinstance(value, list):
        return [humanized(v) for v in value]
    if isinstance(value, dict):
        if _unavailable_note(value) is not None:
            return None
        out = {}
        for k, v in _visible_items(value):
            hv = humanized(v, k)
            if hv is not None:
                out[humanize(k) or str(k)] = hv
        return out
    return display_scalar(str(value), key)


