"""HTML rendering for the Wrapped report: the page shell plus a generic
renderer that turns a section's arbitrary nested stats dict into readable
HTML (stat tiles, tables, charts) instead of a JSON dump.

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


_CSS = """
/* The page is built from the subject's own materials: iOS grouped-list
   surfaces and separators, the real send-blue for outgoing bubbles, and a
   rounded display face -- so the report reads as Messages, not dashboard.
   Chart series stay on the validated data-viz palette (the send-blue is a
   bubble color, not a series color). */
:root {
  color-scheme: light dark;
  --surface-0: #f2f2f7;
  --surface-1: #ffffff;
  --surface-2: #e9e9eb;
  --border: #d8d8dc;
  --text-primary: #000000;
  --text-secondary: #48484a;
  --text-muted: #8e8e93;
  --series-1: #2a78d6;
  --series-2: #008300;
  --series-3: #e87ba4;
  --series-4: #eda100;
  --series-other: #c7c7cc;
  --delta-up: #006300;
  --bubble-in: #e9e9eb;
  --bubble-out: #007aff;
  --q0: #e9e9eb;
  --q1: #cde2fb; --q2: #9ec5f4; --q3: #6da7ec; --q4: #3987e5;
  --q5: #256abf; --q6: #184f95; --q7: #0d366b;
  --dv-neg2: #e34948; --dv-neg1: #f3c4c3; --dv-mid: #ececf0;
  --dv-pos1: #b7d3f6; --dv-pos2: #2a78d6;
}
@media (prefers-color-scheme: dark) {
  :root:where(:not([data-theme="light"])) {
    --surface-0: #000000;
    --surface-1: #1c1c1e;
    --surface-2: #2c2c2e;
    --border: #38383a;
    --text-primary: #ffffff;
    --text-secondary: #c7c7cc;
    --text-muted: #98989e;
    --series-1: #3987e5;
    --series-2: #008300;
    --series-3: #d55181;
    --series-4: #c98500;
    --series-other: #48484a;
    --delta-up: #0ca30c;
    --bubble-in: #3a3a3c;
    --bubble-out: #0a84ff;
    --q0: #2c2c2e;
    --q1: #0d366b; --q2: #184f95; --q3: #256abf; --q4: #2a78d6;
    --q5: #3987e5; --q6: #6da7ec; --q7: #9ec5f4;
    --dv-neg2: #e66767; --dv-neg1: #63302f; --dv-mid: #2c2c2e;
    --dv-pos1: #1c4d8f; --dv-pos2: #3987e5;
  }
}
:root[data-theme="dark"] {
  --surface-0: #000000;
  --surface-1: #1c1c1e;
  --surface-2: #2c2c2e;
  --border: #38383a;
  --text-primary: #ffffff;
  --text-secondary: #c7c7cc;
  --text-muted: #98989e;
  --series-1: #3987e5;
  --series-2: #008300;
  --series-3: #d55181;
  --series-4: #c98500;
  --series-other: #48484a;
  --delta-up: #0ca30c;
  --bubble-in: #3a3a3c;
  --bubble-out: #0a84ff;
  --q0: #2c2c2e;
  --q1: #0d366b; --q2: #184f95; --q3: #256abf; --q4: #2a78d6;
  --q5: #3987e5; --q6: #6da7ec; --q7: #9ec5f4;
  --dv-neg2: #e66767; --dv-neg1: #63302f; --dv-mid: #2c2c2e;
  --dv-pos1: #1c4d8f; --dv-pos2: #3987e5;
}
* { box-sizing: border-box; }
:root {
  --font-display: ui-rounded, -apple-system, BlinkMacSystemFont, "Segoe UI",
    Helvetica, Arial, sans-serif;
}
body {
  margin: 0;
  padding: 2.5rem 1.25rem 5rem;
  background: var(--surface-0);
  color: var(--text-primary);
  font: 16px/1.6 -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif;
  -webkit-font-smoothing: antialiased;
}
main { max-width: 62rem; margin: 0 auto; }
h1 {
  font-family: var(--font-display);
  font-size: clamp(2rem, 5vw, 3rem);
  line-height: 1.1;
  letter-spacing: -0.02em;
  margin: 0 0 0.25rem;
}
.subtitle { color: var(--text-muted); margin: 0 0 2.5rem; }
/* The hero: the year, sent as one message. */
.hero {
  display: flex;
  flex-direction: column;
  align-items: flex-end;
  margin: 1.75rem 0 2.75rem;
}
.hero-bubble {
  background: var(--bubble-out);
  color: #fff;
  font-family: var(--font-display);
  font-size: clamp(1.9rem, 4.5vw, 2.7rem);
  font-weight: 700;
  letter-spacing: -0.02em;
  line-height: 1.15;
  padding: 0.9rem 1.5rem 1rem;
  border-radius: 26px;
  border-bottom-right-radius: 7px;
  position: relative;
  max-width: 90%;
}
.hero-unit {
  display: block;
  font-size: 0.95rem;
  font-weight: 600;
  letter-spacing: 0.01em;
  opacity: 0.85;
}
.hero-bubble .tapback {
  top: -1.1rem;
  left: -0.6rem;
  right: auto;
  font-size: 0.85rem;
}
.hero-cap {
  color: var(--text-muted);
  font-size: 0.8rem;
  margin-top: 1.2rem;
}
h2 {
  font-family: var(--font-display);
  font-size: 1.5rem;
  letter-spacing: -0.01em;
  margin: 0 0 1rem;
  padding-bottom: 0.6rem;
  border-bottom: 1px solid var(--border);
}
.block-title {
  font-size: 0.85rem;
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: 0.06em;
  color: var(--text-secondary);
  margin: 0 0 0.6rem;
}
/* Top-level blocks (group names, subsection titles) outrank the uppercase
   labels of the nested blocks inside them. */
h3.block-title {
  font-size: 1.05rem;
  font-weight: 650;
  text-transform: none;
  letter-spacing: -0.01em;
  color: var(--text-primary);
  margin-top: 0.4rem;
}
section.card {
  background: var(--surface-1);
  border: 1px solid var(--border);
  border-radius: 14px;
  padding: 1.5rem;
  margin-bottom: 1.5rem;
}
section.block { margin: 1.25rem 0 0; }
section.block.d1, section.block.d2, section.block.d3 {
  border-left: 2px solid var(--border);
  padding-left: 1rem;
}
.narrative { font-size: 1.05rem; color: var(--text-secondary); margin: 0 0 1.25rem; }
.note { color: var(--text-muted); font-style: italic; margin: 0.25rem 0; }
.scalar { margin: 0.25rem 0; }
.tiles {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(9rem, 1fr));
  gap: 0.75rem;
  margin-bottom: 0.5rem;
}
.tile {
  background: var(--surface-2);
  border-radius: 10px;
  padding: 0.9rem 1rem;
}
.tile-value {
  font-family: var(--font-display);
  font-size: 1.6rem;
  font-weight: 650;
  letter-spacing: -0.02em;
  line-height: 1.2;
}
.tile-label {
  font-size: 0.75rem;
  text-transform: uppercase;
  letter-spacing: 0.05em;
  color: var(--text-muted);
  margin-top: 0.2rem;
}
.table-wrap { overflow-x: auto; }
table { border-collapse: collapse; width: 100%; font-size: 0.9rem; }
table.kv { max-width: 34rem; }
th, td {
  text-align: left;
  padding: 0.45rem 0.75rem 0.45rem 0;
  border-bottom: 1px solid var(--border);
  vertical-align: top;
}
thead th {
  font-size: 0.72rem;
  text-transform: uppercase;
  letter-spacing: 0.05em;
  color: var(--text-muted);
  font-weight: 600;
  white-space: nowrap;
}
table.kv th { color: var(--text-secondary); font-weight: 500; width: 45%; }
tbody tr:last-child th, tbody tr:last-child td { border-bottom: none; }
td { color: var(--text-secondary); }
.chips { display: flex; flex-wrap: wrap; gap: 0.4rem; }
.chip {
  background: var(--surface-2);
  border-radius: 999px;
  padding: 0.15rem 0.6rem;
  font-size: 0.85rem;
  color: var(--text-secondary);
}
figure.chart { margin: 0.5rem 0 0; }
figcaption {
  font-size: 0.85rem;
  color: var(--text-muted);
  margin-bottom: 0.75rem;
}
.legend { display: flex; gap: 1rem; margin-bottom: 0.75rem; font-size: 0.8rem; }
.key { display: inline-flex; align-items: center; gap: 0.4rem; color: var(--text-secondary); }
.swatch { width: 10px; height: 10px; border-radius: 3px; flex: 0 0 10px; }
.swatch.s1, .bar.s1 { background: var(--series-1); }
.swatch.s2, .bar.s2 { background: var(--series-2); }
.swatch.s3 { background: var(--series-3); }
.swatch.s4 { background: var(--series-4); }
.swatch.s-other { background: var(--series-other); }
.legend-center { justify-content: center; }
.legend-wrap { flex-wrap: wrap; row-gap: 0.35rem; margin-top: 0.7rem; }
.legend-wrap .key b { font-weight: 600; color: var(--text-primary); }
.bars {
  display: flex;
  align-items: flex-end;
  gap: 0.5rem;
  height: 11rem;
  overflow-x: auto;
  padding-bottom: 0.25rem;
  border-bottom: 1px solid var(--border);
}
.bar-col { flex: 1 1 0; min-width: 2.2rem; display: flex; flex-direction: column; height: 100%; }
.bar-group { flex: 1; display: flex; align-items: flex-end; justify-content: center; gap: 2px; }
.bar { width: 12px; height: var(--h, 0%); border-radius: 4px 4px 0 0; min-height: 2px; }
.bar-tick {
  font-size: 0.65rem;
  color: var(--text-muted);
  text-align: center;
  padding-top: 0.35rem;
  white-space: nowrap;
}
details { margin-top: 0.9rem; }
summary {
  cursor: pointer;
  font-size: 0.8rem;
  color: var(--text-muted);
}
.heatmap .hm-row { display: flex; align-items: center; gap: 0.4rem; }
.hm-day {
  width: 2.4rem;
  font-size: 0.7rem;
  color: var(--text-muted);
  text-align: right;
}
.hm-cells { display: flex; gap: 2px; flex: 1; }
.hm-cell { flex: 1; aspect-ratio: 1; border-radius: 2px; background: var(--q0); min-width: 8px; }
.hm-row + .hm-row { margin-top: 2px; }
.hm-head .hm-tick {
  flex: 1;
  font-size: 0.6rem;
  color: var(--text-muted);
  text-align: center;
  min-width: 8px;
}
.hm-cell.q1 { background: var(--q1); } .hm-cell.q2 { background: var(--q2); }
.hm-cell.q3 { background: var(--q3); } .hm-cell.q4 { background: var(--q4); }
.hm-cell.q5 { background: var(--q5); } .hm-cell.q6 { background: var(--q6); }
.hm-cell.q7 { background: var(--q7); }
.hm-legend {
  display: flex;
  align-items: center;
  gap: 3px;
  justify-content: flex-end;
  font-size: 0.7rem;
  color: var(--text-muted);
  margin-top: 0.6rem;
}
.hm-legend .hm-cell { flex: 0 0 10px; }
footer { color: var(--text-muted); font-size: 0.8rem; text-align: center; margin-top: 2rem; }

/* Horizontal bar lists */
.hbars { display: grid; gap: 0.45rem; }
.hb-row {
  display: grid;
  grid-template-columns: minmax(7.5rem, 11rem) 1fr minmax(4rem, auto);
  align-items: center;
  gap: 0.75rem;
}
.hb-label { font-size: 0.9rem; color: var(--text-secondary); overflow-wrap: anywhere; }
.hb-badge {
  display: inline-block;
  background: var(--surface-2);
  color: var(--text-muted);
  border-radius: 999px;
  font-size: 0.68rem;
  font-weight: 600;
  padding: 0.05rem 0.45rem;
  vertical-align: 0.08rem;
  white-space: nowrap;
}
.hb-badge.up { color: var(--delta-up); }
.hb-track { height: 10px; }
.hb-fill {
  display: block;
  height: 100%;
  width: var(--w);
  min-width: 2px;
  border-radius: 0 4px 4px 0;
  background: var(--series-1);
}
.hb-val {
  font-size: 0.85rem;
  color: var(--text-primary);
  font-variant-numeric: tabular-nums;
  text-align: right;
  white-space: nowrap;
}

/* Opposed split bars */
.sbars { display: grid; gap: 0.45rem; margin-top: 0.25rem; }
.sb-row {
  display: grid;
  grid-template-columns: minmax(3.2rem, auto) 1fr minmax(6.5rem, 9.5rem) 1fr minmax(3.2rem, auto);
  align-items: center;
  gap: 0.6rem;
}
.sb-label {
  font-size: 0.85rem;
  color: var(--text-secondary);
  text-align: center;
  overflow-wrap: anywhere;
}
.sb-val {
  font-size: 0.8rem;
  color: var(--text-primary);
  font-variant-numeric: tabular-nums;
  white-space: nowrap;
}
.sb-track { height: 10px; display: flex; }
.sb-track.l { justify-content: flex-end; }
.sb-fill { display: block; height: 100%; width: var(--w); min-width: 2px; }
.sb-track.l .sb-fill { border-radius: 4px 0 0 4px; background: var(--series-1); }
.sb-track.r .sb-fill { border-radius: 0 4px 4px 0; background: var(--series-2); }

/* iMessage-style bubbles */
.bubbles { display: flex; flex-direction: column; gap: 1.1rem; margin: 0.75rem 0 0; }
.msg { margin: 0; max-width: 78%; }
.msg.inc { align-self: flex-start; }
.msg.out { align-self: flex-end; }
.bubble {
  position: relative;
  display: inline-block;
  padding: 0.55rem 0.85rem;
  border-radius: 18px;
  font-size: 0.95rem;
  line-height: 1.4;
  overflow-wrap: anywhere;
}
.msg.inc .bubble {
  background: var(--bubble-in);
  color: var(--text-primary);
  border-bottom-left-radius: 5px;
}
.msg.out .bubble {
  background: var(--bubble-out);
  color: #fff;
  border-bottom-right-radius: 5px;
}
.msg figcaption { font-size: 0.72rem; color: var(--text-muted); margin-top: 0.85rem; }
.msg.out figcaption { text-align: right; }
.tapback {
  position: absolute;
  top: -0.95rem;
  font-size: 0.72rem;
  background: var(--surface-2);
  border: 1px solid var(--border);
  border-radius: 999px;
  padding: 0.12rem 0.5rem;
  white-space: nowrap;
  color: var(--text-secondary);
}
.msg.inc .tapback { right: -0.4rem; }
.msg.out .tapback { left: -0.4rem; }

/* Word cloud */
.cloud {
  display: flex;
  flex-wrap: wrap;
  align-items: baseline;
  gap: 0.35rem 0.9rem;
  padding: 0.4rem 0 0.2rem;
}
.cw { color: var(--text-secondary); line-height: 1.25; cursor: default; }
.cw-top { color: var(--text-primary); font-weight: 650; }
.cw:hover { color: var(--text-primary); }

/* Emoji tiles */
.emoji-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(4.2rem, 1fr));
  gap: 0.5rem;
  margin-top: 0.4rem;
}
.emoji-tile {
  background: var(--surface-2);
  border-radius: 10px;
  text-align: center;
  padding: 0.55rem 0.25rem 0.45rem;
}
.emoji-glyph { font-size: 1.7rem; line-height: 1.2; }
.emoji-count {
  font-size: 0.72rem;
  color: var(--text-muted);
  font-variant-numeric: tabular-nums;
  margin-top: 0.15rem;
}

/* Group share bar */
.share-bar {
  display: flex;
  gap: 2px;
  height: 18px;
  margin-top: 0.4rem;
}
.seg { display: block; width: var(--w); min-width: 3px; }
.seg:first-child { border-radius: 6px 0 0 6px; }
.seg:last-child { border-radius: 0 6px 6px 0; }
.seg.s1 { background: var(--series-1); } .seg.s2 { background: var(--series-2); }
.seg.s3 { background: var(--series-3); } .seg.s4 { background: var(--series-4); }
.seg.s-other { background: var(--series-other); }

/* Tone strip (emotional arc) */
.tstrip { display: flex; gap: 3px; margin-top: 0.5rem; overflow-x: auto; }
.tcell {
  flex: 1 1 0;
  min-width: 4.3rem;
  border-radius: 9px;
  padding: 0.55rem 0.2rem 0.45rem;
  text-align: center;
}
.tcell-emo {
  display: block;
  font-size: 0.63rem;
  font-weight: 650;
  letter-spacing: 0.01em;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}
.tcell-p { display: block; font-size: 0.62rem; opacity: 0.75; margin-top: 0.15rem; }
.tcell.dv-neg2 { background: var(--dv-neg2); color: #fff; }
.tcell.dv-neg1 { background: var(--dv-neg1); color: var(--text-primary); }
.tcell.dv-mid  { background: var(--dv-mid); color: var(--text-secondary); }
.tcell.dv-pos1 { background: var(--dv-pos1); color: var(--text-primary); }
.tcell.dv-pos2 { background: var(--dv-pos2); color: #fff; }
.tstrip-legend { margin-top: 0.6rem; }

/* Inside-joke cards */
.jokes {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(13rem, 1fr));
  gap: 0.75rem;
  margin-top: 0.4rem;
}
.joke {
  background: var(--surface-2);
  border-radius: 12px;
  padding: 0.85rem 1rem;
}
.joke-term {
  font-family: var(--font-display);
  font-size: 1.25rem;
  font-weight: 650;
  letter-spacing: -0.01em;
  display: flex;
  align-items: baseline;
  justify-content: space-between;
  gap: 0.5rem;
  overflow-wrap: anywhere;
}
.joke-count {
  font-size: 0.75rem;
  font-weight: 600;
  color: var(--text-muted);
  font-variant-numeric: tabular-nums;
  white-space: nowrap;
}
.joke-meta {
  font-size: 0.76rem;
  color: var(--text-muted);
  margin-top: 0.25rem;
  overflow-wrap: anywhere;
}

/* Scroll-in animation layer: only when JS is running (html.js) and the
   reader hasn't asked for reduced motion. Without JS everything renders
   fully static and complete. */
@media (prefers-reduced-motion: no-preference) {
  html.js section.card {
    opacity: 0;
    transform: translateY(14px);
    transition: opacity 0.55s ease, transform 0.55s ease;
  }
  html.js section.card.in { opacity: 1; transform: none; }
  .hb-fill, .sb-fill, .seg {
    transition: width 0.9s cubic-bezier(0.22, 0.61, 0.36, 1);
    transition-delay: calc(var(--i, 0) * 45ms);
  }
  .bars .bar { transition: height 0.8s cubic-bezier(0.22, 0.61, 0.36, 1); }
  html.js section.card:not(.in) .hb-fill,
  html.js section.card:not(.in) .sb-fill,
  html.js section.card:not(.in) .seg { width: 0 !important; min-width: 0; }
  html.js section.card:not(.in) .bars .bar { height: 2px !important; }
}
/* Print (and PDF export) must never capture the pre-reveal state. The
   restore rules repeat :not(.in) to TIE the hidden rules' specificity --
   both sides are !important, so a lower-specificity restore silently loses
   and unrevealed cards print with empty bars. The JS beforeprint hook is
   the primary fix; this is the CSS backstop. */
@media print {
  html.js section.card { opacity: 1 !important; transform: none !important; }
  html.js section.card:not(.in) .hb-fill,
  html.js section.card:not(.in) .sb-fill,
  html.js section.card:not(.in) .seg { width: var(--w) !important; }
  html.js section.card:not(.in) .bars .bar { height: var(--h, 0%) !important; }
  section.card, .hb-fill, .sb-fill, .seg, .bars .bar { transition: none !important; }
}
"""

# Self-contained page behavior: reveal cards (and grow their bars) as they
# scroll into view, and count stat tiles up from zero. No network, no
# dependencies; prefers-reduced-motion falls back to static reveal, and
# without JS the CSS above never hides anything.
_JS = """
(function () {
  if (!document.documentElement.classList.contains('js')) return;
  var reduce = matchMedia('(prefers-reduced-motion: reduce)').matches;
  var cards = document.querySelectorAll('section.card');
  // Printing must never capture the pre-reveal state, wherever the reader
  // has scrolled to -- reveal everything before the print snapshot.
  addEventListener('beforeprint', function () {
    cards.forEach(function (c) { c.classList.add('in'); });
  });
  if (reduce || !('IntersectionObserver' in window)) {
    cards.forEach(function (c) { c.classList.add('in'); });
    return;
  }
  var io = new IntersectionObserver(function (entries) {
    entries.forEach(function (e) {
      if (!e.isIntersecting) return;
      e.target.classList.add('in');
      io.unobserve(e.target);
    });
  }, { rootMargin: '0px 0px -8% 0px', threshold: 0.01 });
  cards.forEach(function (c) { io.observe(c); });

  document.querySelectorAll('.tile-value').forEach(function (el) {
    var text = el.textContent.trim();
    if (!/^[\\d,]+$/.test(text)) return;
    var target = parseInt(text.replace(/,/g, ''), 10);
    if (!(target > 1)) return;
    var seen = new IntersectionObserver(function (entries) {
      entries.forEach(function (e) {
        if (!e.isIntersecting) return;
        seen.unobserve(el);
        var t0 = performance.now(), dur = 900;
        (function step(t) {
          var p = Math.min(1, (t - t0) / dur);
          p = 1 - Math.pow(1 - p, 3);
          el.textContent = Math.round(target * p).toLocaleString('en-US');
          if (p < 1) requestAnimationFrame(step); else el.textContent = text;
        })(t0);
      });
    }, { threshold: 0.4 });
    seen.observe(el);
  });
})();
"""


def render_hero(total_messages, caption, reactions_received=None):
    """The signature opener: the year's total, sent as one outgoing message
    (with the reactions it earned as a tapback), marked delivered."""
    tap = ""
    if reactions_received:
        tap = f'<span class="tapback">❤️ ×{int(reactions_received):,}</span>'
    return (
        '<div class="hero"><div class="hero-bubble">'
        f"{tap}{int(total_messages):,}"
        '<span class="hero-unit">messages</span></div>'
        f'<div class="hero-cap">{esc(caption)}</div></div>'
    )


def render_page(title, subtitle, body, footer="", hero=""):
    """Wrap rendered section HTML in the standalone page shell. A hero (see
    render_hero) replaces the plain subtitle line; the tiny head script
    stamps html.js before first paint so the animation CSS only ever applies
    when the reveal script will actually run."""
    foot = f"<footer>{esc(footer)}</footer>" if footer else ""
    intro = hero or (f'<p class="subtitle">{esc(subtitle)}</p>\n' if subtitle else "")
    return (
        "<!doctype html>\n"
        '<html lang="en">\n<head>\n<meta charset="utf-8">\n'
        '<meta name="viewport" content="width=device-width, initial-scale=1">\n'
        f"<title>{esc(title)}</title>\n"
        "<script>if (location.hash !== '#static') "
        "document.documentElement.classList.add('js')</script>\n"
        f"<style>{_CSS}</style>\n</head>\n<body>\n"
        f"<main>\n<h1>{esc(title)}</h1>\n"
        + intro
        + body
        + f"\n{foot}\n</main>\n<script>{_JS}</script>\n</body>\n</html>\n"
    )


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
