"""The Wrapped report's page shell: the inline CSS and JS, the hero opener,
and render_page, which wraps rendered section HTML into a standalone,
network-free document.

The generic renderers (value formatting, tables, charts, the visual builders)
live in wrapped_render.py and the feature-specific ones in wrapped_visuals.py;
both are re-exported here, so `wrapped_html.render_value` /
`wrapped_html.render_voiceprint_card` keep resolving for callers and tests.
"""

from wrapped_render import (  # noqa: F401  (re-exported for callers/tests)
    _CONTAINER_KEYS,
    _HIDDEN_KEYS,
    _ISO_DATE_RE,
    _ISO_MONTH_RE,
    _LABEL_OVERRIDES,
    _LABEL_WORDS,
    _MONTHS,
    _REACTION_EMOJI,
    _SEQ_STEPS,
    _TONE_BUCKETS,
    _WEEKDAYS,
    _columns,
    _format_float,
    _heading,
    _inline,
    _is_scalar,
    _kv_table,
    _numeric_series_keys,
    _open_loop_fallback,
    _quantile_cuts,
    _render_list,
    _rows_table,
    _table_view,
    _tiles,
    _unavailable_note,
    _uniform_dict_table,
    _visible_items,
    block,
    display_scalar,
    esc,
    format_duration,
    format_hour,
    format_scalar,
    heatmap_grid,
    humanize,
    humanized,
    render_bubbles,
    render_emoji_row,
    render_hbars,
    render_heatmap,
    render_joke_cards,
    render_open_loops,
    render_series_chart,
    render_share_bar,
    render_split_bars,
    render_tone_strip,
    render_value,
    render_word_cloud,
)

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


# The feature visual builders were split into wrapped_visuals; re-export them
# so wrapped_sections and the tests keep resolving wrapped_html.render_*.
# Bottom import: wrapped_visuals imports the generic renderers defined above.
from wrapped_visuals import _WELLBEING_STYLE  # noqa: E402,F401
from wrapped_visuals import _WELLBEING_URL_RE  # noqa: E402,F401
from wrapped_visuals import _voiceprint_phrases  # noqa: E402,F401
from wrapped_visuals import render_support_block  # noqa: E402,F401
from wrapped_visuals import render_trajectory  # noqa: E402,F401
from wrapped_visuals import render_voiceprint_card  # noqa: E402,F401
from wrapped_visuals import render_wellbeing  # noqa: E402,F401
from wrapped_visuals import render_window_chips  # noqa: E402,F401
