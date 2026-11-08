"""Research topic network website.

Run:  py html_main.py      then open http://localhost:8000

The page loads its data from JSON endpoints. Those endpoints call
research_data.py, which reads the research map's CSVs in data/ and keeps
the 8 trading-relevant areas (see research_data.AREAS).

  GET /api/network     areas (sized by article count) and the links between them
  GET /api/research    articles per area, in total and this year vs last year (not on the page now)

Person D's live layer (live_data.py, static/js/live.js): /api/heading,
/api/crowding/history, /api/crowding, /api/backtest, /api/briefing, and the
briefing audio at /media/briefing.mp3.

Styles live in static/css/, one file per component (see CSS_FILES).
"""
import json
import mimetypes
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

import live_data  # Person D: crowding, backtest and briefing
import radar_data  # Person A: topic radar and signal decay (data/app/*.json)
import research_data as data

PORT = int(os.environ.get("PORT", "8000"))
STATIC_DIR = Path(__file__).resolve().parent / "static"

# Loaded in this order: base tokens first, then layout, then each component.
CSS_FILES = [
    "base.css",
    "layout.css",
    "panel.css",
    "table.css",
    "network.css",
    "research.css",
    "radar.css",  # Person A: topic drill-down, signals, tests
    "live.css",  # Person D: crowding, backtest, briefing
]
CSS_LINKS = "\n".join(
    f'<link rel="stylesheet" href="/static/css/{name}">' for name in CSS_FILES
)

PAGE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Research Lifecycle Radar</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Manrope:wght@400;500;600;700&display=swap">
__CSS_LINKS__
</head>
<body>
  <header class="site-head">
    <div class="brand">
      <h1>Research Lifecycle Radar</h1>
    </div>
    <div class="head-actions">
      <button type="button" class="head-btn" id="brief-btn" aria-haspopup="dialog">
        <svg class="ico" viewBox="0 0 20 20" aria-hidden="true"><path d="M3 11v-1a7 7 0 0 1 14 0v1"/><rect x="2.5" y="11" width="3.5" height="6" rx="1"/><rect x="14" y="11" width="3.5" height="6" rx="1"/></svg>
        <span>Weekly briefing</span><span class="new-dot" id="brief-new" hidden title="New since your last visit"></span></button>
      <button type="button" class="head-btn ghost" id="heading-btn" aria-haspopup="dialog">
        <svg class="ico" viewBox="0 0 20 20" aria-hidden="true"><path d="M3 15l4.5-4.5 3 3L17 7"/><path d="M12.5 7H17v4.5"/></svg>
        <span>Where research is heading</span></button>
    </div>
  </header>

  <main>
    <section class="panel map-panel">
      <div class="panel-head">
        <h2>Research topic network</h2>
        <div class="net-controls">
          <button type="button" class="path-btn active" id="path-btn" aria-pressed="true"
            title="The biggest topic in the selected year, in red, and the topics whose papers cite the same earlier work as it. Press play to watch which tightly linked topics grow next.">Hide affected path</button>
          <div class="year-control">
            <button type="button" class="play-btn" id="play-btn" aria-label="Play years">&#9654;</button>
            <input type="range" id="year-slider" step="1" aria-label="Year">
            <span class="year-label" id="year-label"></span>
          </div>
        </div>
      </div>
      <div class="net-legend" aria-label="Legend">
        <span class="lg" title="Circle size: how many research articles came out on the topic in the selected year."><i class="lg-size"></i><i class="lg-size big"></i>articles that year</span>
        <span class="lg" title="Lines link topics whose papers cite the same earlier work. Thicker and darker means more shared that year; dotted means none."><i class="lg-line thin"></i><i class="lg-line"></i>shared citations</span>
        <span class="lg" id="lg-path" title="Red: the biggest topic in the selected year. Bright: the topics linked to it by shared citations. Faded: the rest."><i class="lg-dot root"></i><i class="lg-dot"></i>biggest topic and its links</span>
        <span class="lg" id="lg-ring" title="Predicted to grow over the next three years (the forecast's top topics), from 2025 data, so shown from 2025 on. Exploratory: the forecast did not pass its test."><i class="lg-ring"></i>predicted to grow</span>
        <!-- shown instead of the items above while a topic is open: a hint until a cloud is
             clicked, then the arrow key (radar.js sets .arrows-on and the hint text) -->
        <span class="lg-flow-hint" id="lg-flow-hint"></span>
        <span class="lg-flow" title="Arrows are direct citations between sub-topics, from the cited to the citing side. They show who builds on whom, not cause and effect.">
          <span class="lg"><svg class="lg-arrow" viewBox="0 0 30 12" aria-hidden="true"><line x1="1" y1="6" x2="17" y2="6" stroke="#617891"/><path d="M12.5,2 L18,6 L12.5,10" stroke="#617891"/><circle cx="24" cy="6" r="4.5"/></svg>builds on</span>
          <span class="lg"><svg class="lg-arrow" viewBox="0 0 30 12" aria-hidden="true"><circle cx="6" cy="6" r="4.5"/><line x1="12" y1="6" x2="28" y2="6" stroke="#632024"/><path d="M23.5,2 L29,6 L23.5,10" stroke="#632024"/></svg>built on by</span>
          <span class="lg">number on an arrow = citations</span>
        </span>
      </div>
      <div class="map-body">
        <div class="network" id="network"></div>
        <div id="dock"></div>
      </div>
    </section>

    <!-- Person A: "Inside a topic" (radar.js moves it into #dock beside the map while a topic is open) -->
    <section class="panel" id="zoom-panel">
      <div class="panel-head">
        <h2 id="zoom-title">Inside a topic</h2>
        <select id="zoom-select" aria-label="Topic"></select>
      </div>
      <div class="zoom-side" id="zoom-side"></div>
    </section>

    <!-- internal view only (?full) -->
    <section class="panel full-only" id="next-panel">
      <div class="panel-head"><h2>What could be next</h2><span class="badge warn">exploratory</span></div>
      <div class="sub" id="next-sub"></div>
      <div class="next-grid">
        <div><h3>Topics</h3><ol class="rank-list" id="next-topics"></ol></div>
        <div><h3>Sub-topics</h3><ol class="rank-list" id="next-subtopics"></ol></div>
      </div>
    </section>
  </main>

  <div class="scrim" id="scrim" hidden></div>

  <!-- "Where research is heading", opened from the header (live.js) -->
  <aside class="drawer wide" id="heading-drawer" role="dialog" aria-modal="true" aria-labelledby="heading-title" hidden>
    <div class="dlg-head">
      <div><h2 id="heading-title">Where research is heading</h2><span class="sub" id="heading-period"></span></div>
      <button type="button" class="close-btn" data-close aria-label="Close">&times;</button>
    </div>
    <table class="data-table opens-topic" id="heading">
      <thead><tr>
        <th>Topic</th>
        <th id="change-th" title="The topic's share of the year's sampled papers, this year against last year. Shares, not counts, so a partial year does not look like a fall. Level: within half a percentage point.">vs last year</th>
        <th title="Person A's growth forecast: the topic's rank among the 8 topics on the map. Exploratory: it did not pass its pre-registered test.">Forecast rank <span class="badge warn">exploratory</span></th>
        <th id="read-first-th" title="The topic's most cited paper per year since publication, among papers from the last five years.">Read first</th>
      </tr></thead>
      <tbody></tbody>
    </table>
  </aside>

  <!-- the Weekly briefing button (live.js) -->
  <div class="modal" id="brief-modal" role="dialog" aria-modal="true" aria-labelledby="brief-title" hidden>
    <div class="dlg-head">
      <div><h2 id="brief-title">Weekly briefing</h2><span class="sub" id="brief-date"></span></div>
      <button type="button" class="close-btn" data-close aria-label="Close">&times;</button>
    </div>
    <audio id="brief-audio" controls preload="none" hidden></audio>
    <div id="brief-card"></div>
    <details class="more">
      <summary>Does trading on decay help? <span class="sub" id="bt-range"></span></summary>
      <div class="live-chart" id="bt-chart"></div>
      <table class="data-table" id="backtest">
        <thead><tr><th>Strategy</th><th class="num">Sharpe (95% CI)</th><th class="num">vs equal weight</th></tr></thead>
        <tbody></tbody>
      </table>
      <p class="sub">Cumulative return after costs. Walk-forward: each month uses only past returns. Market (table only) is before costs, for scale.</p>
    </details>
    <details class="more">
      <summary>Signals right now <span class="sub" id="signals-asof"></span></summary>
      <p class="sub">Momentum: average return over the last 12 months. Volatility: next month's forecast. Kept: return since publication as a share of the return in the original paper. Over 100% means it earned more after publication than the paper reported; below 0% means it lost money.</p>
      <table class="data-table" id="signals-table">
        <thead><tr><th>Signal</th><th class="num">12-month mean</th><th class="num">Vol forecast</th><th class="num">Kept</th></tr></thead>
        <tbody></tbody>
      </table>
      <p class="sub" id="portfolio-line"></p>
    </details>
    <details class="more">
      <summary>Read the briefing</summary>
      <p id="brief-text"></p>
    </details>
  </div>

<script src="https://cdnjs.cloudflare.com/ajax/libs/d3/7.8.5/d3.min.js"></script>
<script>
const getJSON = url => fetch(url).then(r => r.json());
const esc = s => String(s).replace(/[&<>"]/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c]));

// ---------------------------------------------------------------- network

// Does segment a-b cross segment c-d?
function segmentsCross(a, b, c, d) {
  const side = (p, q, r) => Math.sign((q.x - p.x) * (r.y - p.y) - (q.y - p.y) * (r.x - p.x));
  return side(a, b, c) !== side(a, b, d) && side(c, d, a) !== side(c, d, b);
}

// Shortest distance from point p to segment a-b.
function distanceToSegment(p, a, b) {
  const dx = b.x - a.x, dy = b.y - a.y;
  const t = Math.max(0, Math.min(1, ((p.x - a.x) * dx + (p.y - a.y) * dy) / (dx * dx + dy * dy || 1)));
  return Math.hypot(p.x - (a.x + t * dx), p.y - (a.y + t * dy));
}

// How messy a layout is: arrows crossing arrows, plus arrows running
// through circles they are not attached to (counted double).
function layoutScore(nodes, links) {
  const byId = new Map(nodes.map(n => [n.id, n]));
  const segs = links.map(l => [byId.get(l.source), byId.get(l.target)]);
  let score = 0;
  for (let i = 0; i < segs.length; i++) {
    const [a, b] = segs[i];
    for (let j = i + 1; j < segs.length; j++) {
      const [c, d] = segs[j];
      if (a === c || a === d || b === c || b === d) continue;
      if (segmentsCross(a, b, c, d)) score += 1;
    }
    for (const n of nodes) {
      if (n === a || n === b) continue;
      if (distanceToSegment(n, a, b) < n.rMax + 6) score += 2;
    }
  }
  return score;
}

// Try many starting layouts and keep the one with the fewest crossings.
// Seeded, so the graph looks the same every time the page loads.
function bestLayout(nodes, links, W, H, fit, finish) {
  let best = null;
  for (let seed = 1; seed <= 40; seed++) {
    const rand = d3.randomLcg(seed / 41);
    const ns = nodes.map(n => ({ ...n, x: W * (0.2 + 0.6 * rand()), y: H * (0.2 + 0.6 * rand()) }));
    const ls = links.map(l => ({ ...l }));
    const sim = d3.forceSimulation(ns).randomSource(rand).stop()
      .force("link", d3.forceLink(ls).id(d => d.id).distance(d => 70 + d.source.rMax + d.target.rMax).strength(d => 0.1 + d.strength * 0.2))
      .force("charge", d3.forceManyBody().strength(-500))
      .force("x", d3.forceX(W / 2).strength(0.04))
      .force("y", d3.forceY(H / 2).strength(0.06))
      .force("collide", d3.forceCollide(d => d.space).iterations(3));
    for (let i = 0; i < 300; i++) { sim.tick(); ns.forEach(fit); }
    finish(ns);  // stretch to fill the box before judging it
    const score = layoutScore(ns, links);
    if (!best || score < best.score) best = { score, ns };
  }
  return new Map(best.ns.map(n => [n.id, { x: n.x, y: n.y }]));
}

// The selected year is shared by the network and the table. The slider
// sets it; anything listening for "yearchange" redraws for that year.
const yearState = { year: null };
function setYear(year) {
  yearState.year = year;
  document.dispatchEvent(new CustomEvent("yearchange", { detail: year }));
}

// Crowding colour (D's score: an area's share of papers vs its usual share). The map no
// longer uses it; live.js colours the briefing's chips with it. Warm = crowded, cool =
// quiet, on a +-3 scale; plain grey at 0; pale grey where there is no score.
const CROWD_MAX = 3;
// Piecewise (not a spline), so 0 is exactly the plain grey in the middle.
const crowdRamp = d3.piecewise(d3.interpolateRgb, ["#3f6591", "#c4c2be", "#a3343a"]);
const crowdColour = s => s == null ? "#e3ded6"
  : crowdRamp((Math.max(-CROWD_MAX, Math.min(CROWD_MAX, s)) + CROWD_MAX) / (2 * CROWD_MAX));

async function loadNetwork() {
  const net = await getJSON("/api/network");
  const box = document.getElementById("network");
  const LINE = 15;  // label line height (13px text)
  const years = net.slider_years;
  const countFor = (d, y = yearState.year) => d.by_year[String(y)] || 0;

  // Line thickness and strength: that year's shared links, on one scale across years.
  // Curved (power 1.5) so weak links are hairlines and only strong ones stand out.
  const peakLink = d3.max(net.links, l => d3.max(years, y => countFor(l, y))) || 1;
  const share = n => Math.pow(n / peakLink, 1.5);
  const width = n => n === 0 ? 0.75 : 0.6 + 6.4 * share(n);
  const linkOpacity = n => n === 0 ? 0.35 : 0.35 + 0.6 * share(n);

  // Label each circle with the same topic name as the research table,
  // wrapped onto short lines under the circle.
  const wrap = (text, max = 18) => text.split(" ").reduce((lines, word) => {
    const last = lines[lines.length - 1];
    if (last && (last + " " + word).length <= max) lines[lines.length - 1] = last + " " + word;
    else lines.push(word);
    return lines;
  }, []);

  const STEP_MS = 900;
  let playing = null, view = null;
  // the affected path is on by default; the button turns it off and on
  let pathOn = true;
  const pathBtn = document.getElementById("path-btn");
  pathBtn.addEventListener("click", () => {
    pathOn = !pathOn;
    if (view) { view.clearHover(); view.drawPath(); }
  });

  // Links count as "there" in a year only if that year had shared work.
  const active = l => countFor(l) > 0;
  const touches = (l, id) => l.source === id || l.target === id;
  const neighbours = id => new Set(net.links.filter(l => active(l) && touches(l, id))
    .map(l => l.source === id ? l.target : l.source));

  // Draw the whole map for a box of W x H. Called again whenever the box changes size
  // (first layout, the side panel opening or closing, the window resizing).
  function build(W, H) {
    // One scale for every year, so circle sizes can be compared across years.
    const R_MIN = 4, R_MAX = Math.min(W, H) / 9;
    const peak = d3.max(net.nodes, d => d3.max(years, y => countFor(d, y))) || 1;
    const radius = d3.scaleSqrt().domain([0, peak]).range([R_MIN, R_MAX]);

    net.nodes.forEach(d => {
      d.lines = wrap(d.id);
      // lay out for the biggest the circle gets in any year, so it never overlaps
      d.rMax = radius(d3.max(years, y => countFor(d, y)));
      d.r = d.rMax;
      d.space = d.rMax + 14 + (d.lines.length + 1) * LINE / 2;  // name lines and a little room
    });

    // keep circles and their labels inside the box
    const fit = d => {
      const half = Math.max(d.rMax, 50);
      d.x = Math.max(half + 8, Math.min(W - half - 8, d.x));
      d.y = Math.max(d.rMax + 12, Math.min(H - d.rMax - 12 - (d.lines.length + 1) * LINE, d.y));
    };

    // A layout settles in the middle of the box. Stretch it so the circles
    // (and their labels) reach the edges.
    const stretch = (ns, padLo, padHi, size, get, set) => {
      const a = d3.least(ns, d => get(d)), b = d3.greatest(ns, d => get(d));
      const from0 = get(a), from1 = get(b);
      const to0 = padLo(a), to1 = size - padHi(b);
      if (from1 - from0 < 1 || to1 <= to0) return;
      ns.forEach(d => set(d, to0 + (get(d) - from0) * (to1 - to0) / (from1 - from0)));
    };
    const fillBox = ns => {
      stretch(ns, d => Math.max(d.rMax, 50) + 8, d => Math.max(d.rMax, 50) + 8, W,
              d => d.x, (d, v) => { d.x = v; });
      stretch(ns, d => d.rMax + 12, d => d.rMax + 12 + (d.lines.length + 1) * LINE, H,
              d => d.y, (d, v) => { d.y = v; });
      ns.forEach(fit);
    };

    const pos = bestLayout(net.nodes, net.links, W, H, fit, fillBox);
    net.nodes.forEach(d => Object.assign(d, pos.get(d.id)));
    const byId = new Map(net.nodes.map(d => [d.id, d]));

    const old = box.querySelector(":scope > svg");
    if (old) old.remove();
    const svg = d3.select(box).insert("svg", ":first-child")
      .attr("viewBox", [0, 0, W, H]).attr("width", "100%").attr("height", "100%");

    // Links have no direction: two areas whose papers cite the same earlier work.
    const link = svg.append("g").selectAll("line").data(net.links).join("line")
      .attr("class", "net-link");
    const linkTitle = link.append("title");

    const node = svg.append("g").selectAll("g").data(net.nodes).join("g")
      .attr("class", "net-node")
      .attr("transform", d => `translate(${d.x},${d.y})`);
    const circle = node.append("circle");
    const label = node.append("text").attr("class", "net-label").attr("text-anchor", "middle");
    label.selectAll("tspan.name").data(d => d.lines).join("tspan").attr("class", "name")
      .attr("x", 0).attr("dy", (t, i) => i === 0 ? 0 : LINE).text(t => t);
    const nodeTitle = node.append("title");
    // The mouse target: an invisible circle on top, a little bigger than the visible one and
    // never scaled on hover. If the visible circle took the mouse, its hover grow would move
    // its edge under a still cursor and hover would flicker on and off.
    const hit = node.append("circle").attr("class", "hit");

    // stop each line at the edge of both circles
    const lineEnds = l => {
      const s = byId.get(l.source), t = byId.get(l.target);
      const dx = t.x - s.x, dy = t.y - s.y, dist = Math.hypot(dx, dy) || 1;
      return {
        x1: s.x + dx / dist * s.r, y1: s.y + dy / dist * s.r,
        x2: t.x - dx / dist * t.r, y2: t.y - dy / dist * t.r,
      };
    };
    const placeLinks = sel => sel
      .attr("x1", l => lineEnds(l).x1).attr("y1", l => lineEnds(l).y1)
      .attr("x2", l => lineEnds(l).x2).attr("y2", l => lineEnds(l).y2);

    // Hovering a topic highlights it and the topics it is linked to that year
    // (not while the affected path is shown: that is its own highlight).
    const clearHover = () => { box.classList.remove("focus"); node.classed("hl", false); link.classed("hl", false); };
    node.on("mouseenter", (e, d) => {
      if (pathOn) return;
      const near = neighbours(d.id);
      box.classList.add("focus");
      node.classed("hl", n => n.id === d.id || near.has(n.id));
      link.classed("hl", l => active(l) && touches(l, d.id));
    }).on("mouseleave", clearHover);

    // Affected path (Person C): the biggest circle in the selected year in red, the topics
    // linked to it that year kept bright, the rest faded. Played through the years, it shows
    // how topics tightly linked to the leader grow into the next big ones.
    const drawPath = () => {
      box.classList.toggle("path", pathOn);
      pathBtn.classList.toggle("active", pathOn);
      pathBtn.setAttribute("aria-pressed", pathOn);
      pathBtn.textContent = pathOn ? "Hide affected path" : "Show affected path";
      const root = net.nodes.reduce((a, b) => countFor(b) > countFor(a) ? b : a);
      const hits = neighbours(root.id);
      node.classed("path-root", d => pathOn && d === root)
        .classed("path-hit", d => pathOn && hits.has(d.id));
      link.classed("path-link", l => pathOn && active(l) && touches(l, root.id));
    };

    // Redraw everything for one year.
    const drawYear = (year, ms) => {
      net.nodes.forEach(d => { d.r = radius(countFor(d, year)); });
      const t = sel => ms ? sel.transition().duration(ms) : sel;
      t(circle).attr("r", d => d.r);
      t(hit).attr("r", d => d.r + 6);
      t(label).attr("y", d => d.r + 20);  // clears the dashed forecast ring (r + 4)
      t(link).attr("stroke-width", l => width(countFor(l, year)))
        .attr("stroke-opacity", l => linkOpacity(countFor(l, year)));
      placeLinks(t(link));
      link.classed("quiet", l => countFor(l, year) === 0);
      nodeTitle.text(d => {
        const n = countFor(d, year);
        return `${d.id}\n${n} article${n === 1 ? "" : "s"} in ${year} · ${d.articles} in total`
          + "\nClick to explore its papers";
      });
      linkTitle.text(l => { const n = countFor(l, year);
        return `${l.source} and ${l.target}\n${n} paper pair${n === 1 ? "" : "s"} in ${year} cite${n === 1 ? "s" : ""} the same earlier work`; });
      drawPath();  // the biggest circle can change between years
    };

    return { node, link, drawYear, drawPath, clearHover };
  }

  document.addEventListener("yearchange", e => { if (view) view.drawYear(e.detail, playing ? STEP_MS * 0.8 : 400); });

  // ---- slider and play button
  const slider = document.getElementById("year-slider");
  const yearLabel = document.getElementById("year-label");
  const playBtn = document.getElementById("play-btn");
  slider.min = years[0];
  slider.max = years[years.length - 1];
  slider.value = slider.max;

  const stop = () => {
    clearInterval(playing); playing = null;
    playBtn.textContent = "▶"; playBtn.setAttribute("aria-label", "Play years");
  };
  const show = y => { slider.value = y; yearLabel.textContent = y; setYear(+y); };

  slider.addEventListener("input", () => { stop(); show(slider.value); });
  playBtn.addEventListener("click", () => {
    if (playing) return stop();
    if (+slider.value >= +slider.max) show(slider.min);  // start again from the beginning
    playBtn.textContent = "❚❚"; playBtn.setAttribute("aria-label", "Pause");
    playing = setInterval(() => {
      const next = +slider.value + 1;
      if (next > +slider.max) return stop();
      show(next);
    }, STEP_MS);
  });
  yearLabel.textContent = slider.value;
  yearState.year = +slider.value;

  // Measure the box whenever it changes: on first load it can still be 0 wide, and the
  // side panel or the window can change it later. Sizes too small to draw are skipped.
  let drawn = [0, 0], timer = null;
  const render = () => {
    const W = box.clientWidth, H = box.clientHeight;
    if (W < 80 || H < 80 || (Math.abs(W - drawn[0]) < 2 && Math.abs(H - drawn[1]) < 2)) return;
    drawn = [W, H];
    view = build(W, H);
    view.drawYear(yearState.year, 0);
  };
  new ResizeObserver(() => { clearTimeout(timer); timer = setTimeout(render, 120); }).observe(box);
  render();
}

loadNetwork();
</script>
<script src="/static/js/radar.js"></script>
<script src="/static/js/live.js"></script>
</body>
</html>
""".replace("__CSS_LINKS__", CSS_LINKS)


class Handler(BaseHTTPRequestHandler):
    def _send(self, status, body, content_type):
        payload = body if isinstance(body, bytes) else body.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Cache-Control", "no-cache")  # see CSS edits on reload
        self.end_headers()
        self.wfile.write(payload)

    def _json(self, obj, status=200):
        self._send(status, json.dumps(obj), "application/json")

    def _static(self, rel_path):
        file = (STATIC_DIR / rel_path).resolve()
        if STATIC_DIR not in file.parents or not file.is_file():
            self._json({"error": "not found"}, 404)
            return
        ctype = mimetypes.guess_type(file.name)[0] or "application/octet-stream"
        if ctype.startswith("text/"):
            ctype += "; charset=utf-8"
        self._send(200, file.read_bytes(), ctype)

    def do_GET(self):
        url = urlparse(self.path)
        routes = {
            "/api/network": data.get_topic_network,
            "/api/research": data.get_research_snapshot,
            "/api/topics": radar_data.get_topics,
            "/api/forecasts": radar_data.get_forecasts,
            "/api/signals": radar_data.get_signals,
            "/api/crowding": live_data.get_crowding,
            "/api/crowding/history": live_data.get_crowding_history,
            "/api/heading": live_data.get_heading,
            "/api/backtest": live_data.get_backtest,
            "/api/briefing": live_data.get_briefing,
        }
        if url.path == "/api/topic":
            name = parse_qs(url.query).get("name", [""])[0]
            self._json(radar_data.get_topic(name))
        elif url.path in ("/", "/index.html"):
            self._send(200, PAGE, "text/html; charset=utf-8")
        elif url.path == "/media/briefing.mp3" and live_data.AUDIO_FILE.is_file():
            self._send(200, live_data.AUDIO_FILE.read_bytes(), "audio/mpeg")
        elif url.path.startswith("/static/"):
            self._static(unquote(url.path[len("/static/"):]))
        elif url.path in routes:
            self._json(routes[url.path]())
        else:
            self._json({"error": "not found"}, 404)


if __name__ == "__main__":
    server = ThreadingHTTPServer(("127.0.0.1", PORT), Handler)
    print(f"Stock tracker running at http://localhost:{PORT}")
    server.serve_forever()
