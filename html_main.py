"""Research Radar website: the team's results on one page.

Run:  python html_main.py      then open http://localhost:8000

The page loads its data from JSON endpoints. Those endpoints call real_data.py,
which reads the committed output files in data/ (falling back to data/sample/), so
the site runs offline from a fresh clone.

  GET /api/network     B: research areas (sized by papers) and shared-reference links
  GET /api/research    D: this week's crowding score per area
  GET /api/decay       A: do published signals stop working?
  GET /api/backtest    D: allocation backtest vs baselines, with intervals
  GET /api/briefing    D: decision card and spoken briefing
  GET /api/summary     top strip: what's hidden, inputs, what predicts, the decision
  GET /media/briefing.mp3

synthetic_data.py (C's original made-up data) is kept for building the page
without real files; set DATA_MODULE=synthetic_data to use it (only the network
and research panels have synthetic data).
Styles live in static/css/, one file per component (see CSS_FILES).
"""
import importlib
import json
import mimetypes
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse

data = importlib.import_module(os.environ.get("DATA_MODULE", "real_data"))

PORT = int(os.environ.get("PORT", "8000"))
ROOT = Path(__file__).resolve().parent
STATIC_DIR = ROOT / "static"
AUDIO_FILE = ROOT / "data" / "briefing.mp3"

# Loaded in this order: base tokens first, then layout, then each component.
CSS_FILES = [
    "base.css",
    "layout.css",
    "panel.css",
    "table.css",
    "network.css",
    "research.css",
    "results.css",
]
CSS_LINKS = "\n".join(
    f'<link rel="stylesheet" href="/static/css/{name}">' for name in CSS_FILES
)

PAGE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Research Radar</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Manrope:wght@400;500;600;700&display=swap">
__CSS_LINKS__
</head>
<body>
  <header class="site-head">
    <h1>Research Radar</h1>
    <p class="sub">Where quant-finance research is heading, and how fast published trading signals stop working. Free public data only.</p>
  </header>
  <main>
    <div class="story" id="story"></div>

    <section class="panel net-panel">
      <div class="panel-head">
        <h2>Research map <span class="badge" id="net-badge" hidden>sample data</span></h2>
        <div class="net-controls">
          <button type="button" class="path-btn" id="path-btn" aria-pressed="false">Highlight largest area</button>
          <div class="month-ctl" id="month-ctl" hidden>
            <button type="button" class="play-btn" id="play-btn" aria-label="Play months">&#9654;</button>
            <input type="range" id="month-slider" min="0" max="0" value="0" aria-label="Month">
            <span class="month-label" id="month-label"></span>
          </div>
          <div class="toggle" id="week-toggle">
            <button type="button" data-week="last">Earlier</button>
            <button type="button" data-week="this" class="active">Recent</button>
          </div>
        </div>
      </div>
      <div class="sub" id="net-sub">Circle size shows how many papers each research area has in the period. Lines join areas whose papers cite the same references.</div>
      <div class="network" id="network"></div>
      <div class="net-legend" id="net-legend" hidden>
        <span class="sub">Colour: crowding this week in the live area it rolls up to</span>
        <span class="ramp"><span>quieter</span><i></i><span>more crowded</span></span>
        <span class="nt"><i></i>not tracked live</span>
      </div>
    </section>

    <section class="panel">
      <div class="panel-head">
        <h2>Crowded this week <span class="badge" id="research-badge" hidden>sample data</span></h2>
        <span class="sub" id="research-week"></span>
      </div>
      <div class="sub">New arXiv q-fin papers, tagged by Gemini. Score compares each area's share of the last 4 weeks with its usual share: 0 is normal, +2 unusually crowded.</div>
      <table class="data-table research" id="research">
        <thead><tr><th>Area</th><th class="num">Crowding</th><th class="num">Papers</th></tr></thead>
        <tbody></tbody>
      </table>
    </section>

    <div class="row3">
      <section class="panel">
        <div class="panel-head"><h2>Do published signals stop working?</h2></div>
        <div id="decay-headline" class="headline"></div>
        <table class="data-table" id="decay">
          <thead><tr><th>Signal</th><th class="num">Published</th><th class="num">Before</th><th class="num">After</th><th class="num">Change (95% CI)</th></tr></thead>
          <tbody></tbody>
        </table>
        <div class="sub">Mean monthly return before and after the paper, Ken French data. Bold: the interval excludes zero.</div>
      </section>

      <section class="panel">
        <div class="panel-head"><h2>Does trading on decay help?</h2><span class="sub" id="bt-range"></span></div>
        <div class="chart" id="bt-chart"></div>
        <table class="data-table" id="backtest">
          <thead><tr><th>Strategy</th><th class="num">Sharpe (95% CI)</th><th class="num">vs equal weight</th></tr></thead>
          <tbody></tbody>
        </table>
        <div class="sub">Cumulative return after costs. Walk-forward: each month uses only past returns. Market (table only) is before costs, for scale.</div>
      </section>

      <section class="panel">
        <div class="panel-head"><h2>This week's briefing</h2><span class="sub" id="brief-source"></span></div>
        <audio id="brief-audio" controls preload="none" hidden></audio>
        <div id="brief-card"></div>
        <details class="brief-text"><summary>Read the briefing</summary><p id="brief-text"></p></details>
      </section>
    </div>
  </main>

<script src="https://cdnjs.cloudflare.com/ajax/libs/d3/7.8.5/d3.min.js"></script>
<script>
const getJSON = url => fetch(url).then(r => r.json());
const esc = s => String(s).replace(/[&<>"]/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c]));
const parseDay = iso => new Date(iso + "T00:00:00");
const dayLabel = iso => parseDay(iso).toLocaleDateString("en-GB", { day: "numeric", month: "short", year: "numeric" });
const signed = (x, d = 2) => x == null ? "–" : (x > 0 ? "+" : x < 0 ? "−" : "") + Math.abs(x).toFixed(d);
const showBadge = (id, on) => { document.getElementById(id).hidden = !on; };

// Map nodes are coloured by the crowding score of the live area they roll up to
// (data/area_map.csv). Same +-3 range as the table bars; 0 is neutral.
const crowdColor = d3.scaleLinear().domain([-3, 0, 3])
  .range(["#5f7fa3", "#e9e2d7", "#8e2b30"]).clamp(true);
const NOT_TRACKED_FILL = "#c8c2ba";
const nodeFill = d => d.d_area == null ? null
  : d.crowding_score == null ? NOT_TRACKED_FILL : crowdColor(d.crowding_score);

// Set by drawNetwork: highlight the map nodes in one live area (null clears).
let highlightLiveArea = () => {};
let pinnedArea = null;

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

// How messy a layout is: lines crossing lines, plus lines running
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
  const tries = nodes.length > 16 ? 15 : 40;   // bigger graphs: fewer tries, still seeded
  for (let seed = 1; seed <= tries; seed++) {
    const rand = d3.randomLcg(seed / (tries + 1));
    const ns = nodes.map(n => ({ ...n, x: W * (0.2 + 0.6 * rand()), y: H * (0.2 + 0.6 * rand()) }));
    const ls = links.map(l => ({ ...l }));
    const sim = d3.forceSimulation(ns).randomSource(rand).stop()
      .force("link", d3.forceLink(ls).id(d => d.id).distance(d => 50 + d.source.rMax + d.target.rMax).strength(d => 0.1 + d.strength * 0.2))
      .force("charge", d3.forceManyBody().strength(nodes.length > 16 ? -260 : -500))
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

let NET = null;

function drawNetwork() {
  const net = NET, box = document.getElementById("network");
  // The box can measure 0 wide before layout settles; wait for a real size.
  const W = box.clientWidth, H = box.clientHeight, LINE = 12;
  if (W < 100 || H < 100) return;
  box.innerHTML = "";

  const monthly = net.mode === "monthly";
  const peak = d => monthly ? d3.max(d.series, p => p.share3) : Math.max(d.papers, d.papers_last_week);
  const maxPapers = d3.max(net.nodes, peak) || 1;
  const radius = d3.scaleSqrt().domain([0, maxPapers]).range([4, Math.min(W, H) / (net.nodes.length > 16 ? 14 : 11)]);
  let week = document.querySelector("#week-toggle .active").dataset.week;
  const papersFor = d => monthly ? d.series[MONTH].share3 : week === "this" ? d.papers : d.papers_last_week;
  const period = () => monthly ? net.months[MONTH].label
    : net.periods ? net.periods[week] : (week === "this" ? "this week" : "last week");

  const wrap = (text, max = 18) => text.split(" ").reduce((lines, word) => {
    const last = lines[lines.length - 1];
    if (last && (last + " " + word).length <= max) lines[lines.length - 1] = last + " " + word;
    else lines.push(word);
    return lines;
  }, []);

  const nodes = net.nodes.map(d => ({ ...d }));
  nodes.forEach(d => {
    d.lines = wrap(d.id);
    d.r = radius(papersFor(d));
    // lay out for the bigger of the two periods so switching never overlaps
    d.rMax = radius(peak(d));
    d.space = d.rMax + 10 + d.lines.length * LINE / 2;
  });

  const fit = d => {
    const half = Math.max(d.rMax, 46);
    d.x = Math.max(half + 6, Math.min(W - half - 6, d.x));
    d.y = Math.max(d.rMax + 10, Math.min(H - d.rMax - 6 - d.lines.length * LINE, d.y));
  };
  const stretch = (ns, padLo, padHi, size, get, set) => {
    const a = d3.least(ns, d => get(d)), b = d3.greatest(ns, d => get(d));
    const from0 = get(a), from1 = get(b);
    const to0 = padLo(a), to1 = size - padHi(b);
    if (from1 - from0 < 1 || to1 <= to0) return;
    ns.forEach(d => set(d, to0 + (get(d) - from0) * (to1 - to0) / (from1 - from0)));
  };
  const fillBox = ns => {
    stretch(ns, d => Math.max(d.rMax, 46) + 6, d => Math.max(d.rMax, 46) + 6, W,
            d => d.x, (d, v) => { d.x = v; });
    stretch(ns, d => d.rMax + 10, d => d.rMax + 6 + d.lines.length * LINE, H,
            d => d.y, (d, v) => { d.y = v; });
    ns.forEach(fit);
  };

  const pos = bestLayout(nodes, net.links, W, H, fit, fillBox);
  nodes.forEach(d => Object.assign(d, pos.get(d.id)));
  const byId = new Map(nodes.map(d => [d.id, d]));

  const svg = d3.select(box).append("svg")
    .attr("viewBox", [0, 0, W, H]).attr("width", "100%").attr("height", "100%");
  const directed = net.directed !== false;
  const defs = svg.append("defs");
  [["net-arrow", "net-arrowhead"], ["net-arrow-path", "net-arrowhead-path"]].forEach(([id, cls]) =>
    defs.append("marker")
      .attr("id", id).attr("viewBox", "0 -5 10 10")
      .attr("refX", 10).attr("refY", 0)
      .attr("markerWidth", 6).attr("markerHeight", 6).attr("orient", "auto")
      .append("path").attr("d", "M0,-5L10,0L0,5").attr("class", cls));
  const arrow = id => directed ? `url(#${id})` : null;

  const link = svg.append("g").selectAll("line").data(net.links).join("line")
    .attr("class", "net-link")
    .attr("stroke-width", d => 1 + d.strength * 3)
    .attr("marker-end", arrow("net-arrow"));
  link.append("title").text(l => l.shared ? `${l.source} – ${l.target}\n${l.shared} paper pairs share references` : "");

  const node = svg.append("g").selectAll("g").data(nodes).join("g")
    .attr("class", "net-node")
    .attr("transform", d => `translate(${d.x},${d.y})`);
  const circle = node.append("circle").attr("r", d => d.r)
    .style("fill", d => net.area_map ? nodeFill(d) : null)
    .classed("not-tracked", d => net.area_map && d.d_area != null && d.crowding_score == null);
  const label = node.append("text").attr("class", "net-label")
    .attr("text-anchor", "middle").attr("y", d => d.r + 12);
  label.selectAll("tspan").data(d => d.lines).join("tspan")
    .attr("x", 0).attr("dy", (t, i) => i === 0 ? 0 : LINE).text(t => t);
  const title = node.append("title");
  const liveLine = d => !net.area_map || d.d_area == null ? ""
    : d.crowding_score == null ? "\nNot tracked in the live arXiv feed"
    : `\nLive area: ${d.d_area} (crowding ${signed(d.crowding_score)} this week)`;
  const shareLine = d => d.share == null || d.share_before == null ? ""
    : `\nshare of papers: ${d.share_before}% (${net.periods.last}) → ${d.share}% (${net.periods.this})`;
  const monthTitle = d => {
    const p = d.series[MONTH], first = d.series[0], last = d.series[d.series.length - 1];
    return `${d.id}\n${period()}: ${p.share3}% of q-fin papers (3-month average)` +
      `\n${p.papers} paper${p.papers === 1 ? "" : "s"} this month (${p.share}%)` +
      `\ntrend: ${first.share3}% (${net.months[0].label}) → ${last.share3}% (${net.months[net.months.length - 1].label})` +
      (p.newest ? `\nnewest: ${p.newest}` : "") + (d.stage ? `\nlifecycle (OpenAlex): ${d.stage}` : "");
  };
  const setTitles = () => monthly ? title.text(monthTitle) : title.text(d => `${d.id}\n${papersFor(d)} papers in ${period()}` +
    shareLine(d) + (net.mode === "weekly" ? `\n${d.papers_last_week} last week → ${d.papers} this week` : "") +
    (d.newest ? `\nnewest this week: ${d.newest}` : "") +
    (d.stage ? `\nlifecycle (OpenAlex): ${d.stage}` : "") + liveLine(d));
  setTitles();

  const lineEnds = l => {
    const s = byId.get(l.source), t = byId.get(l.target);
    const dx = t.x - s.x, dy = t.y - s.y, dist = Math.hypot(dx, dy) || 1;
    return {
      x1: s.x + dx / dist * s.r, y1: s.y + dy / dist * s.r,
      x2: t.x - dx / dist * (t.r + (directed ? 2 : 0)), y2: t.y - dy / dist * (t.r + (directed ? 2 : 0)),
    };
  };
  const placeLinks = sel => sel
    .attr("x1", l => lineEnds(l).x1).attr("y1", l => lineEnds(l).y1)
    .attr("x2", l => lineEnds(l).x2).attr("y2", l => lineEnds(l).y2);
  placeLinks(link);

  // Hovering an area highlights it and the areas it is linked to.
  const linked = new Set(net.links.map(l => l.source + "|" + l.target));
  const isLinked = (a, b) => a === b || linked.has(a + "|" + b) || linked.has(b + "|" + a);
  let pathOn = false;
  const clearHl = () => {
    box.classList.remove("focus");
    node.classed("hl", false); link.classed("hl", false);
  };
  // Highlight every map node in one live area, and the links among them.
  highlightLiveArea = area => {
    if (pathOn) return;
    if (!area) { clearHl(); return; }
    const ids = new Set(nodes.filter(n => n.d_area === area).map(n => n.id));
    box.classList.add("focus");
    node.classed("hl", n => ids.has(n.id));
    link.classed("hl", l => ids.has(l.source) && ids.has(l.target));
  };
  const markRow = area => document.querySelectorAll("#research tbody tr")
    .forEach(tr => tr.classList.toggle("row-hl", !!area && tr.dataset.area === area));
  node.on("mouseenter", (e, d) => {
    if (pathOn) return;
    box.classList.add("focus");
    node.classed("hl", n => isLinked(d.id, n.id));
    link.classed("hl", l => l.source === d.id || l.target === d.id);
    if (net.area_map) markRow(d.d_area);
  }).on("mouseleave", () => {
    clearHl();
    markRow(pinnedArea);
    if (pinnedArea) highlightLiveArea(pinnedArea);
  });
  if (pinnedArea) highlightLiveArea(pinnedArea);   // keep a pinned row across redraws

  // Highlight the largest area in this period and the areas it links to.
  const pathBtn = document.getElementById("path-btn");
  const touches = (l, id) => l.source === id || (!directed && l.target === id);
  const drawPath = () => {
    box.classList.toggle("path", pathOn);
    pathBtn.classList.toggle("active", pathOn);
    pathBtn.setAttribute("aria-pressed", pathOn);
    pathBtn.textContent = pathOn ? "Clear highlight" : "Highlight largest area";
    const root = nodes.reduce((a, b) => papersFor(b) > papersFor(a) ? b : a);
    const hits = new Set(net.links.filter(l => touches(l, root.id))
      .map(l => l.source === root.id ? l.target : l.source));
    node.classed("path-root", d => pathOn && d === root)
        .classed("path-hit", d => pathOn && hits.has(d.id));
    link.classed("path-link", l => pathOn && touches(l, root.id))
        .attr("marker-end", l => pathOn && touches(l, root.id) ? arrow("net-arrow-path") : arrow("net-arrow"));
  };
  pathBtn.onclick = () => {
    pathOn = !pathOn;
    clearHl();
    if (!pathOn && pinnedArea) highlightLiveArea(pinnedArea);
    drawPath();
  };

  const resize = (ms = 600) => {
    nodes.forEach(d => { d.r = radius(papersFor(d)); });
    circle.transition().duration(ms).attr("r", d => d.r);
    label.transition().duration(ms).attr("y", d => d.r + 12);
    placeLinks(link.transition().duration(ms));
    setTitles();
    if (pathOn) drawPath();
  };
  showMonth = ms => { document.getElementById("month-label").textContent = period(); resize(ms); };
  if (monthly) showMonth(0);
  document.querySelectorAll("#week-toggle button").forEach(btn => {
    btn.onclick = () => {
      week = btn.dataset.week;
      document.querySelectorAll("#week-toggle button").forEach(b => b.classList.toggle("active", b === btn));
      resize();
    };
  });
}

// Monthly map: the slider picks the month; play steps through them.
let MONTH = 0, showMonth = () => {}, playTimer = null;
function setupMonths() {
  const slider = document.getElementById("month-slider"), play = document.getElementById("play-btn");
  const n = NET.months.length;
  slider.max = n - 1; slider.value = MONTH = n - 1;
  const stop = () => { clearInterval(playTimer); playTimer = null; play.innerHTML = "&#9654;"; };
  slider.oninput = () => { stop(); MONTH = +slider.value; showMonth(250); };
  play.onclick = () => {
    if (playTimer) return stop();
    if (MONTH >= n - 1) MONTH = 0;
    play.innerHTML = "&#10074;&#10074;";
    slider.value = MONTH; showMonth(0);
    playTimer = setInterval(() => {
      if (MONTH >= n - 1) return stop();
      slider.value = ++MONTH; showMonth(500);
    }, 700);
  };
  document.getElementById("month-ctl").hidden = false;
  document.getElementById("week-toggle").hidden = true;
  document.getElementById("net-sub").textContent =
    `How research moves, month by month: ${NET.arxiv_total.toLocaleString()} arXiv q-fin papers ` +
    `(${NET.months[0].label} – ${NET.months[n - 1].label}), tagged by Gemini into ${NET.nodes.length} areas. ` +
    `Circle size: the area's share of that month's papers (3-month average). Press ▶ to play. ` +
    `Lines join areas whose literatures cite the same work (${NET.papers_total.toLocaleString()} OpenAlex papers).`;
}

async function loadNetwork() {
  NET = await getJSON("/api/network");
  showBadge("net-badge", NET.sample);
  if (NET.periods) {
    const [early, recent] = document.querySelectorAll("#week-toggle button");
    early.textContent = NET.periods.last; recent.textContent = NET.periods.this;
    const live = new Set(NET.nodes.map(n => n.d_area).filter(a => a && a !== "Not tracked"));
    document.getElementById("net-sub").textContent =
      `${NET.papers_total.toLocaleString()} OpenAlex papers, a fixed random sample per year, so circle size is each area's share of research in the period. ` +
      "Lines join areas whose papers cite the same references (strongest links only)." +
      (NET.area_map ? ` The ${NET.nodes.length} map areas roll up into ${live.size} of the 8 live areas on the right; hover or click a row to find them.` : "");
  }
  document.getElementById("net-legend").hidden = !NET.area_map;
  if (NET.mode === "monthly") setupMonths();
  drawNetwork();
  // redraw when the box changes size (also covers a first layout at width 0)
  let lastW = 0, timer = null;
  new ResizeObserver(entries => {
    const w = Math.round(entries[0].contentRect.width);
    if (Math.abs(w - lastW) < 40) return;
    lastW = w; clearTimeout(timer); timer = setTimeout(drawNetwork, 150);
  }).observe(document.getElementById("network"));
}

// ---------------------------------------------------------------- crowding table

function paperChange(r) {
  const diff = r.papers_change, vs = `the same days last week (${r.papers_last_week})`;
  if (diff > 0) return `<span class="arrow up" title="Up ${diff} on ${vs}">&#9650;</span>`;
  if (diff < 0) return `<span class="arrow down" title="Down ${-diff} on ${vs}">&#9660;</span>`;
  return `<span class="arrow same" title="Same as ${vs}">&ndash;</span>`;
}

// Bar centred on zero: right for crowded, left for quiet. Scale: +-3.
function scoreBar(x) {
  const w = Math.min(Math.abs(x) / 3, 1) * 50;
  const side = x >= 0 ? `left:50%;width:${w}%` : `left:${50 - w}%;width:${w}%`;
  return `<span class="zbar"><i class="${x >= 0 ? "pos" : "neg"}${Math.abs(x) >= 2 ? " strong" : ""}" style="${side}"></i></span>`;
}

async function loadResearch() {
  const snap = await getJSON("/api/research");
  showBadge("research-badge", snap.sample);
  document.getElementById("research-week").textContent = "Week of " + dayLabel(snap.week_start) +
    (snap.papers_compare ? " · papers " + snap.papers_compare : "");
  document.querySelector("#research tbody").innerHTML = snap.topics.map(r => {
    const tip = `${r.papers_4w} papers in the last 4 weeks` +
      (r.newest ? `\nNewest: ${r.newest.title} (${r.newest.published})` : "");
    return `
    <tr title="${esc(tip)}" data-area="${esc(r.topic)}">
      <td>${esc(r.topic)}<span class="row-note"></span></td>
      <td class="num"><div class="pop-bar">${signed(r.score)} ${scoreBar(r.score ?? 0)}</div></td>
      <td class="num papers">${r.papers} ${paperChange(r)}</td>
    </tr>`;
  }).join("");
}

// Link table rows to the map: hover a row to light up its map areas, click to pin.
// Rows with no map area (too new or too narrow for 2000-2026) get a short note.
function linkTableToMap() {
  if (!NET || !NET.area_map) return;
  const counts = d3.rollup(NET.nodes, v => v.length, n => n.d_area);
  document.querySelectorAll("#research tbody tr").forEach(tr => {
    const area = tr.dataset.area, n = counts.get(area) || 0;
    tr.classList.add("linked");
    tr.querySelector(".row-note").textContent = n ? ` · ${n} on map`
      : " · not on the map: too new or too narrow for 2000–2026";
    tr.onmouseenter = () => highlightLiveArea(area);
    tr.onmouseleave = () => highlightLiveArea(pinnedArea);
    tr.onclick = () => {
      pinnedArea = pinnedArea === area ? null : area;
      document.querySelectorAll("#research tbody tr")
        .forEach(t => t.classList.toggle("row-hl", t.dataset.area === pinnedArea));
      highlightLiveArea(pinnedArea);
    };
  });
}

// ---------------------------------------------------------------- summary strip
// The brief's questions: what's hidden, how we found it, what predicts, the decision.

async function loadSummary() {
  const s = await getJSON("/api/summary");
  const cards = [];
  const h = s.hidden;
  if (h) cards.push(`
    <div class="card"><span class="step">What's hidden</span>
      <p class="big">Published trading signals keep only <b>${h.after_publication}%</b> of their in-sample return after publication.</p>
      <p class="sub">${h.n_signals} signals: ${h.after_sample}% left once the paper's sample ends, a further drop at publication (p = ${h.publication_p}).
      Fake publication dates don't reproduce it (${h.placebo_p === 0 ? "p < 0.0001" : "p = " + h.placebo_p}).</p>
      ${h.crowded_area ? `<p class="sub">And where researchers are crowding in this week: <b>${esc(h.crowded_area)}</b> (${signed(h.crowded_score, 1)}).</p>` : ""}
    </div>`);
  if (s.inputs.length) cards.push(`
    <div class="card"><span class="step">How we found it: free public data</span>
      <ul class="inputs">${s.inputs.map(i => `<li><b>${i.n.toLocaleString()}</b> ${esc(i.what)} <span class="sub">${esc(i.detail)}</span></li>`).join("")}</ul>
      <p class="sub">Market data and alternative data (papers, citations) combined.</p>
    </div>`);
  const p = s.predict, rows = [];
  if (p.momentum) rows.push(`<li class="ok">Next month's factor momentum: past winners beat losers by <b>${p.momentum.spread_pct}%</b> a month (t = ${p.momentum.t}, ${p.momentum.months} months).</li>`);
  if (p.volatility) rows.push(`<li class="ok">Next month's volatility: the model beats a simple baseline (t = ${p.volatility.t})${p.volatility.sharpe_managed ? `; volatility-managed Sharpe ${p.volatility.sharpe_managed} vs ${p.volatility.sharpe_plain}, before costs` : ""}.</li>`);
  if (p.decay_model) rows.push(`<li class="no">Which signal will decay: ${p.decay_model.beat_baseline} of ${p.decay_model.n_models} models beat a constant guess.</li>`);
  if (p.attention) {
    const moved = p.attention_moved && p.attention_moved[0];
    rows.push(`<li class="no">${moved ? `Research attention moved to ${esc(moved.area)} (${moved.before}% → ${moved.after}% of papers), but a` : "A"}ttention doesn't predict decay (ρ = ${p.attention.rho}, p = ${p.attention.p}, ${p.attention.n} signals).</li>`);
  }
  if (rows.length) cards.push(`
    <div class="card"><span class="step">What we can predict, and what we can't</span><ul class="checks">${rows.join("")}</ul></div>`);
  const d = s.decide;
  if (d) {
    const held = d.held.map(x => `${esc(x.signal)} ${Math.round(x.weight * 100)}%`).join(", ") || "cash";
    cards.push(`
    <div class="card decide"><span class="step">The decision</span>
      <p class="big">Hold for ${esc(d.for_month)}: <b>${held}</b>.</p>
      ${d.naive ? `<p class="sub">Trusting the paper's numbers is <b>reliably worse</b> than equal weight after costs: ${signed(d.naive.vs_equal)} Sharpe (95% CI ${signed(d.naive.vs_equal_ci[0])} to ${signed(d.naive.vs_equal_ci[1])}).</p>` : ""}
      ${d.tilt ? `<p class="sub">Our decay rule: ${signed(d.tilt.vs_equal)} vs equal weight (${signed(d.tilt.vs_equal_ci[0])} to ${signed(d.tilt.vs_equal_ci[1])}): not proven, so we say so.</p>` : ""}
    </div>`);
  }
  document.getElementById("story").innerHTML = cards.join("");
}

// ---------------------------------------------------------------- decay (A)

async function loadDecay() {
  const d = await getJSON("/api/decay");
  const h = d.headline;
  if (h) {
    const p = h.placebo_p === 0 ? "p < 0.0001" : `p = ${h.placebo_p}`;
    document.getElementById("decay-headline").innerHTML = `
      <div class="stats">
        <div><b>100%</b><span>in-sample return</span></div>
        <div><b>${h.after_sample}%</b><span>left after the sample ends</span></div>
        <div><b>${h.after_publication}%</b><span>left after publication</span></div>
      </div>
      <p class="sub">Across ${h.n_signals} published signals (Open Source Asset Pricing). The extra drop at publication
      is significant (p = ${h.publication_p}); fake publication dates give a much smaller drop (${h.placebo_n} signals, ${p}).
      Replicates McLean &amp; Pontiff (2016).</p>`;
  }
  document.querySelector("#decay tbody").innerHTML = d.signals.map(s => `
    <tr class="${s.clear ? "clear" : ""}">
      <td>${esc(s.signal)}</td>
      <td class="num">${s.published}</td>
      <td class="num">${s.pre.toFixed(2)}%</td>
      <td class="num">${s.post.toFixed(2)}%</td>
      <td class="num change">${signed(s.post - s.pre)}<span class="sub">${signed(s.diff_low)} to ${signed(s.diff_high)}</span></td>
    </tr>`).join("");
}

// ---------------------------------------------------------------- backtest (D)

function drawBacktest(bt) {
  const box = document.getElementById("bt-chart");
  box.innerHTML = "";
  const W = box.clientWidth || 360, H = 170, M = { t: 8, r: 8, b: 20, l: 34 };
  const parse = d3.utcParse("%Y-%m-%d");
  const series = bt.series.map(s => ({ ...s, pts: s.points.map(([d, v]) => [parse(d), v]) }));
  const all = series.flatMap(s => s.pts);
  const x = d3.scaleUtc().domain(d3.extent(all, p => p[0])).range([M.l, W - M.r]);
  const y = d3.scaleLinear().domain(d3.extent(all, p => p[1])).nice().range([H - M.b, M.t]);
  const svg = d3.select(box).append("svg").attr("viewBox", [0, 0, W, H]).attr("width", "100%");
  svg.append("g").attr("class", "axis").attr("transform", `translate(0,${H - M.b})`)
    .call(d3.axisBottom(x).ticks(5).tickSizeOuter(0));
  svg.append("g").attr("class", "axis").attr("transform", `translate(${M.l},0)`)
    .call(d3.axisLeft(y).ticks(4).tickFormat(d3.format(".0%")).tickSizeOuter(0));
  svg.append("line").attr("class", "zero").attr("x1", M.l).attr("x2", W - M.r)
    .attr("y1", y(0)).attr("y2", y(0));
  const line = d3.line().x(p => x(p[0])).y(p => y(p[1]));
  svg.append("g").selectAll("path").data(series).join("path")
    .attr("class", s => "series " + s.key).attr("d", s => line(s.pts));
  const legend = d3.select(box).append("div").attr("class", "legend");
  series.forEach(s => legend.append("span").attr("class", "key " + s.key).text(s.label));
}

async function loadBacktest() {
  const bt = await getJSON("/api/backtest");
  document.getElementById("bt-range").textContent = `${bt.start.slice(0, 4)}–${bt.end.slice(0, 4)}`;
  drawBacktest(bt);
  document.querySelector("#backtest tbody").innerHTML = bt.rows.map(r => `
    <tr class="${r.key === "decay_tilt" ? "ours" : ""}">
      <td>${esc(r.label)}</td>
      <td class="num">${r.sharpe.toFixed(2)} (${signed(r.ci[0])}, ${signed(r.ci[1])})</td>
      <td class="num">${r.vs_equal == null ? "" :
        `<span class="${r.vs_equal_ci[0] > 0 || r.vs_equal_ci[1] < 0 ? "sig" : ""}">${signed(r.vs_equal)}</span>
         <span class="sub">(${signed(r.vs_equal_ci[0])}, ${signed(r.vs_equal_ci[1])})</span>`}</td>
    </tr>`).join("");
}

// ---------------------------------------------------------------- briefing (D)

async function loadBriefing() {
  const b = await getJSON("/api/briefing");
  if (b.audio) {
    const audio = document.getElementById("brief-audio");
    audio.src = "/media/briefing.mp3"; audio.hidden = false;
  }
  document.getElementById("brief-text").textContent = b.text;
  const c = b.card;
  if (!c) return;
  const src = (c.text_source || "").startsWith("gemini") ? "worded by Gemini, numbers checked" : "template text";
  document.getElementById("brief-source").textContent = "Week of " + dayLabel(c.week) + " · " + src;
  const a = c.allocation;
  const held = a.held.length ? a.held.map(h => `<li><b>${esc(h.signal)}</b> ${Math.round(h.weight * 100)}%</li>`).join("")
                             : "<li>Cash: no signal clears the cost hurdle</li>";
  const cv = c.caveat;
  document.getElementById("brief-card").innerHTML = `
    <div class="card-grid">
      <div><span class="sub">Most crowded vs usual</span>
        <b>${esc(c.busiest[0].area)}</b> <span class="sub">${signed(c.busiest[0].score, 1)}, ${c.busiest[0].papers_4w} papers / 4 wks</span></div>
      <div><span class="sub">Quietest vs usual</span>
        <b>${esc(c.quietest.area)}</b> <span class="sub">${signed(c.quietest.score, 1)}</span></div>
      <div><span class="sub">Hold for ${esc(a.for_month)}</span><ul class="held">${held}</ul>
        <span class="sub">Skipped: ${a.skipped.map(esc).join(", ") || "none"}</span></div>
      <div><span class="sub">Caveat</span>
        <span>Beat equal weight by ${signed(cv.sharpe_minus_equal_weight)} Sharpe (95% CI ${signed(cv.ci_low)} to ${signed(cv.ci_high)})${cv.significant ? "" : ": not proven"}.</span></div>
    </div>`;
}

Promise.all([loadNetwork(), loadResearch()]).then(linkTableToMap);
loadSummary(); loadDecay(); loadBacktest(); loadBriefing();
</script>
</body>
</html>
""".replace("__CSS_LINKS__", CSS_LINKS)


ROUTES = {
    "/api/network": "get_topic_network",
    "/api/research": "get_research_snapshot",
    "/api/decay": "get_decay",
    "/api/backtest": "get_backtest",
    "/api/briefing": "get_briefing",
    "/api/summary": "get_summary",
}


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

    def log_message(self, fmt, *args):  # keep the console quiet during demos
        pass

    def do_GET(self):
        url = urlparse(self.path)
        if url.path in ("/", "/index.html"):
            self._send(200, PAGE, "text/html; charset=utf-8")
        elif url.path.startswith("/static/"):
            self._static(unquote(url.path[len("/static/"):]))
        elif url.path == "/media/briefing.mp3" and AUDIO_FILE.is_file():
            self._send(200, AUDIO_FILE.read_bytes(), "audio/mpeg")
        elif url.path in ROUTES and hasattr(data, ROUTES[url.path]):
            try:
                self._json(getattr(data, ROUTES[url.path])())
            except Exception as exc:  # one broken file must not take the page down
                self._json({"error": f"{type(exc).__name__}: {exc}"}, 500)
        else:
            self._json({"error": "not found"}, 404)


if __name__ == "__main__":
    server = ThreadingHTTPServer(("127.0.0.1", PORT), Handler)
    print(f"Research Radar running at http://localhost:{PORT}")
    server.serve_forever()
