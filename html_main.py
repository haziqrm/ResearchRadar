"""Research topic network website.

Run:  py html_main.py      then open http://localhost:8000

The page loads its data from JSON endpoints. Those endpoints call
research_data.py, which reads the research map's CSVs in data/ and keeps
the 8 trading-relevant areas (see research_data.AREAS).

  GET /api/network     areas (sized by article count) and the links between them
  GET /api/research    articles per area, in total and this year vs last year

Styles live in static/css/, one file per component (see CSS_FILES).
"""
import json
import mimetypes
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

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
]
CSS_LINKS = "\n".join(
    f'<link rel="stylesheet" href="/static/css/{name}">' for name in CSS_FILES
)

PAGE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Stock Tracker</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Manrope:wght@400;500;600;700&display=swap">
__CSS_LINKS__
</head>
<body>
  <main>
    <section class="panel">
      <div class="panel-head">
        <h2>Research topic network</h2>
        <div class="net-controls">
          <button type="button" class="path-btn" id="path-btn" aria-pressed="false">Show affected path</button>
          <div class="year-control">
            <button type="button" class="play-btn" id="play-btn" aria-label="Play years">&#9654;</button>
            <input type="range" id="year-slider" step="1" aria-label="Year">
            <span class="year-label" id="year-label"></span>
          </div>
        </div>
      </div>
      <div class="sub">Circle size shows how many research articles came out on a topic in the selected year. Lines link topics whose papers cite the same earlier work; thicker means more shared that year, faint means none. Drag the slider or press play to watch the years change.</div>
      <div class="network" id="network"></div>
    </section>

    <section class="panel">
      <div class="panel-head">
        <h2>Research topics</h2>
        <span class="sub" id="research-range"></span>
      </div>
      <table class="data-table research" id="research">
        <thead><tr><th>Topic</th><th class="num">Articles</th><th class="num" id="year-col">This year</th></tr></thead>
        <tbody></tbody>
      </table>
    </section>

    <!-- Person A: topic drill-down (radar.js). Click a circle above, or pick a topic. -->
    <section class="panel radar-wide" id="zoom-panel">
      <div class="panel-head">
        <h2 id="zoom-title">Inside a topic</h2>
        <select id="zoom-select" aria-label="Topic"></select>
      </div>
      <div class="sub" id="zoom-sub">Click a circle on the map (or pick a topic here) to open its sub-topic map: one island per sub-topic, with similar sub-topics placed close together. Each dot is a paper, and inside an island similar papers sit together (t-SNE on SPECTER2 paper vectors). Island size follows the last five years of papers, so press play to watch sub-topics grow, shrink and emerge. Arrows are direct citations between sub-topics (from the cited to the citing side), with moving dots showing the direction. Click an island to see who it built on and who it fed; hover a paper for its citations; click a paper for its citation trail. Rings mark suggested reading. Citations show who builds on whom, not cause and effect.</div>
      <div class="zoom-side" id="zoom-side"></div>
    </section>

    <!-- Person A: what's next, signal decay, and how each claim was tested (radar.js) -->
    <section class="panel radar-wide" id="next-panel">
      <div class="panel-head"><h2>What could be next</h2><span class="badge warn">exploratory</span></div>
      <div class="sub" id="next-sub"></div>
      <div class="next-grid">
        <div><h3>Topics</h3><ol class="rank-list" id="next-topics"></ol></div>
        <div><h3>Sub-topics</h3><ol class="rank-list" id="next-subtopics"></ol></div>
      </div>
    </section>

    <section class="panel radar-wide" id="decay-panel">
      <div class="panel-head"><h2>What happens to an edge once it is published</h2></div>
      <div class="sub">Average return of 208 published trading signals, as a share of their in-sample return, by years since publication. Shaded: 95% interval.</div>
      <div class="decay-body">
        <div class="decay-plot" id="decay-plot"></div>
        <div class="haircut">
          <h3>Haircut calculator</h3>
          <label>Paper claims <input type="number" id="hc-input" value="1.0" step="0.1" min="0"> % a month</label>
          <div id="hc-out"></div>
          <div class="sub" id="hc-caveat"></div>
        </div>
      </div>
    </section>

    <section class="panel radar-wide" id="signals-panel">
      <div class="panel-head"><h2>Signals right now</h2><span class="sub" id="signals-asof"></span></div>
      <div class="sub">Momentum: average return over the last 12 months (past winners kept winning, H9). Volatility: next month's forecast (H10 model). Kept: share of the edge that survived publication.</div>
      <table class="data-table" id="signals-table">
        <thead><tr><th>Signal</th><th class="num">12-month mean</th><th class="num">Vol forecast</th><th class="num">Kept after publication</th></tr></thead>
        <tbody></tbody>
      </table>
      <div class="sub" id="portfolio-line"></div>
    </section>

    <section class="panel radar-wide" id="tests-panel">
      <div class="panel-head"><h2>How we tested it</h2></div>
      <div class="sub">Every claim was tested on data the model had not seen, against simple rules, with the pass rule written down before the run. For the forecasts, the grey bars are 200 simulated worlds where no topic truly grows; the line is the real result.</div>
      <div class="tests-grid" id="tests-grid"></div>
    </section>
  </main>

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

async function loadNetwork() {
  const net = await getJSON("/api/network");
  const box = document.getElementById("network");
  const W = box.clientWidth, H = box.clientHeight, LINE = 12;
  const years = net.slider_years;
  const countFor = (d, y = yearState.year) => d.by_year[String(y)] || 0;

  // One scale for every year, so circle sizes can be compared across years.
  const R_MIN = 4, R_MAX = Math.min(W, H) / 9;
  const peak = d3.max(net.nodes, d => d3.max(years, y => countFor(d, y))) || 1;
  const radius = d3.scaleSqrt().domain([0, peak]).range([R_MIN, R_MAX]);

  // Line thickness: that year's shared links, on one scale across years.
  const peakLink = d3.max(net.links, l => d3.max(years, y => countFor(l, y))) || 1;
  const width = n => n === 0 ? 1 : 1.5 + 6 * n / peakLink;

  // Label each circle with the same topic name as the research table,
  // wrapped onto short lines under the circle.
  const wrap = (text, max = 18) => text.split(" ").reduce((lines, word) => {
    const last = lines[lines.length - 1];
    if (last && (last + " " + word).length <= max) lines[lines.length - 1] = last + " " + word;
    else lines.push(word);
    return lines;
  }, []);

  net.nodes.forEach(d => {
    d.lines = wrap(d.id);
    // lay out for the biggest the circle gets in any year, so it never overlaps
    d.rMax = radius(d3.max(years, y => countFor(d, y)));
    d.r = d.rMax;
    d.space = d.rMax + 14 + d.lines.length * LINE / 2;
  });

  // keep circles and their labels inside the box
  const fit = d => {
    const half = Math.max(d.rMax, 50);
    d.x = Math.max(half + 8, Math.min(W - half - 8, d.x));
    d.y = Math.max(d.rMax + 12, Math.min(H - d.rMax - 8 - d.lines.length * LINE, d.y));
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
    stretch(ns, d => d.rMax + 12, d => d.rMax + 8 + d.lines.length * LINE, H,
            d => d.y, (d, v) => { d.y = v; });
    ns.forEach(fit);
  };

  const pos = bestLayout(net.nodes, net.links, W, H, fit, fillBox);
  net.nodes.forEach(d => Object.assign(d, pos.get(d.id)));
  const byId = new Map(net.nodes.map(d => [d.id, d]));

  const svg = d3.select(box).append("svg")
    .attr("viewBox", [0, 0, W, H]).attr("width", "100%").attr("height", "100%");

  // Big faint year in the background, so you can see where the animation is.
  const yearMark = svg.append("text").attr("class", "net-year")
    .attr("x", W - 12).attr("y", H - 12).attr("text-anchor", "end");

  // Links have no direction: two areas whose papers cite the same earlier work.
  const link = svg.append("g").selectAll("line").data(net.links).join("line")
    .attr("class", "net-link");
  const linkTitle = link.append("title");

  const node = svg.append("g").selectAll("g").data(net.nodes).join("g")
    .attr("class", "net-node")
    .attr("transform", d => `translate(${d.x},${d.y})`);
  const circle = node.append("circle");
  const label = node.append("text").attr("class", "net-label").attr("text-anchor", "middle");
  label.selectAll("tspan").data(d => d.lines).join("tspan")
    .attr("x", 0).attr("dy", (t, i) => i === 0 ? 0 : LINE).text(t => t);
  const nodeTitle = node.append("title");

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

  // Links count as "there" in a year only if that year had shared work.
  const active = l => countFor(l) > 0;
  const touches = (l, id) => l.source === id || l.target === id;
  const neighbours = id => new Set(net.links.filter(l => active(l) && touches(l, id))
    .map(l => l.source === id ? l.target : l.source));

  // Hovering a topic highlights it and the topics it is linked to that year.
  let pathOn = false;
  node.on("mouseenter", (e, d) => {
    if (pathOn) return;
    const near = neighbours(d.id);
    box.classList.add("focus");
    node.classed("hl", n => n.id === d.id || near.has(n.id));
    link.classed("hl", l => active(l) && touches(l, d.id));
  }).on("mouseleave", () => {
    box.classList.remove("focus");
    node.classed("hl", false); link.classed("hl", false);
  });

  // Show affected path: the largest circle in the selected year, and the
  // topics linked to it that year.
  const pathBtn = document.getElementById("path-btn");
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
  pathBtn.addEventListener("click", () => {
    pathOn = !pathOn;
    box.classList.remove("focus"); node.classed("hl", false); link.classed("hl", false);
    drawPath();
  });

  // Redraw everything for one year.
  const drawYear = (year, ms) => {
    net.nodes.forEach(d => { d.r = radius(countFor(d, year)); });
    const t = sel => ms ? sel.transition().duration(ms) : sel;
    t(circle).attr("r", d => d.r);
    t(label).attr("y", d => d.r + 13);
    t(link).attr("stroke-width", l => width(countFor(l, year)));
    placeLinks(t(link));
    link.classed("quiet", l => countFor(l, year) === 0);
    yearMark.text(year);
    nodeTitle.text(d => `${d.id}\n${countFor(d, year)} articles in ${year}\n${d.articles} in total`);
    linkTitle.text(l => `${l.source} and ${l.target}\n${countFor(l, year)} paper pairs in ${year} cite the same earlier work`);
    if (pathOn) drawPath();  // the largest circle can change between years
  };
  document.addEventListener("yearchange", e => drawYear(e.detail, playing ? STEP_MS * 0.8 : 400));

  // ---- slider and play button
  const slider = document.getElementById("year-slider");
  const yearLabel = document.getElementById("year-label");
  const playBtn = document.getElementById("play-btn");
  slider.min = years[0];
  slider.max = years[years.length - 1];
  slider.value = slider.max;

  const STEP_MS = 900;
  let playing = null;
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

  drawYear(+slider.value, 0);
  yearLabel.textContent = slider.value;
  yearState.year = +slider.value;
}

// ---------------------------------------------------------------- research table

function yearChange(now, before, prevYear) {
  const diff = now - before;
  if (diff > 0) return `<span class="arrow up" title="Up ${diff} from ${prevYear}">&#9650;</span>`;
  if (diff < 0) return `<span class="arrow down" title="Down ${-diff} from ${prevYear}">&#9660;</span>`;
  return `<span class="arrow same" title="Same as ${prevYear}">&ndash;</span>`;
}

async function loadResearch() {
  const snap = await getJSON("/api/research");
  const most = Math.max(...snap.topics.map(r => r.articles), 1);
  document.getElementById("research-range").textContent = `Articles ${snap.first_year}–${snap.this_year}`;
  const count = (r, y) => r.by_year[String(y)] || 0;

  // The last column follows the slider: that year's articles, arrow vs the year before.
  const render = year => {
    document.getElementById("year-col").textContent = year;
    document.querySelector("#research tbody").innerHTML = snap.topics.map(r => `
      <tr>
        <td>${esc(r.topic)}</td>
        <td class="num"><div class="pop-bar">${r.articles}
          <span class="track"><i class="fill" style="width:${(100 * r.articles / most).toFixed(1)}%"></i></span></div></td>
        <td class="num papers" title="${count(r, year - 1)} in ${year - 1}">${count(r, year)} ${yearChange(count(r, year), count(r, year - 1), year - 1)}</td>
      </tr>`).join("");
  };
  document.addEventListener("yearchange", e => render(e.detail));
  render(yearState.year || snap.this_year);
}

loadNetwork(); loadResearch();
</script>
<script src="/static/js/radar.js"></script>
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
        }
        if url.path == "/api/topic":
            name = parse_qs(url.query).get("name", [""])[0]
            self._json(radar_data.get_topic(name))
        elif url.path in ("/", "/index.html"):
            self._send(200, PAGE, "text/html; charset=utf-8")
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
