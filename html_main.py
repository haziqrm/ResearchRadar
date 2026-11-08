"""Stock tracker and research prediction website.

Run:  py html_main.py      then open http://localhost:8000

The page loads its data from JSON endpoints. Those endpoints call
synthetic_data.py, so swapping in real data means changing that module only.

  GET /api/network     topics (sized by paper count) and knock-on links
  GET /api/research    this week's research topic snapshot
  GET /api/account     account value summary (not shown on the page now)
  GET /api/stocks      most traded stocks today (not shown on the page now)

The knock-on links are read from data/topic_links.csv.
Styles live in static/css/, one file per component (see CSS_FILES).
"""
import json
import mimetypes
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse

import synthetic_data as data  # swap this import for the live data module

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
          <div class="toggle" id="week-toggle">
            <button type="button" data-week="last">Last week</button>
            <button type="button" data-week="this" class="active">This week</button>
          </div>
        </div>
      </div>
      <div class="sub">Circle size shows how many research papers are out on a topic. Arrows show knock-on effects of one topic on another. Show affected path highlights the largest topic and the topics it affects directly.</div>
      <div class="network" id="network"></div>
    </section>

    <section class="panel">
      <div class="panel-head">
        <h2>Research topics</h2>
        <span class="sub" id="research-week"></span>
      </div>
      <table class="data-table research" id="research">
        <thead><tr><th>Topic</th><th class="num">Popularity</th><th class="num">Papers</th></tr></thead>
        <tbody></tbody>
      </table>
    </section>
  </main>

<script src="https://cdnjs.cloudflare.com/ajax/libs/d3/7.8.5/d3.min.js"></script>
<script>
const getJSON = url => fetch(url).then(r => r.json());
const esc = s => String(s).replace(/[&<>"]/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c]));
const parseDay = iso => new Date(iso + "T00:00:00");
const dayLabel = iso => parseDay(iso).toLocaleDateString("en-GB", { day: "numeric", month: "short", year: "numeric" });

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

async function loadNetwork() {
  const net = await getJSON("/api/network");
  const box = document.getElementById("network");
  const W = box.clientWidth, H = box.clientHeight, LINE = 12;

  const maxPapers = d3.max(net.nodes, n => Math.max(n.papers, n.papers_last_week));
  const radius = d3.scaleSqrt().domain([0, maxPapers]).range([5, Math.min(W, H) / 11]);
  let week = "this";
  const papersFor = d => week === "this" ? d.papers : d.papers_last_week;

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
    d.r = radius(papersFor(d));
    // lay out for the bigger of the two weeks so switching never overlaps
    d.rMax = radius(Math.max(d.papers, d.papers_last_week));
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

  const defs = svg.append("defs");
  [["net-arrow", "net-arrowhead"], ["net-arrow-path", "net-arrowhead-path"]].forEach(([id, cls]) =>
    defs.append("marker")
      .attr("id", id).attr("viewBox", "0 -5 10 10")
      .attr("refX", 10).attr("refY", 0)
      .attr("markerWidth", 6).attr("markerHeight", 6).attr("orient", "auto")
      .append("path").attr("d", "M0,-5L10,0L0,5").attr("class", cls));

  const link = svg.append("g").selectAll("line").data(net.links).join("line")
    .attr("class", "net-link")
    .attr("stroke-width", d => 1 + d.strength * 3)
    .attr("marker-end", "url(#net-arrow)");

  const node = svg.append("g").selectAll("g").data(net.nodes).join("g")
    .attr("class", "net-node")
    .attr("transform", d => `translate(${d.x},${d.y})`);
  const circle = node.append("circle").attr("r", d => d.r);
  const label = node.append("text").attr("class", "net-label")
    .attr("text-anchor", "middle").attr("y", d => d.r + 13);
  label.selectAll("tspan").data(d => d.lines).join("tspan")
    .attr("x", 0).attr("dy", (t, i) => i === 0 ? 0 : LINE).text(t => t);
  const title = node.append("title");
  const setTitles = () => title.text(d => `${d.id}\n${papersFor(d)} papers ${week === "this" ? "this" : "last"} week`);
  setTitles();

  // stop each line at the edge of the circle so the arrowhead shows
  const lineEnds = l => {
    const s = byId.get(l.source), t = byId.get(l.target);
    const dx = t.x - s.x, dy = t.y - s.y, dist = Math.hypot(dx, dy) || 1;
    return {
      x1: s.x + dx / dist * s.r, y1: s.y + dy / dist * s.r,
      x2: t.x - dx / dist * (t.r + 2), y2: t.y - dy / dist * (t.r + 2),
    };
  };
  const placeLinks = sel => sel
    .attr("x1", l => lineEnds(l).x1).attr("y1", l => lineEnds(l).y1)
    .attr("x2", l => lineEnds(l).x2).attr("y2", l => lineEnds(l).y2);
  placeLinks(link);

  // Hovering a topic highlights it and the topics it is linked to.
  const linked = new Set(net.links.map(l => l.source + "|" + l.target));
  const isLinked = (a, b) => a === b || linked.has(a + "|" + b) || linked.has(b + "|" + a);
  let pathOn = false;
  node.on("mouseenter", (e, d) => {
    if (pathOn) return;
    box.classList.add("focus");
    node.classed("hl", n => isLinked(d.id, n.id));
    link.classed("hl", l => l.source === d.id || l.target === d.id);
  }).on("mouseleave", () => {
    box.classList.remove("focus");
    node.classed("hl", false); link.classed("hl", false);
  });

  // Show affected path: the largest circle this week, and the circles its
  // arrows point to (the topics it has a knock-on effect on).
  const pathBtn = document.getElementById("path-btn");
  const drawPath = () => {
    box.classList.toggle("path", pathOn);
    pathBtn.classList.toggle("active", pathOn);
    pathBtn.setAttribute("aria-pressed", pathOn);
    pathBtn.textContent = pathOn ? "Hide affected path" : "Show affected path";
    const root = net.nodes.reduce((a, b) => papersFor(b) > papersFor(a) ? b : a);
    const hits = new Set(net.links.filter(l => l.source === root.id).map(l => l.target));
    node.classed("path-root", d => pathOn && d === root)
        .classed("path-hit", d => pathOn && hits.has(d.id));
    link.classed("path-link", l => pathOn && l.source === root.id)
        .attr("marker-end", l => pathOn && l.source === root.id ? "url(#net-arrow-path)" : "url(#net-arrow)");
  };
  pathBtn.addEventListener("click", () => {
    pathOn = !pathOn;
    box.classList.remove("focus"); node.classed("hl", false); link.classed("hl", false);
    drawPath();
  });

  // Last week / this week: circles grow or shrink to that week's paper count.
  document.querySelectorAll("#week-toggle button").forEach(btn =>
    btn.addEventListener("click", () => {
      week = btn.dataset.week;
      document.querySelectorAll("#week-toggle button").forEach(b => b.classList.toggle("active", b === btn));
      net.nodes.forEach(d => { d.r = radius(papersFor(d)); });
      circle.transition().duration(600).attr("r", d => d.r);
      label.transition().duration(600).attr("y", d => d.r + 13);
      placeLinks(link.transition().duration(600));
      setTitles();
      if (pathOn) drawPath();  // the largest circle can change between weeks
    }));
}

// ---------------------------------------------------------------- research table

function paperChange(r) {
  const diff = r.papers_change;
  if (diff > 0) return `<span class="arrow up" title="Up ${diff} from last week">&#9650;</span>`;
  if (diff < 0) return `<span class="arrow down" title="Down ${-diff} from last week">&#9660;</span>`;
  return `<span class="arrow same" title="Same as last week">&ndash;</span>`;
}

async function loadResearch() {
  const snap = await getJSON("/api/research");
  document.getElementById("research-week").textContent = "Week of " + dayLabel(snap.week_start);
  document.querySelector("#research tbody").innerHTML = snap.topics.map(r => `
    <tr>
      <td>${esc(r.topic)}</td>
      <td class="num"><div class="pop-bar">${r.popularity}
        <span class="track"><i class="fill" style="width:${r.popularity}%"></i></span></div></td>
      <td class="num papers">${r.papers} ${paperChange(r)}</td>
    </tr>`).join("");
}

loadNetwork(); loadResearch();
</script>
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
            "/api/account": data.get_account,
            "/api/stocks": data.get_most_traded,
        }
        if url.path in ("/", "/index.html"):
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
