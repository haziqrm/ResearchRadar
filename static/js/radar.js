// Radar (Person A): topic drill-down, "what could be next", signal decay, and test results.
// Data comes from radar_data.py (/api/topics, /api/topic, /api/forecasts, /api/signals),
// written ahead of time by `python -m src.topic_radar.app_export`.
// Hooks into the map without changing it: clicking a .net-node opens that topic below.
(() => {
  const DEFAULT_TOPIC = "Deep learning and LLM forecasting";
  const HIDE_LABEL = /\b(dan|yang|pada|terhadap|dalam|untuk)\b/i;  // a non-English title cluster
  const OTHER = -1;
  const pct = (v, d = 0) => v == null ? "–" : `${(v * 100).toFixed(d)}%`;
  const num = (v, d = 2) => v == null ? "–" : (+v).toFixed(d);
  const esc = s => String(s ?? "").replace(/[&<>"]/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
  const getJSON = url => fetch(url).then(r => r.json());
  const showError = (el, res) => { el.innerHTML = `<div class="sub">${esc(res.error)}</div>`; };

  let topicNames = [];
  let topicSize = new Map();
  let forecastByTopic = new Map();
  // forecast set sizes, from forecasts.json: every "rank N of M" and "top K" on the page
  // uses these, so the denominators always agree with the forecast itself
  let fcCount = { n: 25, nMap: 8, top: 5, subN: 75, subTop: 15 };

  // ---------------------------------------------------------------- layout
  // "Inside a topic" sits in a dock beside the map, shown only while a topic is open
  // (body.topic-open narrows the map, and the map redraws for its new width).
  // ?full is an unlinked internal view that also shows the extra panels.
  document.body.classList.toggle("full-view", new URLSearchParams(location.search).has("full"));
  const dock = document.getElementById("dock");
  dock.append(document.getElementById("zoom-panel"));

  // ---------------------------------------------------------------- topic drill-down
  // Opens inside the map box: the network zooms into the clicked circle and fades, the
  // topic's papers fly out of that spot into the sub-topic map; "All topics" reverses it.
  const box = document.getElementById("network");
  const zoom = {
    data: null, selected: null, pinned: null, open: false, origin: null, W: 0, H: 0, flows: [],
    side: document.getElementById("zoom-side"),
    select: document.getElementById("zoom-select"),
  };
  const PAD = 24, BAR = 34, FLY_MS = 750, BACK_MS = 450, MOVE_MS = 350;
  // links to other topics sit in a band under the buttons, so they never cover the islands
  const PORT_ROWS = 4, PORT_ROW = 14, PORT_BAND = 30 + PORT_ROWS * PORT_ROW, TOP = BAR + PORT_BAND;
  const WINDOW = 5;  // islands are sized by their papers in the last five years
  const overlay = d3.select(box).append("div").attr("class", "topic-map");
  const bar = overlay.append("div").attr("class", "topic-bar");
  bar.append("button").attr("type", "button").attr("class", "path-btn back-btn").text("← All topics")
    .on("click", () => closeTopic());
  // citation arrows stay hidden unless asked for (or an island is clicked)
  const flowsBtn = bar.append("button").attr("type", "button").attr("class", "path-btn flows-btn")
    .text("Show citation flows")
    .on("click", () => {
      zoom.showFlows = !zoom.showFlows;
      flowsBtn.classed("active", zoom.showFlows).text(zoom.showFlows ? "Hide citation flows" : "Show citation flows");
      drawTopic();
    });
  const barTitle = bar.append("span").attr("class", "topic-name");
  const barWindow = bar.append("span").attr("class", "sub");
  const svg = overlay.append("svg");
  const gHull = svg.append("g"), gFlow = svg.append("g"), gPort = svg.append("g"), gParticle = svg.append("g");
  const gLink = svg.append("g"), gTrail = svg.append("g"), gDot = svg.append("g"), gLabel = svg.append("g");
  const tip = overlay.append("div").attr("class", "radar-tip");

  function colourFor(subs) {
    const palette = d3.schemeTableau10.concat(["#8c6d31", "#5254a3", "#6b6ecf", "#9c9ede"]);
    const map = new Map();
    subs.filter(s => s.id !== OTHER).forEach((s, i) => map.set(s.id, palette[i % palette.length]));
    map.set(OTHER, "#b9ad9f");
    return map;
  }

  // dashed, rounded outline around a sub-topic: pad each dot, take the hull, smooth it
  function hullPath(points, pad = 9) {
    if (points.length < 3) return null;
    const cx = d3.median(points, p => p[0]), cy = d3.median(points, p => p[1]);
    const dist = points.map(p => Math.hypot(p[0] - cx, p[1] - cy));
    const cut = d3.quantile([...dist].sort(d3.ascending), 0.85);  // ignore far-flung strays
    const core = points.filter((p, i) => dist[i] <= cut);
    const padded = [];
    core.forEach(([x, y]) => { for (let k = 0; k < 8; k++) padded.push([x + pad * Math.cos(k * Math.PI / 4), y + pad * Math.sin(k * Math.PI / 4)]); });
    const hull = d3.polygonHull(padded);
    return hull ? d3.line().curve(d3.curveCatmullRomClosed.alpha(0.5))(hull) : null;
  }

  // best-fit outline: a smooth contour around where the papers are dense (it can bend
  // inwards, unlike a hull); small sub-topics fall back to the rounded hull. Only the
  // outer edge of each piece is kept (no holes) and slivers under a quarter of the
  // biggest piece are dropped, so a sub-topic reads as one clean blob.
  function outlinePath(points) {
    if (points.length < 8) return hullPath(points);
    const density = d3.contourDensity().x(p => p[0]).y(p => p[1])
      .size([zoom.W, zoom.H]).cellSize(4).bandwidth(12);
    const contours = density.contours(points);
    const rings = contours(contours.max * 0.05).coordinates
      .map(poly => poly[0]).map(r => ({ r, a: Math.abs(d3.polygonArea(r)) }));
    const biggest = d3.max(rings, x => x.a) || 0;
    const keep = rings.filter(x => x.a >= biggest * 0.25);
    if (!keep.length) return hullPath(points);
    return keep.map(x => d3.line().curve(d3.curveBasisClosed)(x.r.slice(0, -1))).join("");
  }

  // Positions (x, v) fitted into the box. The papers' own extent is fitted (not the 0..1
  // square), so the islands use the whole width instead of bunching in the middle. Each
  // axis may stretch up to STRETCH times the other: the layout's distances are only
  // relative, and a wide box would otherwise leave its sides empty. SIDE keeps room for
  // labels at the edges; LABEL_ROOM for labels under the lowest islands.
  const SIDE = 80, LABEL_ROOM = 22, STRETCH = 1.6;
  function scales(t) {
    if (!t.extent) t.extent = { x: d3.extent(t.papers, p => p.x), v: d3.extent(t.papers, p => p.v) };
    const [x0, x1] = t.extent.x, [v0, v1] = t.extent.v;
    const dx = (x1 - x0) || 1, dv = (v1 - v0) || 1;
    const aw = Math.max(50, zoom.W - 2 * SIDE), ah = Math.max(50, zoom.H - TOP - PAD - LABEL_ROOM);
    let kx = aw / dx, ky = ah / dv;
    kx = Math.min(kx, ky * STRETCH);
    ky = Math.min(ky, kx * STRETCH);
    const ox = (zoom.W - kx * dx) / 2 - kx * x0;
    const oy = TOP + 8 + (ah - ky * dv) / 2 - ky * v0;
    return { x: p => ox + kx * p.x, y: p => oy + ky * p.v };
  }

  // per-topic lookups, built once when a topic opens
  function prepare(t) {
    t.papers.forEach((p, i) => { p._i = i; });
    t.links = t.links || []; t.cites = t.cites || []; t.external = t.external || [];
    t.adj = t.papers.map(() => []);
    t.links.forEach(([i, j]) => { t.adj[i].push(j); t.adj[j].push(i); });
    t.citesOut = t.papers.map(() => []);   // papers this one cites
    t.citedBy = t.papers.map(() => []);    // papers that cite this one
    t.cites.forEach(([a, b]) => { t.citesOut[a].push(b); t.citedBy[b].push(a); });
    t.ext = t.papers.map(() => []);
    t.external.forEach(([i, topic, dir, n]) => t.ext[i].push({ topic, dir, n }));
    t.total = d3.rollup(t.papers, v => v.length, p => p.s);
    t.byYear = d3.rollup(t.papers, v => v.length, p => p.s, p => p.y);
    [t.y0, t.y1] = d3.extent(t.papers, p => p.y);
    // island scale: area follows papers in the window; the busiest island-year fills its slot
    let peak = 0;
    for (const [s, n] of t.total) for (let y = t.y0; y <= t.y1; y++) peak = Math.max(peak, windowCount(t, s, y) / n);
    t.scaleK = 1 / Math.sqrt(peak || 1);
  }

  function windowCount(t, s, year) {
    const m = t.byYear.get(s);
    let n = 0;
    if (m) for (let y = year - WINDOW + 1; y <= year; y++) n += m.get(y) || 0;
    return n;
  }

  function windowTotal(t, year) {
    return t.papers.filter(p => p.y > year - WINDOW && p.y <= year).length;
  }

  // where each island sits and how big it is right now
  function islandGeometry(t, x, y, year) {
    const geo = new Map();
    const all = windowTotal(t, year), before = windowTotal(t, year - WINDOW);
    for (const [s, ps] of d3.group(t.papers, p => p.s)) {
      const cx = d3.mean(ps, p => x(p)), cy = d3.mean(ps, p => y(p));
      // 80th percentile, not the max: one stray paper shouldn't push arrows far off the island
      const reach = d3.quantile(ps.map(p => Math.hypot(x(p) - cx, y(p) - cy)).sort(d3.ascending), 0.8) || 4;
      const n = windowCount(t, s, year), prev = windowCount(t, s, year - WINDOW);
      const k = Math.min(1, t.scaleK * Math.sqrt(n / t.total.get(s)));
      const share = n / (all || 1), shareBefore = prev / (before || 1);
      geo.set(s, { cx, cy, reach, k, n, prev, emerging: s !== OTHER && n >= 8 && share >= 1.5 * shareBefore });
    }
    return geo;
  }

  // citation flows between sub-topics: papers in the window citing earlier papers elsewhere
  function flows(t, year, lo) {
    const out = d3.rollup(
      t.cites.filter(([a, b]) => {
        const pa = t.papers[a], pb = t.papers[b];
        return pa.y >= lo && pa.y <= year && pb.y <= year && pa.s !== pb.s && pa.s !== OTHER && pb.s !== OTHER;
      }),
      v => v.length, ([a, b]) => `${t.papers[b].s}>${t.papers[a].s}`);
    return [...out].map(([key, n]) => { const [from, to] = key.split(">").map(Number); return { key, from, to, n }; });
  }

  function arrowGeometry(f, geo) {
    const A = geo.get(f.from), B = geo.get(f.to);
    const dx = B.cx - A.cx, dy = B.cy - A.cy, d = Math.hypot(dx, dy) || 1;
    const ra = A.reach * Math.max(A.k, 0.3) + 6, rb = B.reach * Math.max(B.k, 0.3) + 10;
    const x1 = A.cx + dx / d * ra, y1 = A.cy + dy / d * ra, x2 = B.cx - dx / d * rb, y2 = B.cy - dy / d * rb;
    const mx = (x1 + x2) / 2 - dy / d * d * 0.18, my = (y1 + y2) / 2 + dx / d * d * 0.18;
    return { x1, y1, x2, y2, mx, my };
  }

  function drawFlows(t, geo, year, lo, colour, fly) {
    const sel0 = zoom.selected;
    // one arrow per pair of sub-topics, pointing the way most citations go (A->B and
    // B->A drawn separately doubled the lines and crossed each other)
    const pairs = new Map();
    flows(t, year, lo).filter(f => geo.has(f.from) && geo.has(f.to)).forEach(f => {
      const k = f.from < f.to ? `${f.from}~${f.to}` : `${f.to}~${f.from}`;
      const p = pairs.get(k) || { key: k, total: 0, dirs: [] };
      p.total += f.n; p.dirs.push(f);
      pairs.set(k, p);
    });
    const fs = [...pairs.values()].map(p => {
      const main = p.dirs.reduce((a, b) => b.n > a.n ? b : a);
      const back = p.dirs.find(d => d !== main);
      return { key: p.key, from: main.from, to: main.to, n: p.total, fwd: main.n, back: back ? back.n : 0 };
    })
      .filter(f => sel0 == null || f.from === sel0 || f.to === sel0)
      .sort((a, b) => b.n - a.n).slice(0, sel0 != null ? 8 : 5);
    zoom.flows = fs;
    const w = d3.scaleSqrt().domain([1, Math.max(4, d3.max(fs, f => f.n) || 1)]).range([1.2, 6]);
    const sel = zoom.selected;
    const on = f => sel == null || f.from === sel || f.to === sel;
    const g = gFlow.selectAll("g.flow").data(fs, f => f.key).join(enter => {
      const e = enter.append("g").attr("class", "flow");
      e.append("path").attr("class", "flow-line");
      e.append("path").attr("class", "flow-head");
      e.append("title");
      return e;
    });
    g.attr("opacity", f => on(f) ? (fly ? 0 : 1) : 0.08)
      .call(s => fly && s.transition().delay(FLY_MS).duration(400).attr("opacity", f => on(f) ? 1 : 0.08));
    g.each(function (f) {
      const a = arrowGeometry(f, geo), c = colour.get(f.from), sw = w(f.n);
      f.path = `M${a.x1},${a.y1}Q${a.mx},${a.my} ${a.x2},${a.y2}`;
      const ux = a.x2 - a.mx, uy = a.y2 - a.my, ul = Math.hypot(ux, uy) || 1, hx = ux / ul, hy = uy / ul;
      const hl = 6 + sw, hw = 3 + sw * 0.7;
      const el = d3.select(this);
      el.select(".flow-line").attr("d", f.path).attr("stroke", c).attr("stroke-width", sw);
      el.select(".flow-head").attr("fill", c).attr("d",
        `M${a.x2 + hx * 2},${a.y2 + hy * 2}L${a.x2 - hx * hl - hy * hw},${a.y2 - hy * hl + hx * hw}`
        + `L${a.x2 - hx * hl + hy * hw},${a.y2 - hy * hl - hx * hw}Z`);
      el.select("title").text(`${label(t, f.from)} → ${label(t, f.to)}: ${f.fwd} citation${f.fwd === 1 ? "" : "s"} `
        + `from ${label(t, f.to)} papers to ${label(t, f.from)} papers, up to ${year}`
        + (f.back ? ` (and ${f.back} the other way)` : ""));
    });
    return fs.filter(on).map(f => ({ key: f.key, path: f.path, n: f.n, colour: colour.get(f.from) }));
  }

  // other topics this one builds on (left, arrows in) and that build on it (right, arrows out)
  function drawPorts(t, geo, year, fly, withArrows = true) {
    const agg = new Map();
    t.papers.forEach(p => {
      if (p.y > year) return;
      t.ext[p._i].forEach(e => {
        const k = `${e.dir}|${e.topic}`;
        if (!agg.has(k)) agg.set(k, { dir: e.dir, topic: e.topic, n: 0, by: new Map() });
        const a = agg.get(k);
        a.n += e.n;
        a.by.set(p.s, (a.by.get(p.s) || 0) + e.n);
      });
    });
    const sel = zoom.selected;
    const side = dir => [...agg.values()].filter(a => a.dir === dir && (sel == null || a.by.has(sel)))
      .sort((a, b) => b.n - a.n).slice(0, PORT_ROWS);
    const ports = [];
    [["out", 10, "start"], ["in", zoom.W - 10, "end"]].forEach(([dir, px, anchor]) => {
      side(dir).forEach((a, i, arr) => {
        const py = BAR + 30 + i * PORT_ROW;
        ports.push({ ...a, key: `${dir}|${a.topic}`, px, py, anchor });
      });
    });
    // small headings over the two lists in the band
    const heads = [["out", 10, "start", "Builds on"], ["in", zoom.W - 10, "end", "Built on by"]]
      .filter(([dir]) => ports.some(p => p.dir === dir));
    gPort.selectAll("text.port-head").data(heads, h => h[0]).join("text").attr("class", "port-head")
      .attr("x", h => h[1]).attr("y", BAR + 14).attr("text-anchor", h => h[2]).text(h => h[3]);
    const w = d3.scaleSqrt().domain([1, Math.max(4, d3.max(ports, p => d3.max([...p.by.values()])) || 1)]).range([1, 6]);
    const arrows = [];
    ports.forEach(p => {
      const into = p.dir === "out";  // this topic's papers cite that topic: knowledge flows in
      // default: one arrow to the island that uses it most; focused: the arrow to that island
      p.arrows = [...p.by].filter(([s]) => withArrows && geo.has(s) && geo.get(s).n > 0 && (sel == null || s === sel))
        .sort((a, b) => b[1] - a[1]).slice(0, 1).map(([s, n]) => {
        const g = geo.get(s), r = g.reach * Math.max(g.k, 0.3) + 6;
        const dx = g.cx - p.px, dy = g.cy - p.py, d = Math.hypot(dx, dy) || 1;
        const ix = g.cx - dx / d * r, iy = g.cy - dy / d * r;
        const sx = p.px + (into ? 40 : -40), sy = p.py + 7, mx = (sx + ix) / 2, my = (sy + iy) / 2;
        const path = into ? `M${sx},${sy}Q${mx},${my} ${ix},${iy}` : `M${ix},${iy}Q${mx},${my} ${sx},${sy}`;
        const a = { key: `${p.key}>${s}`, path, n, colour: into ? "#617891" : "#632024", s };
        arrows.push(a);
        return a;
      });
    });
    const g = gPort.selectAll("g.port").data(ports, p => p.key).join(enter => {
      const e = enter.append("g").attr("class", "port");
      e.append("g").attr("class", "port-arrows");
      e.append("text").attr("class", "port-label");
      return e;
    }).attr("opacity", fly ? 0 : 1)
      .call(s => fly && s.transition().delay(FLY_MS).duration(400).attr("opacity", 1));
    g.select(".port-label").attr("x", p => p.px).attr("y", p => p.py).attr("dy", "0.35em")
      .attr("text-anchor", p => p.anchor).attr("class", p => `port-label ${p.dir}`)
      .text(p => `${p.topic} · ${p.n}`)
      .on("click", (e, p) => { e.stopPropagation(); openTopic(p.topic, { x: p.px, y: p.py }); })
      .selectAll("title").data(p => [p]).join("title")
      .text(p => p.dir === "out" ? `Papers here cite ${p.topic} ${p.n} times (up to ${year}). Click to open it.`
        : `${p.topic} papers cite papers here ${p.n} times (up to ${year}). Click to open it.`);
    g.select(".port-arrows").selectAll("path").data(p => p.arrows, a => a.key).join("path")
      .attr("class", "port-arrow").attr("d", a => a.path).attr("stroke", a => a.colour).attr("stroke-width", a => w(a.n));
    return arrows;
  }

  // moving dots show which way knowledge flows
  function drawParticles(arrows) {
    const per = a => Math.min(a.n, zoom.selected != null ? 3 : 2);
    const parts = arrows.flatMap(a => d3.range(per(a)).map(k => ({ a, off: k / per(a) + Math.random() * 0.1, key: `${a.key}:${k}` })));
    gParticle.selectAll("circle").data(parts, p => p.key).join("circle")
      .attr("class", "particle").attr("r", 2.2).attr("fill", p => p.a.colour);
    startParticles();
  }

  let particleTimer = null;
  function startParticles() {
    if (particleTimer) return;
    const probe = document.createElementNS("http://www.w3.org/2000/svg", "path");
    particleTimer = d3.timer(elapsed => {
      gParticle.selectAll("circle").each(function (p) {
        probe.setAttribute("d", p.a.path);
        const len = probe.getTotalLength();
        const pt = probe.getPointAtLength(((elapsed / 2200 + p.off) % 1) * len);
        this.setAttribute("cx", pt.x);
        this.setAttribute("cy", pt.y);
      });
    });
  }
  function stopParticles() { if (particleTimer) { particleTimer.stop(); particleTimer = null; } }

  // Island labels: try above the cluster, then below, then across its middle, and take
  // the first spot that doesn't hit a label already placed (biggest clusters first).
  const measure = document.createElement("canvas").getContext("2d");
  function placeLabels(gs) {
    measure.font = `600 11px ${getComputedStyle(document.body).fontFamily}`;
    const placed = [];
    const hits = b => placed.reduce((n, o) => n + Math.max(0, Math.min(b.x1, o.x1) - Math.max(b.x0, o.x0))
      * Math.max(0, Math.min(b.y1, o.y1) - Math.max(b.y0, o.y0)), 0);
    gs.slice().sort((a, b) => b.n - a.n).forEach(g => {
      g.text = (g.sub && g.sub.predicted_boom ? "▲ " : "") + g.name + ` · ${g.n}` + (g.g && g.g.emerging ? "  ↑ emerging" : "");
      const w = measure.measureText(g.text).width + 6, h = 13;
      const ys = g.pts.map(p => p[1]);
      const cx = g.other ? d3.median(g.pts, p => p[0]) : g.g.cx;
      const x = Math.max(w / 2 + 4, Math.min(zoom.W - w / 2 - 4, cx));
      const tries = [d3.min(ys) - 10, d3.max(ys) + 16, d3.median(ys) + 4]
        .map(y => Math.max(TOP + 10, Math.min(zoom.H - 6, y)))
        .map(y => ({ y, b: { x0: x - w / 2, x1: x + w / 2, y0: y - h + 2, y1: y + 3 } }));
      let best = tries.find(c => hits(c.b) === 0);
      if (!best) {  // all taken: step down from the top spot until it is clear
        best = tries.reduce((a, c) => hits(c.b) < hits(a.b) ? c : a);
        for (let k = 1; k < 8 && hits(best.b) > 0; k++) {
          const y = Math.min(zoom.H - 6, tries[0].y + k * h);
          best = { y, b: { ...best.b, y0: y - h + 2, y1: y + 3 } };
        }
      }
      placed.push(best.b);
      g.lx = x; g.ly = best.y;
    });
  }

  // readable sub-topic name (data/subtopic_names.csv via radar_data.py); the keyword
  // list it was named from stays available as `label` for tooltips
  const label = (t, s) => {
    const sub = t.subtopics.find(u => u.id === s);
    return sub ? (sub.name || sub.label.split(", ").slice(0, 2).join(", ")) : "other";
  };

  // hover: who this paper cites (solid, to older papers) and who cites it (dashed);
  // shared-reference neighbours as faint dots
  function paperLines(t, p, shown, pos) {
    const cites = t.citesOut[p._i].filter(j => shown.has(j)).map(j => ({ q: t.papers[j], kind: "cites" }));
    const citedBy = t.citedBy[p._i].filter(j => shown.has(j)).map(j => ({ q: t.papers[j], kind: "cited-by" }));
    const shared = t.adj[p._i].filter(j => shown.has(j)).map(j => ({ q: t.papers[j], kind: "shared" }));
    const [px, py] = pos(p);
    gLink.selectAll("line").data(cites.concat(citedBy, shared), d => d.kind + d.q.id).join("line")
      .attr("class", d => `plink ${d.kind}`)
      .attr("x1", px).attr("y1", py).attr("x2", d => pos(d.q)[0]).attr("y2", d => pos(d.q)[1]);
    return { cites: cites.length, citedBy: citedBy.length, shared: shared.length };
  }

  // click a paper: its citation trail two steps back in time, plus where it leaves the topic
  function drawTrail(t, shown, pos, x, y) {
    gTrail.selectAll("*").remove();
    const root = zoom.pinned;
    if (root == null) return null;
    const hop = new Map([[root, 0]]), edges = [];
    let frontier = [root];
    for (let h = 1; h <= 2; h++) {
      const next = [];
      for (const i of frontier) for (const j of t.citesOut[i]) {
        if (!shown.has(j)) continue;
        edges.push({ a: i, b: j, h });
        if (!hop.has(j)) { hop.set(j, h); next.push(j); }
      }
      frontier = next;
    }
    gTrail.selectAll("line").data(edges).join("line").attr("class", "trail-line")
      .attr("opacity", e => e.h === 1 ? 0.9 : 0.45)
      .attr("x1", e => pos(t.papers[e.a])[0]).attr("y1", e => pos(t.papers[e.a])[1])
      .attr("x2", e => pos(t.papers[e.b])[0]).attr("y2", e => pos(t.papers[e.b])[1]);
    // exits: citations to and from other topics, stacked on the right edge
    const exits = d3.rollups(t.ext[root], v => d3.sum(v, e => e.n), e => e.topic, e => e.dir)
      .flatMap(([topic, dirs]) => dirs.map(([dir, n]) => ({ topic, dir, n })))
      .sort((a, b) => b.n - a.n).slice(0, 6);
    const [rx, ry] = pos(t.papers[root]);
    const ex = zoom.W - 8;
    exits.forEach((e, i) => { e.yy = BAR + 30 + i * 24; e.text = `${e.dir === "out" ? "cites" : "cited by"} ${e.topic} (${e.n}) →`; });
    const g = gTrail.selectAll("g.exit").data(exits).join("g").attr("class", "exit")
      .attr("transform", e => `translate(${ex},${e.yy})`)
      .on("click", (ev, e) => { ev.stopPropagation(); openTopic(e.topic, { x: ex - 40, y: e.yy }); });
    g.append("text").attr("text-anchor", "end").attr("dy", "0.35em")
      .text(e => e.text)
      .each(function (e) { e.left = ex - this.getComputedTextLength() - 6; });
    gTrail.selectAll("path.exit-line").data(exits).join("path").attr("class", "exit-line")
      .attr("d", e => `M${rx},${ry}C${(rx + e.left) / 2},${ry} ${(rx + e.left) / 2},${e.yy} ${e.left},${e.yy}`);
    return { trail: hop, exits: exits.length };
  }

  // fly: true when opening, so dots start at the clicked circle and spread out
  function drawTopic(fly = false) {
    const t = zoom.data;
    if (!t || !zoom.open) return;
    const year = Math.min(yearState.year || t.y1, t.y1 + 1);
    const lo = year - WINDOW + 1;
    const { x, y } = scales(t);
    const geo = islandGeometry(t, x, y, year);
    const pos = p => {
      const g = geo.get(p.s), k = p.y < lo ? Math.max(g.k, 0.3) : g.k;
      return [g.cx + (x(p) - g.cx) * Math.max(k, 0.05), g.cy + (y(p) - g.cy) * Math.max(k, 0.05)];
    };
    const colour = colourFor(t.subtopics);
    const subById = new Map(t.subtopics.map(s => [s.id, s]));
    const recent = t.papers.filter(p => p.y >= lo && p.y <= year);
    const shown = new Set(t.papers.filter(p => p.y <= year).map(p => p._i));
    const sel = zoom.selected;
    const o = zoom.origin;

    barWindow.text(`papers from ${lo}–${year}`);

    // outlines from the papers in the window, at their current (scaled) positions
    // Every cluster is named, including the grey "other" papers and the non-English
    // cluster (which used to be left blank); only real sub-topics get an outline.
    const named = d3.groups(recent, p => p.s)
      .map(([id, ps]) => ({ id, n: ps.length, sub: subById.get(id), g: geo.get(id), pts: ps.map(pos), other: id === OTHER }))
      .filter(g => g.other ? g.n >= 5 : g.sub)
      .map(g => ({ ...g, d: g.other ? null : outlinePath(g.pts),
        name: g.other ? "other small sub-topics" : HIDE_LABEL.test(g.sub.label) ? "non-English titles" : label(t, g.id) }));
    const groups = named.filter(g => g.d);
    placeLabels(named);
    const hulls = gHull.selectAll("path").data(groups, g => g.id).join(enter => enter.append("path").attr("opacity", 0))
      .attr("class", g => "hull" + (sel != null && sel !== g.id ? " dim" : "") + (g.g.emerging ? " emerging" : ""))
      .attr("d", g => g.d).attr("stroke", g => colour.get(g.id)).attr("fill", g => colour.get(g.id))
      .on("click", (e, g) => { e.stopPropagation(); zoom.selected = zoom.selected === g.id ? null : g.id; drawTopic(); });
    const labels = gLabel.selectAll("text").data(named, g => g.id).join(enter => enter.append("text").attr("opacity", 0))
      .attr("class", g => "hull-label" + (g.other ? " other" : ""))
      .attr("text-anchor", "middle")
      .attr("x", g => g.lx).attr("y", g => g.ly)
      .attr("fill", g => g.other ? "#8f8579" : colour.get(g.id))
      .text(g => g.text);
    const labelOpacity = g => sel != null && sel !== g.id ? 0.2 : 1;
    if (fly) {
      hulls.attr("opacity", 0).transition().delay(FLY_MS * 0.7).duration(400).attr("opacity", 1);
      labels.attr("opacity", 0).transition().delay(FLY_MS * 0.8).duration(400).attr("opacity", labelOpacity);
    } else {
      hulls.attr("opacity", 1);
      labels.attr("opacity", labelOpacity);
    }

    const flowsOn = zoom.showFlows || zoom.selected != null;
    let arrows = [];
    if (zoom.pinned != null) { gFlow.selectAll("*").remove(); gPort.selectAll("*").remove(); }
    else {
      if (flowsOn) arrows = drawFlows(t, geo, year, t.y0, colour, fly);
      else gFlow.selectAll("*").remove();
      // other topics are listed in the band at the top (filtered to the focused island);
      // lines from there to the islands only bundled up and crossed the map, so none
      drawPorts(t, geo, year, fly, false);
      if (flowsOn && !zoom.flows.length) barWindow.text(`papers from ${lo}–${year} · no citations between these sub-topics yet`);
    }
    drawParticles(zoom.selected != null ? arrows : []);  // moving dots only when focused
    gLink.selectAll("line").remove();
    const trail = drawTrail(t, shown, pos, x, y);

    const cls = p => "paper-dot" + (p.y > year ? " later" : p.y < lo ? " past" : "") + (p.seed != null ? " seed" : "")
      + (trail ? (trail.trail.has(p._i) ? " trail" : " dim") : sel != null && sel !== p.s ? " dim" : "")
      + (zoom.pinned === p._i ? " pinned" : "");
    const dots = gDot.selectAll("circle").data(t.papers, p => p.id).join("circle")
      .attr("class", cls)
      .attr("fill", p => colour.get(p.s) || colour.get(OTHER))
      .on("mouseenter", (e, p) => {
        const s = subById.get(p.s);
        const n = paperLines(t, p, shown, pos);
        tip.html(`<b>${esc(p.t)}</b><br>${p.y} · ${esc(s ? s.label : "other")}`
          + `<br>cites ${n.cites} · cited by ${n.citedBy} here (direct) · ${n.shared} shared-reference links`
          + (t.ext[p._i].length ? `<br>${d3.sum(t.ext[p._i], e => e.n)} citations to/from other topics` : "")
          + (p.seed != null ? `<br>Suggested reading #${p.seed}` : "") + `<br><span class="tip-hint">click for its citation trail</span>`)
          .style("opacity", 1);
      })
      .on("mousemove", e => {
        const [mx, my] = d3.pointer(e, overlay.node());
        tip.style("left", `${Math.max(4, Math.min(mx + 12, zoom.W - 290))}px`).style("top", `${my + 12}px`);
      })
      .on("mouseleave", () => { tip.style("opacity", 0); gLink.selectAll("line").remove(); })
      .on("click", (e, p) => { e.stopPropagation(); zoom.pinned = zoom.pinned === p._i ? null : p._i; drawTopic(); });
    const r = p => p.y < lo ? 1.8 : p.seed != null ? 4.5 : 3;
    if (fly) {
      dots.interrupt().attr("cx", o.x).attr("cy", o.y).attr("r", 0)
        .transition().delay(() => Math.random() * 250).duration(FLY_MS).ease(d3.easeCubicOut)
        .attr("cx", p => pos(p)[0]).attr("cy", p => pos(p)[1]).attr("r", r);
    } else {
      dots.interrupt().transition().duration(MOVE_MS)
        .attr("cx", p => pos(p)[0]).attr("cy", p => pos(p)[1]).attr("r", r);
    }

    drawSide(t, colour, geo, year, lo);
  }

  // small timeline for the focused sub-topic: its papers per year, and citations coming in
  function timeline(el, t, s, colour) {
    const years = d3.range(t.y0, t.y1 + 1);
    const papers = years.map(y => ({ y, n: (t.byYear.get(s) || new Map()).get(y) || 0 }));
    const inflow = d3.rollup(t.cites.filter(([a, b]) => t.papers[a].s === s && t.papers[b].s !== s && t.papers[b].s !== OTHER),
      v => v.length, ([a]) => t.papers[a].y, ([, b]) => t.papers[b].s);
    const sources = [...new Set(t.cites.filter(([a, b]) => t.papers[a].s === s && t.papers[b].s !== s && t.papers[b].s !== OTHER)
      .map(([, b]) => t.papers[b].s))];
    const w = 520, h = 150, m = { t: 8, r: 8, b: 20, l: 28 }, mid = 92;
    const xb = d3.scaleBand().domain(years).range([m.l, w - m.r]).padding(0.15);
    const yp = d3.scaleLinear().domain([0, d3.max(papers, d => d.n) || 1]).nice().range([mid, m.t]);
    const stack = years.map(y => { const r = { y }; sources.forEach(k => { r[k] = (inflow.get(y) || new Map()).get(k) || 0; }); return r; });
    const series = d3.stack().keys(sources)(stack);
    const yi = d3.scaleLinear().domain([0, Math.max(3, d3.max(series.flat(), d => d[1]) || 1)]).range([mid + 6, h - m.b]);
    const sv = d3.select(el).append("svg").attr("viewBox", `0 0 ${w} ${h}`);
    sv.append("g").selectAll("rect").data(papers).join("rect").attr("fill", colour.get(s))
      .attr("x", d => xb(d.y)).attr("width", xb.bandwidth()).attr("y", d => yp(d.n)).attr("height", d => mid - yp(d.n))
      .append("title").text(d => `${d.y}: ${d.n} papers`);
    sv.append("g").selectAll("g").data(series).join("g").attr("fill", d => colour.get(+d.key)).attr("opacity", 0.8)
      .selectAll("rect").data(d => d.map(v => ({ v, key: d.key }))).join("rect")
      .attr("x", d => xb(d.v.data.y)).attr("width", xb.bandwidth())
      .attr("y", d => yi(d.v[0])).attr("height", d => yi(d.v[1]) - yi(d.v[0]))
      .append("title").text(d => `${d.v.data.y}: ${d.v[1] - d.v[0]} citations to ${label(t, +d.key)}`);
    sv.append("line").attr("stroke", "var(--line)").attr("x1", m.l).attr("x2", w - m.r).attr("y1", mid + 3).attr("y2", mid + 3);
    sv.append("g").attr("class", "axis").attr("transform", `translate(0,${h - m.b})`)
      .call(d3.axisBottom(xb).tickValues(years.filter(y => y % 5 === 0)));
    sv.append("text").attr("class", "tl-note").attr("x", m.l).attr("y", m.t + 8).text("papers per year");
    sv.append("text").attr("class", "tl-note").attr("x", m.l).attr("y", h - m.b - 4).text("citations to other sub-topics (by colour)");
  }

  function drawSide(t, colour, geo, year, lo) {
    const subs = t.subtopics.filter(s => !HIDE_LABEL.test(s.label));
    const sel = zoom.selected;
    // ranked among the map's topics, like the heading table; a topic opened from the
    // dropdown or a "builds on" link may be off the map, so it says "of all" instead
    const f = forecastByTopic.get(t.name);
    const grow = t.predicted_boom ? " · <b>predicted to grow</b>" : "";
    const rank = f && f.map_rank ? `Forecast rank ${f.map_rank} of ${fcCount.nMap} on the map${grow}`
      : t.rank ? `Forecast rank ${t.rank} of all ${fcCount.n} (not on the map)${grow}` : "";
    const seeds = t.papers.filter(p => p.seed != null && (sel == null || p.s === sel))
      .sort((a, b) => a.seed - b.seed).slice(0, 8);
    const all = flows(t, t.y1, t.y0);  // every year, for the focus lists
    const fedBy = sel == null ? [] : all.filter(f => f.to === sel).sort((a, b) => b.n - a.n);
    const fed = sel == null ? [] : all.filter(f => f.from === sel).sort((a, b) => b.n - a.n);
    const flowList = (fs, key) => fs.length ? fs.map(f => `<li><span class="swatch-inline" style="background:${colour.get(f[key])}"></span>${esc(label(t, f[key]))} <span class="sub">${f.n}</span></li>`).join("") : `<li class="sub">none in the sample</li>`;
    zoom.side.innerHTML = `
      <div class="sub">${t.n} papers in the sample · showing ${lo}–${year} · stage: ${esc(t.stage || "–")} · ${rank}</div>
      <div class="zoom-cols">
        <div>
          <h3>Sub-topics</h3>
          <div class="sub-row head"><span></span><span class="sub">click to focus</span>
            <span class="counts">${lo}–${String(year).slice(2)}</span>
            <span class="counts">${String(t.prior[0]).slice(2)}–${String(t.prior[1]).slice(2)} → ${String(t.recent[0]).slice(2)}–${String(t.recent[1]).slice(2)}</span></div>
          ${subs.map(s => { const g = geo.get(s.id) || {}; return `
            <div class="sub-row${sel === s.id ? " on" : ""}" data-id="${s.id}" title="Keywords: ${esc(s.label)}">
              <span class="swatch" style="border-color:${colour.get(s.id)}"></span>
              <span>${esc(s.name || s.label)}${s.rank && s.rank <= fcCount.subTop ? ` <span class="rank">#${s.rank} of ${fcCount.subN}</span>` : ""}${g.emerging ? ` <span class="emerging-tag">↑ emerging</span>` : ""}</span>
              <span class="counts">${g.n ?? 0}</span>
              <span class="counts">${s.prior} → ${s.recent}</span>
            </div>`; }).join("")}
          <div class="sub key-line"><span title="Predicted top ${fcCount.subTop} of ${fcCount.subN} sub-topics. Exploratory: the forecast did not pass its test.">▲ / #rank = forecast</span> · <span title="Its share of this topic's papers is at least 1.5× what it was in the five years before.">↑ = emerging</span></div>
        </div>
        <div>
          ${sel != null ? `
          <h3>${esc(label(t, sel))}: who fed whom</h3>
          <div class="flow-cols"><div><div class="sub">built on (it cites them)</div><ul class="flow-list">${flowList(fedBy, "from")}</ul></div>
          <div><div class="sub">fed (they cite it)</div><ul class="flow-list">${flowList(fed, "to")}</ul></div></div>
          <div class="tl" id="zoom-timeline"></div>` : ""}
          <h3>Suggested reading${sel != null ? " in this sub-topic" : ""}</h3>
          ${seeds.length ? `<ol class="seed-list">${seeds.map(p => `<li>${esc(p.t)} <span class="sub">(${p.y})</span></li>`).join("")}</ol>`
            : `<div class="sub">No suggested papers here.</div>`}
          <div class="sub key-line" title="Arrows inside the map are direct citations between sub-topics, from the cited to the citing side. The band at the top lists the topics this one builds on (left) and that build on it (right). They show who builds on whom, not cause and effect.">How to read the arrows ⓘ</div>
        </div>
      </div>`;
    if (sel != null) timeline(document.getElementById("zoom-timeline"), t, sel, colour);
    zoom.side.querySelectorAll(".sub-row[data-id]").forEach(row => row.addEventListener("click", () => {
      const id = +row.dataset.id;
      zoom.selected = zoom.selected === id ? null : id;
      drawTopic();
    }));
  }

  // origin: {x, y} in map-box pixels where the papers fly out from (the clicked circle)
  async function openTopic(name, origin) {
    const res = await getJSON(`/api/topic?name=${encodeURIComponent(name)}`);
    if (res.error) return showError(zoom.side, res);
    document.body.classList.add("topic-open");
    dock.classList.add("topic-open");
    zoom.W = box.clientWidth;
    zoom.H = box.clientHeight;
    zoom.origin = origin || { x: zoom.W / 2, y: zoom.H / 2 };
    prepare(res);
    zoom.data = res;
    zoom.selected = null;
    zoom.pinned = null;
    zoom.open = true;
    hover.classed("on", false);
    hideHint();
    zoom.select.value = name;
    document.getElementById("zoom-title").textContent = name;
    barTitle.text(name);
    svg.attr("viewBox", [0, 0, zoom.W, zoom.H]);
    box.style.setProperty("--zx", `${zoom.origin.x}px`);
    box.style.setProperty("--zy", `${zoom.origin.y}px`);
    box.classList.add("zoomed");
    overlay.classed("open", true);
    drawTopic(true);
  }

  function closeTopic() {
    if (!zoom.open) return;
    const o = zoom.origin;
    zoom.open = false;
    dock.classList.remove("topic-open");
    document.body.classList.remove("topic-open");
    zoom.pinned = null;
    stopParticles();
    tip.style("opacity", 0);
    gHull.selectAll("path").transition().duration(200).attr("opacity", 0);
    gFlow.selectAll("g").transition().duration(200).attr("opacity", 0);
    gPort.selectAll("g").transition().duration(200).attr("opacity", 0);
    gParticle.selectAll("circle").remove();
    gTrail.selectAll("*").remove();
    gLink.selectAll("line").remove();
    gLabel.selectAll("text").transition().duration(200).attr("opacity", 0);
    box.classList.remove("zoomed");
    gDot.selectAll("circle").interrupt().transition().duration(BACK_MS).ease(d3.easeCubicIn)
      .attr("cx", o.x).attr("cy", o.y).attr("r", 0);
    // a timer, not the transition's end, so it closes even if animation frames are paused
    setTimeout(() => { if (!zoom.open) overlay.classed("open", false); }, BACK_MS + 50);
  }

  // where a map circle sits inside the box, in pixels
  function circleCentre(node) {
    const c = node.querySelector("circle").getBoundingClientRect(), b = box.getBoundingClientRect();
    return { x: c.left + c.width / 2 - b.left, y: c.top + c.height / 2 - b.top };
  }

  const scrollToMap = () => {
    const panel = box.closest(".panel"), top = panel.getBoundingClientRect().top;
    if (top < -40 || top > window.innerHeight / 3) panel.scrollIntoView({ behavior: "smooth", block: "start" });
  };

  // any element with data-topic (a row in "Where research is heading") opens that topic,
  // flying out of its map circle, the same as clicking the circle
  document.addEventListener("click", e => {
    const row = e.target.closest("[data-topic]");
    if (!row || row.closest(".topic-map") || e.target.closest("a")) return;
    const name = row.dataset.topic;
    if (!topicNames.includes(name)) return;
    scrollToMap();
    hideHint();
    const node = [...box.querySelectorAll(".net-node")].find(g => { const d = d3.select(g).datum(); return d && d.id === name; });
    openTopic(name, node ? circleCentre(node) : null);
  });
  zoom.select.addEventListener("change", () => { scrollToMap(); openTopic(zoom.select.value); });
  document.addEventListener("yearchange", () => drawTopic());
  let resizeTimer = null;
  new ResizeObserver(() => {
    clearTimeout(resizeTimer);
    resizeTimer = setTimeout(() => {
      if (!zoom.open || box.clientWidth < 80 || box.clientHeight < 80) return;
      if (Math.abs(box.clientWidth - zoom.W) < 2 && Math.abs(box.clientHeight - zoom.H) < 2) return;
      zoom.W = box.clientWidth; zoom.H = box.clientHeight;
      svg.attr("viewBox", [0, 0, zoom.W, zoom.H]);
      drawTopic();
    }, 150);
  }).observe(box);
  svg.on("click", () => {  // empty space: drop the pinned paper, then the focus
    if (zoom.pinned != null) zoom.pinned = null; else zoom.selected = null;
    drawTopic();
  });
  box.addEventListener("click", e => {
    const node = e.target.closest(".net-node");
    if (!node || zoom.open) return;
    const d = d3.select(node).datum();
    const name = d && (d.id || d.name);
    if (topicNames.includes(name)) { hideHint(); openTopic(name, circleCentre(node)); }
  });

  // ---------------------------------------------------------------- "you can click this"
  // A hint in the corner of the map until the first topic is opened, and a small label
  // under the mouse on each circle saying what a click does.
  const hint = d3.select(box).append("div").attr("class", "map-hint")
    .html(`<span class="map-hint-dot"></span>Click a circle to explore its papers`);
  function hideHint() {
    hint.classed("gone", true);
    try { localStorage.setItem("radar-hint-seen", "1"); } catch (_) { /* private mode */ }
  }
  try { if (localStorage.getItem("radar-hint-seen")) hint.classed("gone", true); } catch (_) { /* ignore */ }
  const hover = d3.select(box).append("div").attr("class", "node-cta");
  box.addEventListener("mouseover", e => {
    const node = e.target.closest(".net-node");
    if (!node || zoom.open) return;
    const d = d3.select(node).datum();
    const name = d && (d.id || d.name);
    if (!topicNames.includes(name)) return;
    const c = circleCentre(node);
    const n = topicSize.get(name);
    hover.html(`Explore ${n ? `${n} papers` : "papers"} <span aria-hidden="true">→</span>`);
    const w = hover.node().offsetWidth / 2 + 6;  // centred on the circle, kept inside the map
    hover.style("left", `${Math.max(w, Math.min(box.clientWidth - w, c.x))}px`).style("top", `${c.y}px`).classed("on", true);
  });
  box.addEventListener("mouseout", e => {
    const node = e.target.closest(".net-node");
    if (node && !node.contains(e.relatedTarget)) hover.classed("on", false);
  });
  document.addEventListener("keydown", e => {
    if (e.key !== "Escape" || !zoom.open) return;
    if (zoom.pinned != null) { zoom.pinned = null; drawTopic(); } else closeTopic();
  });

  // "next up" rings on the map circles for the predicted top 5 topics. The forecast is
  // made from one year's data (f.year, 2025) about the years after it, so the rings show
  // only from that year on; on earlier years of the slider they would claim too much.
  function syncRings() {
    const shown = typeof yearState === "undefined" ? null : yearState.year;
    d3.selectAll("#network .net-node").each(function (d) {
      const g = d3.select(this);
      const f = d && forecastByTopic.get(d.id);
      const circle = g.select("circle:not(.next-ring)");
      let ring = g.select("circle.next-ring");
      if (!f || !f.predicted_boom || circle.empty() || (shown != null && f.year && shown < f.year)) {
        ring.remove(); return;
      }
      if (ring.empty()) {
        ring = g.insert("circle", "text").attr("class", "next-ring");
        ring.append("title").text(`Predicted to grow over the next three years (forecast from ${f.year} data; exploratory)`);
      }
      const r = String(+circle.attr("r") + 4);
      if (ring.attr("r") !== r) ring.attr("r", r).attr("cx", circle.attr("cx")).attr("cy", circle.attr("cy"));
    });
  }
  new MutationObserver(recs => { if (recs.some(r => !overlay.node().contains(r.target))) syncRings(); }).observe(box,
    { childList: true, subtree: true, attributes: true, attributeFilter: ["r"] });
  document.addEventListener("yearchange", syncRings);

  // ---------------------------------------------------------------- what could be next
  function drawNext(fc) {
    const topics = fc.topics.slice(0, 10);
    document.getElementById("next-sub").textContent =
      "Ranked by a model trained on 2005–2022 to predict which topics grow their share of papers over the next three years. "
      + "In testing it ranked better than chance but did not clear every pre-registered bar, so treat this as a lead, not a call.";
    document.getElementById("next-topics").innerHTML = topics.map(t => `
      <li class="${t.predicted_boom ? "top" : ""}"><a href="#zoom-panel" data-topic="${esc(t.topic)}">${esc(t.topic)}</a>
      <span class="meta">${t.predicted_boom ? ` · predicted top ${fcCount.top}` : ""}</span></li>`).join("");
    const subs = fc.subtopics.filter(s => !HIDE_LABEL.test(s.label)).slice(0, 10);
    document.getElementById("next-subtopics").innerHTML = subs.map(s => `
      <li class="${s.predicted_boom ? "top" : ""}" title="Keywords: ${esc(s.label)}">${esc(s.name || s.label)}
      <span class="meta"> · in <a href="#zoom-panel" data-topic="${esc(s.parent_group)}">${esc(s.parent_group)}</a>
      · ${s.papers_2020_22} → ${s.papers_2023_25} papers</span></li>`).join("");
    document.querySelectorAll("#next-panel a[data-topic]").forEach(a => a.addEventListener("click", e => {
      e.preventDefault();
      scrollToMap();
      openTopic(a.dataset.topic);
    }));
  }

  // ---------------------------------------------------------------- decay curve and haircut
  function drawDecay(sig) {
    const rows = sig.curve.map(r => ({ ...r, mid: (r.years_from + r.years_to) / 2 }));
    const w = 640, h = 292, m = { t: 16, r: 16, b: 42, l: 48 };
    const x = d3.scaleLinear().domain(d3.extent(rows, r => r.mid)).nice().range([m.l, w - m.r]);
    const y = d3.scaleLinear().domain([0, Math.max(1.2, d3.max(rows, r => r.ci_high))]).nice().range([h - m.b, m.t]);
    const s = d3.select("#decay-plot").html("").append("svg").attr("viewBox", `0 0 ${w} ${h}`);
    const x0 = x(0), yTop = m.t, yBot = h - m.b;
    // the years after publication get a light background, so the two zones read at a glance
    s.append("rect").attr("class", "after-zone").attr("x", x0).attr("y", yTop)
      .attr("width", w - m.r - x0).attr("height", yBot - yTop);
    s.append("g").attr("class", "axis").attr("transform", `translate(0,${yBot})`)
      .call(d3.axisBottom(x).ticks(6).tickFormat(d => d === 0 ? "published" : d > 0 ? `+${d}` : d));
    s.append("g").attr("class", "axis").attr("transform", `translate(${m.l},0)`).call(d3.axisLeft(y).ticks(5).tickFormat(d3.format(".0%")));
    s.append("text").attr("class", "zone-label").attr("x", w - m.r).attr("y", h - 4).attr("text-anchor", "end").text("years since publication");
    s.append("text").attr("class", "zone-label").attr("x", x0 - 6).attr("y", yTop + 14).attr("text-anchor", "end").text("Paper's sample period");
    s.append("text").attr("class", "zone-label").attr("x", x0 + 6).attr("y", yTop + 14).text("After publication");
    s.append("line").attr("class", "ref-line").attr("x1", m.l).attr("x2", w - m.r).attr("y1", y(1)).attr("y2", y(1));
    s.append("text").attr("class", "zone-label").attr("x", w - m.r - 4).attr("y", y(1) - 6).attr("text-anchor", "end")
      .text("100% = the return the paper reported");
    s.append("path").attr("class", "ci-band").attr("d", d3.area().x(r => x(r.mid)).y0(r => y(r.ci_low)).y1(r => y(r.ci_high))(rows));
    s.append("line").attr("class", "pub-line").attr("x1", x0).attr("x2", x0).attr("y1", yTop).attr("y2", yBot);
    s.append("path").attr("class", "curve-line").attr("d", d3.line().x(r => x(r.mid)).y(r => y(r.share_of_in_sample_mean))(rows));
    s.selectAll(".curve-dot").data(rows).join("circle").attr("class", "curve-dot").attr("r", 3.5)
      .attr("cx", r => x(r.mid)).attr("cy", r => y(r.share_of_in_sample_mean))
      .append("title").text(r => `${r.years_from} to ${r.years_to} years: ${pct(r.share_of_in_sample_mean)} of in-sample (95%: ${pct(r.ci_low)}–${pct(r.ci_high)}), ${r.n_signals} signals`);

    // The finding, written on the chart: an arrow from 100% down to the first years after publication.
    const first = rows.find(r => r.years_from === 0);
    if (first) {
      const ax = x(first.mid) + (x(first.mid) - x0) * 0.9, ay = y(first.share_of_in_sample_mean);
      const note = s.append("g").attr("class", "finding");
      note.append("line").attr("x1", ax).attr("x2", ax).attr("y1", y(1) + 4).attr("y2", ay - 6);
      note.append("path").attr("d", `M${ax - 4},${ay - 12}L${ax},${ay - 5}L${ax + 4},${ay - 12}`);
      const tx = ax + 8, ty = (y(1) + ay) / 2 - 10;
      note.append("text").attr("class", "big").attr("x", tx).attr("y", ty).text(`Loses ${pct(1 - first.share_of_in_sample_mean)}`);
      note.append("text").attr("x", tx).attr("y", ty + 16).text(`keeps ${pct(first.share_of_in_sample_mean)} in the first ${first.years_to} years`);
      if (sig.haircut && sig.haircut.median != null)
        note.append("text").attr("x", tx).attr("y", ty + 31).text(`typical signal keeps ${pct(sig.haircut.median)}`);
    }
    // Context notes, only when the data shows them.
    const pre = rows.filter(r => r.years_to <= 0);
    if (pre.length >= 2 && pre[0].share_of_in_sample_mean - pre[pre.length - 1].share_of_in_sample_mean > 0.1) {
      // under the band, starting at the first point, on two short lines
      const p = pre[pre.length - 1], t = s.append("text").attr("class", "zone-label note")
        .attr("x", x(pre[0].mid)).attr("y", y(p.ci_low) + 28);
      t.append("tspan").text("Already fading");
      t.append("tspan").attr("x", x(pre[0].mid)).attr("dy", 15).text("before publication");
    }
    const last = rows[rows.length - 1];
    if (first && last !== first && last.n_signals < first.n_signals * 0.6) {
      s.append("text").attr("class", "zone-label note").attr("x", x(last.mid)).attr("y", Math.min(yBot - 6, y(last.ci_low) + 16))
        .attr("text-anchor", "end").text(`Wider band: only ${last.n_signals} signals are this old`);
    }

    drawHaircut(sig.haircut);
  }

  // ---------------------------------------------------------------- haircut calculator
  // Builds its own markup inside #haircut, so the page HTML only needs the empty box.
  function drawHaircut(hc) {
    const box = document.getElementById("haircut");
    if (!box || !hc) return;
    box.innerHTML = `
      <div class="hc-in">
        <label for="hc-input">The paper claims</label>
        <div class="hc-field"><input type="number" id="hc-input" value="1.0" step="0.1" min="0"><span>% a month</span></div>
        <div class="hc-err" role="alert"></div>
      </div>
      <div class="hc-out" aria-live="polite"></div>
      <div class="sub hc-foot">Based on ${hc.n_signals} published signals. ${pct(hc.share_below_zero)} lost money after
        publication; ${pct(hc.share_above_one)} did better than the paper said.</div>`;
    const input = box.querySelector("input"), out = box.querySelector(".hc-out"), err = box.querySelector(".hc-err");
    const f2 = v => v.toFixed(2);
    // One big number (the median), then the middle half of signals as a bar from 0 to the claim.
    const show = () => {
      const c = parseFloat(input.value);
      if (input.value.trim() === "" || isNaN(c) || c < 0) { err.textContent = "Enter a return of 0 or more"; return; }
      err.textContent = "";
      const med = c * hc.median, lo = c * hc.p25, hi = c * hc.p75;
      const top = Math.max(c, hi, 1e-9), at = v => Math.max(0, Math.min(100, v / top * 100));
      out.innerHTML = `<div class="hc-label">Expect after publication</div>
        <div class="hc-big"><b>${f2(med)}%</b> a month</div>
        <div class="sub" title="95% interval for the median">typical signal, 95% range ${f2(c * hc.median_ci_low)}–${f2(c * hc.median_ci_high)}%</div>
        <div class="hc-scale" role="img" aria-label="Middle half of signals keep ${f2(lo)}% to ${f2(hi)}% a month">
          <i class="iqr" style="left:${at(lo)}%;width:${at(hi) - at(lo)}%"></i>
          <i class="claim" style="left:${at(c)}%" title="What the paper claims"></i>
          <i class="med" style="left:${at(med)}%"></i>
        </div>
        <div class="hc-ends"><span>Worse quarter <b>${f2(lo)}%</b></span><span>Better quarter <b>${f2(hi)}%</b></span></div>`;
    };
    input.addEventListener("input", show);
    show();
  }

  // ---------------------------------------------------------------- signals table
  function drawSignals(sig) {
    const rows = sig.signals.filter(r => r.mom_12m != null).sort((a, b) => b.mom_12m - a.mom_12m);
    const pick = rows.slice(0, 8).concat(rows.slice(-4));
    const maxAbs = d3.max(pick, r => Math.abs(r.mom_12m)) || 1;
    document.getElementById("signals-asof").textContent = rows.length ? `as of ${rows[0].as_of} · ${rows.length} signals, best 8 and worst 4` : "";
    document.querySelector("#signals-table tbody").innerHTML = pick.map(r => `
      <tr><td title="${r.code && r.code !== r.name ? `Code: ${esc(r.code)}` : ""}">${esc(r.name)}${r.paper ? ` <span class="sub">${esc(r.paper)}</span>` : ""}</td>
      <td class="num ${r.mom_12m >= 0 ? "up" : "down"}">${(r.mom_12m * 100).toFixed(2)}%/mo
        <span class="bar-cell ${r.mom_12m >= 0 ? "pos" : "neg"}" style="width:${Math.round(40 * Math.abs(r.mom_12m) / maxAbs)}px"></span></td>
      <td class="num">${pct(r.vol_forecast, 1)}</td>
      <td class="num">${pct(r.share_kept)}</td></tr>`).join("");
    const p = sig.portfolios.filter(r => r.cost_bps === 25);
    document.getElementById("portfolio-line").textContent = p.length
      ? "Backtest Sharpe ratio after 25 bp costs: " + p.map(r => `${r.portfolio.replace(/_/g, " ")} ${num(r.sharpe)}`).join(" · ") : "";
  }

  // ---------------------------------------------------------------- test cards
  function histogram(el, noise, real) {
    const w = 300, h = 110, m = { t: 8, r: 8, b: 22, l: 8 };
    const x = d3.scaleLinear().domain(d3.extent(noise.concat([real, 0]))).nice().range([m.l, w - m.r]);
    const bins = d3.bin().domain(x.domain()).thresholds(24)(noise);
    const y = d3.scaleLinear().domain([0, d3.max(bins, b => b.length)]).range([h - m.b, m.t]);
    const s = d3.select(el).append("svg").attr("viewBox", `0 0 ${w} ${h}`);
    s.selectAll("rect").data(bins).join("rect").attr("class", "null-bar")
      .attr("x", b => x(b.x0) + 0.5).attr("width", b => Math.max(0, x(b.x1) - x(b.x0) - 1))
      .attr("y", b => y(b.length)).attr("height", b => y(0) - y(b.length));
    const p95 = d3.quantile([...noise].sort(d3.ascending), 0.95);
    s.append("line").attr("class", "p95-line").attr("x1", x(p95)).attr("x2", x(p95)).attr("y1", m.t).attr("y2", h - m.b);
    s.append("line").attr("class", "real-line").attr("x1", x(real)).attr("x2", x(real)).attr("y1", m.t).attr("y2", h - m.b);
    s.append("g").attr("class", "axis").attr("transform", `translate(0,${h - m.b})`).call(d3.axisBottom(x).ticks(5));
  }

  const NAMES = {
    spearman_ridge_minus_random: "rank skill vs random", p5_ridge_minus_random: "top-5 hits vs random",
    p15_ridge_minus_random: "top-15 hits vs random", spearman_ridge_minus_momentum: "rank skill vs 'trend continues'",
    ridge_vs_random: "reading list vs random", ridge_vs_density: "reading list vs 'crowded spots'",
  };
  const verdict = p => p == null ? "" : p < 0.05 ? `<span class="up">p ${num(p, 3)}</span>` : `<span class="down">p ${num(p, 3)}</span>`;

  function drawTests(fc) {
    const grid = document.getElementById("tests-grid");
    const order = ["H9", "H10", "H11", "H13", "H12"];
    grid.innerHTML = "";
    fc.tests.slice().sort((a, b) => order.indexOf(a.key) - order.indexOf(b.key)).forEach(t => {
      const card = document.createElement("div");
      card.className = "test-card";
      let body = "";
      if (t.noise_spearman) {
        body += `<p>Rank correlation with what actually happened: <b>${num(t.real_spearman)}</b> (red line).
          Grey: ${t.noise_spearman.length} worlds with no real growth; dashed: their 95th percentile.</p>`;
      }
      if (t.precision) body += `<p>Share of picks that became hubs: ${t.precision.map(r => `${r.model} ${pct(r.value)}`).join(" · ")}</p>`;
      if (t.key === "H10") {
        body += `<table class="mini">${t.tests.map(r => `<tr><td>${r.model}</td><td class="num">t ${num(r.t_stat)}</td><td class="num">${verdict(r.p_holm)}</td><td>${esc(r.verdict || "")}</td></tr>`).join("")}</table>`;
      } else if (t.table) {
        body += `<table class="mini">${t.table.filter(r => r.p_one_sided != null).map(r => `<tr><td>${esc(r.test)}</td><td class="num">t ${num(r.t_stat)}</td><td class="num">${verdict(r.p_one_sided)}</td></tr>`).join("")}</table>`;
      } else if (t.tests) {
        body += `<table class="mini">${t.tests.map(r => `<tr><td>${NAMES[r.model] || esc(r.model)}</td><td class="num">${num(r.value)}</td><td class="num">${verdict(r.holm_p)}</td></tr>`).join("")}</table>
          <div class="sub">p values corrected for testing several things at once (Holm). All must be under 0.05 to pass.</div>`;
      }
      card.innerHTML = `<div class="head"><span><span class="key">${t.key}</span> <b>${esc(t.label)}</b></span>
        <span class="badge ${t.passed ? "pass" : "fail"}">${t.passed ? "passed" : "did not pass"}</span></div>
        <p class="sub">${esc(t.question)}</p><div class="hist"></div>${body}`;
      grid.appendChild(card);
      if (t.noise_spearman && t.noise_spearman.length) histogram(card.querySelector(".hist"), t.noise_spearman, t.real_spearman);
    });
  }

  // ---------------------------------------------------------------- start
  (async () => {
    const [topics, fc, sig] = await Promise.all([getJSON("/api/topics"), getJSON("/api/forecasts"), getJSON("/api/signals")]);
    if (topics.error) { showError(zoom.side, topics); return; }
    topicNames = topics.topics.map(t => t.name);
    topicSize = new Map(topics.topics.map(t => [t.name, t.n]));
    zoom.select.innerHTML = topics.topics.slice().sort((a, b) => a.name.localeCompare(b.name))
      .map(t => `<option value="${esc(t.name)}">${esc(t.name)}${t.predicted_boom ? " ▲" : ""}</option>`).join("");
    if (!fc.error) {
      forecastByTopic = new Map(fc.topics.map(t => [t.topic, t]));
      fcCount = { n: fc.n_topics ?? fc.topics.length, nMap: fc.n_map ?? 8, top: fc.top_topics ?? fc.topics.filter(t => t.predicted_boom).length,
                  subN: fc.n_subtopics ?? fc.subtopics.length, subTop: fc.top_subtopics ?? 15 };
      document.querySelectorAll(".fc-top").forEach(el => { el.textContent = fcCount.top; });
      document.querySelectorAll(".fc-n").forEach(el => { el.textContent = fcCount.n; });
      const lg = document.getElementById("lg-ring");
      if (lg) lg.title = `Predicted to grow over the next three years (top ${fcCount.top} of ${fcCount.n} topics), `
        + "from 2025 data, so shown from 2025 on. Exploratory: the forecast did not pass its test.";
      drawNext(fc);
      drawTests(fc);
      syncRings();
    } else showError(document.getElementById("tests-grid"), fc);
    if (!sig.error) { drawDecay(sig); drawSignals(sig); } else showError(document.getElementById("decay-plot"), sig);
    zoom.select.value = topicNames.includes(DEFAULT_TOPIC) ? DEFAULT_TOPIC : topicNames[0];
    zoom.side.innerHTML = `<div class="sub">No topic open yet: click a circle on the map above, or pick one from the list.</div>`;
  })();
})();
