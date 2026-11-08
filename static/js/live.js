// Live layer (Person D): crowding score, allocation backtest and the spoken briefing.
// Data comes from live_data.py (/api/crowding, /api/backtest, /api/briefing), written
// ahead of time by `python -m src.live.run_live`. Uses d3 from the page; touches only
// its own panels (#crowding-panel, #backtest-panel, #briefing-panel).
(() => {
  const esc = s => String(s ?? "").replace(/[&<>"]/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
  const getJSON = url => fetch(url).then(r => r.json());
  const signed = (v, d = 2) => v == null ? "–" : (v > 0 ? "+" : v < 0 ? "−" : "") + Math.abs(v).toFixed(d);
  const showError = (id, res) => {
    document.getElementById(id).insertAdjacentHTML("beforeend", `<div class="sub">${esc(res.error)}</div>`);
  };

  // Bar centred on zero for a score on roughly a ±3 scale; +2 or more is "strong".
  function scoreBar(score) {
    const w = Math.min(Math.abs(score), 3) / 3 * 50;
    const cls = score >= 0 ? "pos" + (score >= 2 ? " strong" : "") : "neg";
    const left = score >= 0 ? 50 : 50 - w;
    return `<span class="zbar"><i class="${cls}" style="left:${left}%;width:${w}%"></i></span>`;
  }

  function change(now, before, label) {
    const d = now - before;
    if (d > 0) return `<span class="arrow up" title="Up ${d} from ${label}">&#9650;</span>`;
    if (d < 0) return `<span class="arrow down" title="Down ${-d} from ${label}">&#9660;</span>`;
    return `<span class="arrow same" title="Same as ${label}">&ndash;</span>`;
  }

  const periodName = (unit, start) => unit === "year" ? start.slice(0, 4)
    : "week of " + new Date(start + "T00:00:00Z").toLocaleDateString("en-GB", { day: "numeric", month: "short", timeZone: "UTC" });

  // ---------------------------------------------------------------- crowding
  async function loadCrowding() {
    const c = await getJSON("/api/crowding");
    if (c.error) return showError("crowding-panel", c);
    const yearly = c.unit === "year";
    const now = periodName(c.unit, c.period_start), prev = c.prev_start ? periodName(c.unit, c.prev_start) : "before";
    document.getElementById("crowding-title").textContent = yearly ? `Crowded in ${now}` : "Crowded this week";
    document.getElementById("crowding-sub").textContent = yearly
      ? `OpenAlex quant-finance papers (B's sample of 380 a year). Score compares each area's share of ${now} so far with its share over the ${c.baseline} years before: 0 is normal, +2 unusually crowded.`
      : `New arXiv q-fin papers, tagged by Gemini. Score compares each area's share of the last ${c.window} weeks with its usual share: 0 is normal, +2 unusually crowded.`;
    document.getElementById("crowding-papers").textContent = yearly ? `Papers ${now}` : "Papers";
    document.querySelector("#crowding tbody").innerHTML = c.topics.map(t => {
      const ex = t.example;
      const tip = ex ? (yearly ? `Most cited in ${now}: ${ex.title} (${ex.citations} citations)`
                               : `Newest: ${ex.title} (${ex.published})`) : "";
      return `
      <tr title="${esc(tip)}">
        <td>${esc(t.topic)}</td>
        <td class="num"><div class="pop-bar">${signed(t.score)} ${scoreBar(t.score ?? 0)}</div></td>
        <td class="num papers" title="${t.papers_prev} in ${esc(prev)}">${t.papers} ${change(t.papers, t.papers_prev, prev)}</td>
      </tr>`;
    }).join("");
  }

  // ---------------------------------------------------------------- backtest
  function drawBacktest(bt) {
    const box = document.getElementById("bt-chart");
    box.innerHTML = "";
    const W = box.clientWidth || 360, H = 170, M = { t: 8, r: 8, b: 20, l: 38 };
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
    if (bt.error) return showError("backtest-panel", bt);
    document.getElementById("bt-range").textContent = `${bt.start.slice(0, 4)}–${bt.end.slice(0, 4)}`;
    drawBacktest(bt);
    document.querySelector("#backtest tbody").innerHTML = bt.rows.map(r => `
      <tr class="${r.key === "decay_tilt" ? "ours" : ""}">
        <td>${esc(r.label)}</td>
        <td class="num">${r.sharpe.toFixed(2)} <span class="sub">(${signed(r.ci[0])}, ${signed(r.ci[1])})</span></td>
        <td class="num">${r.vs_equal == null ? "" :
          `<span class="${r.vs_equal_ci[0] > 0 || r.vs_equal_ci[1] < 0 ? "sig" : ""}">${signed(r.vs_equal)}</span>
           <span class="sub">(${signed(r.vs_equal_ci[0])}, ${signed(r.vs_equal_ci[1])})</span>`}</td>
      </tr>`).join("");
  }

  // ---------------------------------------------------------------- briefing
  async function loadBriefing() {
    const b = await getJSON("/api/briefing");
    if (b.error) return showError("briefing-panel", b);
    if (b.audio) {
      const audio = document.getElementById("brief-audio");
      audio.src = "/media/briefing.mp3";
      audio.hidden = false;
    }
    document.getElementById("brief-text").textContent = b.text;
    const c = b.card;
    if (!c) return;
    const yearly = c.unit === "year";
    const when = periodName(c.unit, c.period_start);
    const src = (c.text_source || "").startsWith("gemini") ? "worded by Gemini, numbers checked" : "template text";
    document.getElementById("brief-title").textContent = yearly ? `Briefing: ${when} so far` : "This week's briefing";
    document.getElementById("brief-source").textContent = src;
    const span = yearly ? `papers in ${when}` : `papers / ${c.window} wks`;
    const a = c.allocation, cv = c.caveat;
    const held = a.held.length ? a.held.map(h => `<li><b>${esc(h.signal)}</b> ${Math.round(h.weight * 100)}%</li>`).join("")
                               : "<li>Cash: no signal clears the cost hurdle</li>";
    document.getElementById("brief-card").innerHTML = `
      <div class="card-grid">
        <div><span class="sub">Most crowded vs usual</span>
          <b>${esc(c.busiest[0].area)}</b> <span class="sub">${signed(c.busiest[0].score, 1)}, ${c.busiest[0].papers} ${span}</span></div>
        <div><span class="sub">Quietest vs usual</span>
          <b>${esc(c.quietest.area)}</b> <span class="sub">${signed(c.quietest.score, 1)}</span></div>
        <div><span class="sub">Hold for ${esc(a.for_month)}</span><ul class="held">${held}</ul>
          <span class="sub">Skipped: ${a.skipped.map(esc).join(", ") || "none"}</span></div>
        <div><span class="sub">Caveat</span>
          <span>Beat equal weight by ${signed(cv.sharpe_minus_equal_weight)} Sharpe (95% CI ${signed(cv.ci_low)} to ${signed(cv.ci_high)})${cv.significant ? "" : ": not proven"}.</span></div>
      </div>`;
  }

  loadCrowding(); loadBacktest(); loadBriefing();
})();
