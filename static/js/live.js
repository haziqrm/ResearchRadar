// Live layer (Person D): the headline above the map, "Where research is heading", the
// one-line lead on the decay panel, the ? drawer and the Weekly briefing modal.
// Data: live_data.py (/api/heading, /api/briefing, /api/backtest) and Person A's
// /api/signals. Uses d3 and crowdColour() from the page, so the table's shading matches
// the map circles. Rows carry data-topic; radar.js opens that topic on click.
(() => {
  const $ = id => document.getElementById(id);
  const esc = s => String(s ?? "").replace(/[&<>"]/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
  const getJSON = url => fetch(url).then(r => r.json());
  const signed = (v, d = 2) => {   // rounded first, so -0.02 shows as 0.0, not −0.0
    if (v == null) return "–";
    const r = +v.toFixed(d);
    return (r > 0 ? "+" : r < 0 ? "−" : "") + Math.abs(r).toFixed(d);
  };
  const colour = s => typeof crowdColour === "function" ? crowdColour(s) : "transparent";
  const ink = s => s != null && Math.abs(s) >= 1.6 ? "#fff" : "var(--text)";  // readable on the shade
  // crowding is a z-score: standard deviations above (+) or below (−) the usual share
  const sd = s => s == null ? "–" : `${signed(s, 1)} SD`;
  const plural = (n, word) => `${n} ${word}${n === 1 ? "" : "s"}`;
  const chip = s => `<span class="chip" style="background:${colour(s)};color:${ink(s)}">${sd(s)}</span>`;
  // Table bar: on a soft track from −5 to +5 SD, with the "normal" ±2 SD zone shaded and a
  // zero line. The bar grows right (busier) or left (quieter) and is rounded at its outer end.
  const BAR_MAX = 5, NORMAL = 2;
  const pos = v => (Math.max(-BAR_MAX, Math.min(BAR_MAX, v)) + BAR_MAX) / (2 * BAR_MAX) * 100;
  const tone = s => s == null ? "none" : s >= NORMAL ? "hot" : s <= -NORMAL ? "cold" : "normal";
  const crowdBar = s => {
    const track = `<i class="normal" style="left:${pos(-NORMAL)}%;width:${pos(NORMAL) - pos(-NORMAL)}%"></i><i class="zero"></i>`;
    if (s == null) return `<span class="cbar">${track}</span>`;
    const w = Math.abs(pos(s) - 50);
    const fill = s >= 0 ? `left:50%;width:${w}%` : `right:50%;width:${w}%`;
    return `<span class="cbar">${track}<i class="fill ${s >= 0 ? "pos" : "neg"}" style="${fill};background:${colour(s)}"></i></span>`;
  };
  const periodLabel = (unit, start) => unit === "year" ? `${start.slice(0, 4)} so far` : "This week";
  const SEEN_KEY = "radar-briefing-seen";
  const store = {
    get: k => { try { return localStorage.getItem(k); } catch (_) { return null; } },
    set: (k, v) => { try { localStorage.setItem(k, v); } catch (_) { /* private mode */ } },
  };

  // ---------------------------------------------------------------- dialogs
  const scrim = $("scrim");
  let openDlg = null, lastFocus = null;
  function openDialog(el) {
    closeDialog();
    lastFocus = document.activeElement;
    el.hidden = false;
    scrim.hidden = false;
    openDlg = el;
    el.querySelector(".close-btn").focus();
  }
  function closeDialog() {
    if (!openDlg) return;
    openDlg.hidden = true;
    scrim.hidden = true;
    openDlg = null;
    if (lastFocus) lastFocus.focus();
  }
  scrim.addEventListener("click", closeDialog);
  document.querySelectorAll("[data-close]").forEach(b => b.addEventListener("click", closeDialog));
  // capture phase: Escape closes the dialog first, without also closing an open topic
  document.addEventListener("keydown", e => {
    if (e.key === "Escape" && openDlg) { e.stopImmediatePropagation(); closeDialog(); }
  }, true);
  $("help-btn").addEventListener("click", () => openDialog($("help-drawer")));

  // ---------------------------------------------------------------- where research is heading
  async function loadHeading() {
    const h = await getJSON("/api/heading");
    const body = document.querySelector("#heading tbody");
    if (h.error) { body.innerHTML = `<tr><td colspan="4" class="sub">${esc(h.error)}</td></tr>`; return; }
    $("heading-period").textContent = `crowding: ${periodLabel(h.unit, h.period_start)} · click a row to open it on the map`;
    if (h.read_first_years) {
      const [a, b] = h.read_first_years;
      $("read-first-th").title = `The topic's most cited paper per year since publication, among papers from ${a}–${b}.`;
    }
    body.innerHTML = h.topics.map(t => {
      const f = t.forecast;
      const fc = !f ? "—" : `#${f.rank} of ${f.of}` + (f.likely ? ` <span class="likely">predicted to grow</span>` : "");
      const fTip = f ? `#${f.rank} of the ${f.of} topics on the map (#${f.rank_all} of all ${f.of_all} forecast topics). `
        + `"Predicted to grow" means in the forecast's top ${f.top} of all ${f.of_all}. Exploratory.` : "";
      const r = t.read_first;
      const read = r ? `<span class="title" title="${esc(r.title)}">${esc(r.title)}</span>${r.year} · ${plural(r.citations, "citation")}` : "—";
      return `
      <tr data-topic="${esc(t.topic)}" tabindex="0" title="Open ${esc(t.topic)} on the map">
        <td class="topic">${esc(t.topic)}</td>
        <td class="crowd" title="${plural(t.papers, "paper")} in ${h.period_start.slice(0, 4)}"><span class="v ${tone(t.score)}">${sd(t.score)}</span>${crowdBar(t.score)}</td>
        <td class="forecast" title="${esc(fTip)}">${fc}</td>
        <td class="read">${read}</td>
      </tr>`;
    }).join("");
    body.querySelectorAll("tr[data-topic]").forEach(tr => tr.addEventListener("keydown", e => {
      if (e.key === "Enter" || e.key === " ") { e.preventDefault(); tr.click(); }
    }));
  }

  // ---------------------------------------------------------------- decay panel lead
  async function loadDecayLead() {
    const sig = await getJSON("/api/signals");
    if (sig.error) return;
    const after = sig.curve.find(r => r.years_from === 0), med = sig.haircut && sig.haircut.median;
    if (!after || med == null || !$("decay-lead")) return;
    // The finding is the title (the chart carries the numbers); the line under it says how to read the chart.
    const kept = after.share_of_in_sample_mean;
    if (kept > 0.4 && kept < 0.6 && $("decay-title")) $("decay-title").textContent = "Published signals keep about half their edge";
    $("decay-lead").textContent = `Return after publication as a share of what the paper reported, by years since publication`
      + `${sig.haircut.n_signals ? `, across ${sig.haircut.n_signals} signals` : ""}. Shaded band: 95% range.`;
  }

  // ---------------------------------------------------------------- briefing: headline, button, modal
  const mondayOf = iso => {
    const d = new Date(iso + "T00:00:00Z");
    d.setUTCDate(d.getUTCDate() - ((d.getUTCDay() + 6) % 7));
    return d.toLocaleDateString("en-GB", { day: "numeric", month: "short", timeZone: "UTC" });
  };

  async function loadBriefing() {
    const b = await getJSON("/api/briefing");
    if (b.error || !b.card) return;
    const c = b.card, top = c.busiest[0], q = c.quietest;
    const when = periodLabel(c.unit, c.period_start);

    $("headline").innerHTML = `<b>${esc(when)}:</b> ${esc(top.area)} is the most crowded area `
      + `(<span class="score" title="Standard deviations above its usual share of papers">${sd(top.score)}</span> vs its usual share); `
      + `${esc(q.area)} is the quietest (<span class="score">${sd(q.score)}</span>).`;

    // "new" dot: the briefing is newer than the last one this browser opened
    const seen = store.get(SEEN_KEY);
    $("brief-new").hidden = !c.generated || (seen && seen >= c.generated);

    $("brief-date").textContent = (c.generated ? `Week of ${mondayOf(c.generated)} · ` : "")
      + `crowding for ${when.toLowerCase()}`;
    if (b.audio) { $("brief-audio").src = "/media/briefing.mp3"; $("brief-audio").hidden = false; }
    $("brief-text").textContent = b.text;
    const a = c.allocation, cv = c.caveat;
    const held = a.held.length ? a.held.map(h => `<b>${esc(h.signal)}</b> ${Math.round(h.weight * 100)}%`).join(", ")
                               : "nothing: cash, because no signal clears the cost hurdle";
    const record = `beat equal weight by ${signed(cv.sharpe_minus_equal_weight)} Sharpe (95% CI ${signed(cv.ci_low)} to ${signed(cv.ci_high)})`;
    $("brief-card").innerHTML = `
      <div class="card-grid">
        <div><span class="label">Most crowded vs usual</span>
          <ol>${c.busiest.map(r => `<li>${esc(r.area)}${chip(r.score)}</li>`).join("")}</ol></div>
        <div><span class="label">Quietest</span><span>${esc(q.area)}${chip(q.score)}</span></div>
        <div class="wide"><span class="label">Hold for ${esc(a.for_month)}</span><span>${held}</span>
          ${a.skipped.length ? `<span class="sub">Skipped after costs: ${a.skipped.map(esc).join(", ")}</span>` : ""}</div>
        <div class="wide"><span class="label">Caveat</span>
          <span>${cv.significant ? `Since 1981 the rule ${record}.` : `Edge not proven: since 1981 the rule ${record}.`}</span></div>
      </div>`;

    $("brief-btn").addEventListener("click", () => {
      openDialog($("brief-modal"));
      if (c.generated) { store.set(SEEN_KEY, c.generated); $("brief-new").hidden = true; }
    });
  }

  // ---------------------------------------------------------------- backtest (inside the modal)
  let backtest = null;
  function drawBacktest(bt) {
    const box = $("bt-chart");
    box.innerHTML = "";
    const W = box.clientWidth || 520, H = 170, M = { t: 8, r: 8, b: 20, l: 38 };
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
    if (bt.error) return;
    backtest = bt;
    $("bt-range").textContent = `${bt.start.slice(0, 4)}–${bt.end.slice(0, 4)}`;
    document.querySelector("#backtest tbody").innerHTML = bt.rows.map(r => `
      <tr class="${r.key === "decay_tilt" ? "ours" : ""}">
        <td>${esc(r.label)}</td>
        <td class="num">${r.sharpe.toFixed(2)} <span class="sub">(${signed(r.ci[0])}, ${signed(r.ci[1])})</span></td>
        <td class="num">${r.vs_equal == null ? "" :
          `<span class="${r.vs_equal_ci[0] > 0 || r.vs_equal_ci[1] < 0 ? "sig" : ""}">${signed(r.vs_equal)}</span>
           <span class="sub">(${signed(r.vs_equal_ci[0])}, ${signed(r.vs_equal_ci[1])})</span>`}</td>
      </tr>`).join("");
    // the chart needs a width, so draw it when its section is first opened
    $("bt-chart").closest("details").addEventListener("toggle", e => {
      if (e.target.open && backtest && !$("bt-chart").firstChild) drawBacktest(backtest);
    });
  }

  loadHeading(); loadDecayLead(); loadBriefing(); loadBacktest();
})();
