// Fad check (optional add-on): how each map topic's share of research has moved.
// Descriptive only, no prediction. Adds its own
// header button and drawer, so removing it means deleting this file, fads.css and the
// lines marked "Fad check" in html_main.py and live.js (see fad_data.py).
// Data: fad_data.py (/api/fads). Uses d3 and live.js's window.radarDialog.
(() => {
  const esc = s => String(s ?? "").replace(/[&<>"]/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
  const LABELS = {
    booming: ["booming", "Its share is at least 1.5 times, and 1 point above, what it was five years ago."],
    "past fad": ["past fad", "Boomed, then fell to under 60% of its peak."],
    fading: ["fading", "Below 60% of its peak without a boom first: a slow decline."],
    steady: ["steady", "Neither booming nor far below its peak."],
  };

  // header button, after "Where research is heading"
  const actions = document.querySelector(".head-actions");
  const btn = document.createElement("button");
  btn.type = "button";
  btn.className = "head-btn ghost";
  btn.id = "fad-btn";
  btn.setAttribute("aria-haspopup", "dialog");
  btn.innerHTML = `<svg class="ico" viewBox="0 0 20 20" aria-hidden="true"><path d="M2 16c3 0 4-10 7-10s3 6 5 6 2-3 4-3"/></svg><span>Fad check</span>`;
  actions.appendChild(btn);

  const drawer = document.createElement("aside");
  drawer.className = "drawer wide";
  drawer.id = "fad-drawer";
  drawer.hidden = true;
  drawer.setAttribute("role", "dialog");
  drawer.setAttribute("aria-modal", "true");
  drawer.setAttribute("aria-labelledby", "fad-title");
  drawer.innerHTML = `
    <div class="dlg-head">
      <div><h2 id="fad-title">Fad check</h2><span class="sub" id="fad-sub"></span></div>
      <button type="button" class="close-btn" aria-label="Close">&times;</button>
    </div>
    <div class="fad-grid" id="fad-grid"></div>
    <p class="fad-key" id="fad-key"></p>`;
  document.body.appendChild(drawer);

  const dialog = window.radarDialog;
  drawer.querySelector(".close-btn").addEventListener("click", () => dialog.close());
  // a card opens its topic on the map (radar.js listens for data-topic), so close first
  drawer.addEventListener("click", e => { if (e.target.closest("[data-topic]")) dialog.close(); }, true);

  let loaded = false;
  btn.addEventListener("click", () => {
    dialog.open(drawer);
    if (!loaded) { loaded = true; load(); }
  });

  async function load() {
    const d = await fetch("/api/fads").then(r => r.json());
    const grid = document.getElementById("fad-grid");
    if (d.error) { grid.textContent = d.error; return; }
    const r = d.rules;
    document.getElementById("fad-sub").textContent =
      `Share of each year's sampled papers, ${d.years[0]}–${d.years[1]}. Line: ${r.smooth}-year average; dots: single years.`;
    const ymax = d3.max(d.topics, t => d3.max(t.series, p => p.share));
    grid.innerHTML = d.topics.map((t, i) => `
      <div class="fad-card" data-topic="${esc(t.topic)}" title="Open ${esc(t.topic)} on the map">
        <div class="fad-name">${esc(t.topic)}</div>
        <span class="fad-tag ${t.label.replace(" ", "-")}" title="${esc(LABELS[t.label][1])}">${LABELS[t.label][0]}</span>
        <svg class="fad-spark" id="fad-spark-${i}" viewBox="0 0 200 56" aria-hidden="true"></svg>
        <div class="fad-nums">${t.five_ago}% → <b>${t.now}%</b> in five years · peak ${t.peak}% (${t.peak_year})</div>
      </div>`).join("");
    d.topics.forEach((t, i) => spark(`#fad-spark-${i}`, t.series, d.years, ymax, t.label));
    document.getElementById("fad-key").innerHTML =
      `<b>Booming</b>: share at least ${r.ratio}× and ${r.points} point above five years ago. ` +
      `<b>Fading</b>: under ${Math.round(100 * r.fade)}% of its peak. <b>Past fad</b>: fading after a boom. ` +
      `<b>Steady</b>: neither. Labels use ${d.last_full}, the last full year.`;
  }

  function spark(sel, series, years, ymax, label) {
    const x = d3.scaleLinear().domain(years).range([2, 198]);
    const y = d3.scaleLinear().domain([0, ymax]).range([54, 2]);
    const svg = d3.select(sel);
    svg.selectAll("circle").data(series).join("circle")
      .attr("class", "fad-dot").attr("cx", p => x(p.year)).attr("cy", p => y(p.share)).attr("r", 1.4);
    svg.append("path").attr("class", `fad-line ${label.replace(" ", "-")}`)
      .attr("d", d3.line().defined(p => p.smooth != null).x(p => x(p.year)).y(p => y(p.smooth))(series));
  }
})();
