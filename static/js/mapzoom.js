// Zoom and pan for both maps (the topic network and a topic's sub-topic map).
// Scroll or pinch to zoom, drag to pan, or use the +, − and reset buttons in the
// bottom-right corner. The map never zooms out past its normal size, and panning
// stops at its edges, so it can't be lost off screen.
window.MapZoom = (() => {
  const MAX = 5, STEP = 1.6, MS = 250;

  // Zoom svg by moving layer (a <g> holding everything drawn). W x H is the viewBox.
  // onZoom(k) is told the scale after every change.
  function attach(svg, layer, W, H, onZoom) {
    const z = d3.zoom().scaleExtent([1, MAX]).clickDistance(5)  // a click that wobbles a little is still a click
      .on("zoom", e => {
        layer.attr("transform", e.transform);
        if (onZoom) onZoom(e.transform.k);
      });
    const size = (w, h) => z.extent([[0, 0], [w, h]]).translateExtent([[0, 0], [w, h]]);
    size(W, H);
    svg.call(z).on("dblclick.zoom", null);  // a double click opens things; it shouldn't zoom too
    const by = f => svg.transition().duration(MS).call(z.scaleBy, f);
    return {
      zoomIn: () => by(STEP),
      zoomOut: () => by(1 / STEP),
      reset: (ms = MS) => (ms ? svg.transition().duration(ms) : svg).call(z.transform, d3.zoomIdentity),
      size,
    };
  }

  // The three buttons, added to parent. target() returns the attach() result to act on
  // (the main map is redrawn on resize, so it changes).
  function controls(parent, target) {
    const box = d3.select(parent).append("div").attr("class", "zoom-ctl")
      .attr("role", "group").attr("aria-label", "Zoom");
    const btn = (text, label, act) => box.append("button").attr("type", "button")
      .attr("class", "path-btn").attr("title", label).attr("aria-label", label).text(text)
      .on("click", e => { e.stopPropagation(); const t = target(); if (t) act(t); });
    btn("+", "Zoom in", t => t.zoomIn());
    btn("−", "Zoom out", t => t.zoomOut());
    const reset = btn("⟲", "Reset zoom", t => t.reset());
    // the reset button only shows once the map is zoomed
    return { show: k => box.classed("is-zoomed", k > 1.01), node: box.node(), reset };
  }

  return { attach, controls };
})();
