const PHASE_COLORS = {
  eccentric: "#ef4444",
  concentric: "#22c55e",
};

function fmt(v, digits) {
  return v == null ? "—" : v.toFixed(digits);
}

function renderRepTable(containerId, phases) {
  const container = document.getElementById(containerId);
  if (!container) return;

  if (!phases.length) {
    container.innerHTML = "";
    return;
  }

  const reps = new Map();
  phases.forEach((p) => {
    if (!reps.has(p.rep)) reps.set(p.rep, {});
    reps.get(p.rep)[p.phase] = p;
  });

  const rows = [...reps.entries()].sort((a, b) => a[0] - b[0]).map(([rep, byPhase]) => {
    const ecc = byPhase.eccentric;
    const con = byPhase.concentric;
    return `
      <tr class="border-b border-gray-800 last:border-0">
        <td class="py-2 px-3 text-white font-medium">${rep}</td>
        <td class="py-2 px-3 text-gray-300">${ecc ? fmt(ecc.duration, 2) + " s" : "—"}</td>
        <td class="py-2 px-3 text-gray-300">${con ? fmt(con.duration, 2) + " s" : "—"}</td>
        <td class="py-2 px-3 text-gray-300">${con ? fmt(con.v_avg, 2) + " m/s" : "—"}</td>
        <td class="py-2 px-3 text-gray-300">${con ? fmt(con.v_peak, 2) + " m/s" : "—"}</td>
        <td class="py-2 px-3 text-gray-300">${con && con.v_sticking != null ? fmt(con.v_sticking, 2) + " m/s" : "—"}</td>
        <td class="py-2 px-3 text-gray-300">${con && con.rir != null ? "~" + Math.round(con.rir) : "—"}</td>
      </tr>`;
  }).join("");

  container.innerHTML = `
    <div class="bg-card border border-gray-700 rounded-xl overflow-x-auto">
      <table class="w-full text-sm text-left whitespace-nowrap">
        <thead>
          <tr class="text-gray-500 uppercase text-xs border-b border-gray-700">
            <th class="py-2 px-3 font-medium">Rep</th>
            <th class="py-2 px-3 font-medium">Eccentric</th>
            <th class="py-2 px-3 font-medium">Concentric</th>
            <th class="py-2 px-3 font-medium">Avg vel.</th>
            <th class="py-2 px-3 font-medium">Peak vel.</th>
            <th class="py-2 px-3 font-medium">Sticking vel.</th>
            <th class="py-2 px-3 font-medium">RIR (est.)</th>
          </tr>
        </thead>
        <tbody>${rows}</tbody>
      </table>
    </div>
    <p class="text-gray-600 text-xs mt-2 px-1">
      RIR estimado a partir de la pérdida de velocidad del sticking point respecto a la 1ª rep
      (González-Badillo et al., 2017, <em>Int J Sports Med</em> 38(3):217-225) —
      ecuación validada en press de banca, aproximada para otros ejercicios.
    </p>`;
}

async function renderChart(runId, containerId) {
  containerId = containerId || "chart";
  const res = await fetch(`/runs/${runId}/chart-data`);
  const data = await res.json();

  const traces = [{
    x: data.t,
    y: data.height,
    type: "scatter",
    mode: "lines",
    line: { color: "#3b82f6", width: 1.6 },
    name: "height",
    hoverinfo: "skip",
  }];

  const shapes = data.phases.map((p) => ({
    type: "rect",
    xref: "x",
    yref: "paper",
    x0: p.t0,
    x1: p.t1,
    y0: 0,
    y1: 1,
    fillcolor: PHASE_COLORS[p.phase] || "#888",
    opacity: 0.18,
    line: { width: 0 },
  }));

  // invisible full-height fill per phase: gives a hover target over the whole
  // shaded region (any y, not just near a single marker point) via Plotly's
  // hoveron:"fills", without matplotlib-style image-map fragility
  data.phases.forEach((p) => {
    const text = p.phase === "concentric"
      ? `rep ${p.rep} — concentric<br>avg: ${p.v_avg.toFixed(2)} m/s` +
        (p.v_sticking != null ? `<br>sticking point: ${p.v_sticking.toFixed(2)} m/s` : "") +
        (p.rir != null ? `<br>RIR (est.): ~${Math.round(p.rir)}` : "")
      : `rep ${p.rep} — eccentric<br>avg: ${p.v_avg.toFixed(2)} m/s`;
    traces.push({
      x: [p.t0, p.t1, p.t1, p.t0],
      y: [-0.05, -0.05, 1.1, 1.1],
      fill: "toself",
      fillcolor: "rgba(0,0,0,0)",
      line: { width: 0 },
      mode: "none",
      hoveron: "fills",
      // hovertemplate is silently ignored for hoveron:"fills" in this Plotly
      // version — it falls back to a generic "trace N" label. text +
      // hoverinfo:"text" is what actually renders custom fill-hover content.
      text,
      hoverinfo: "text",
      showlegend: false,
    });
  });

  // Crop to ~1s before the first rep and ~1s after the last one, clamped to
  // the actual data extent so we never zoom past the recorded track.
  let xRange;
  if (data.phases.length && data.t.length) {
    const dataMin = data.t[0];
    const dataMax = data.t[data.t.length - 1];
    const firstT0 = Math.min(...data.phases.map((p) => p.t0));
    const lastT1 = Math.max(...data.phases.map((p) => p.t1));
    xRange = [Math.max(dataMin, firstT0 - 1), Math.min(dataMax, lastT1 + 1)];
  }

  const layout = {
    shapes,
    margin: { t: 20, r: 20 },
    xaxis: { title: "time (s)", range: xRange, gridcolor: "#334155", color: "#94a3b8", zerolinecolor: "#334155" },
    yaxis: { title: "normalized height", range: [-0.05, 1.1], gridcolor: "#334155", color: "#94a3b8", zerolinecolor: "#334155" },
    height: 480,
    paper_bgcolor: "#1e293b",
    plot_bgcolor: "#1e293b",
    font: { color: "#cbd5e1" },
    legend: { font: { color: "#cbd5e1" } },
  };

  await Plotly.newPlot(containerId, traces, layout, { responsive: true, displayModeBar: false });
  renderRepTable(`rep-table-${runId}`, data.phases);
}

// Charts are rendered after htmx settles the swap (not via an inline <script>
// that runs mid-swap) so the container has its final layout size before
// Plotly reads it — rendering too early left shapes/axes laid out for a
// stale (often zero) width that never got corrected on resize.
document.addEventListener("htmx:afterSettle", function (event) {
  event.target.querySelectorAll("[data-render-chart]").forEach(function (el) {
    if (el.dataset.rendered) return;
    el.dataset.rendered = "1";
    renderChart(el.dataset.renderChart, el.id);
  });
});
