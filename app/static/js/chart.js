async function renderChart(runId) {
  const res = await fetch(`/runs/${runId}/chart-data`);
  const data = await res.json();

  const traces = [{
    x: data.t,
    y: data.height,
    type: "scatter",
    mode: "lines",
    line: { color: "#1f77b4", width: 1.6 },
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
    fillcolor: p.phase === "eccentric" ? "#d62728" : "#2ca02c",
    opacity: 0.15,
    line: { width: 0 },
  }));

  // invisible marker per phase at its midpoint: gives a hover target over the
  // shaded region without matplotlib-style image-map fragility
  data.phases.forEach((p) => {
    const midT = (p.t0 + p.t1) / 2;
    const text = p.phase === "concentric"
      ? `rep ${p.rep} — concentric<br>avg: ${p.v_avg.toFixed(2)} m/s` +
        (p.v_sticking != null ? `<br>sticking point: ${p.v_sticking.toFixed(2)} m/s` : "")
      : `rep ${p.rep} — eccentric<br>avg: ${p.v_avg.toFixed(2)} m/s`;
    traces.push({
      x: [midT],
      y: [0.5],
      yaxis: "y2",
      mode: "markers",
      marker: { size: 40, color: "rgba(0,0,0,0)" },
      hovertemplate: text + "<extra></extra>",
      showlegend: false,
    });
  });

  const layout = {
    shapes,
    margin: { t: 20, r: 20 },
    xaxis: { title: "time (s)" },
    yaxis: { title: "normalized height", range: [-0.05, 1.1] },
    yaxis2: { overlaying: "y", range: [0, 1], visible: false },
    height: 480,
  };

  Plotly.newPlot("chart", traces, layout, { responsive: true });
}
