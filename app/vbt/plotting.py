"""Static matplotlib rendering of a run's phases, saved as a downloadable
artifact (`data/runs/{id}/phases.png`). The interactive chart shown in the
main panel is built client-side from `/runs/{id}/chart-data` (see
app/static/js/chart.js) — this PNG is not used there.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from app.vbt.phases import CONCENTRIC, ECCENTRIC, Phase


def save_phases_plot(path: Path, t: np.ndarray, height: np.ndarray, phases: list[Phase], title: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(14, 5))
    for phase in phases:
        color = "#d62728" if phase.kind == ECCENTRIC else "#2ca02c"
        ax.axvspan(phase.t0, phase.t1, color=color, alpha=0.16, lw=0)
        if phase.kind == CONCENTRIC:
            ax.annotate(str(phase.rep), (0.5 * (phase.t0 + phase.t1), 1.04),
                        ha="center", fontsize=9, color="#2ca02c")
    ax.axhline(0.85, ls=":", lw=0.9, color="gray")
    ax.axhline(0.15, ls=":", lw=0.9, color="gray")
    ax.plot(t, height, lw=1.4, color="#1f77b4")
    ax.set_xlabel("time (s)")
    ax.set_ylabel("normalized height")
    ax.set_title(f"{title} — red: eccentric, green: concentric (numbered by rep)")
    ax.set_ylim(-0.05, 1.12)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
