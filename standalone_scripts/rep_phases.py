#!/usr/bin/env python3
"""Separa la curva de altura del disco en fases (bajada / subida) y calcula
metricas de VBT para cada una.

Parte del CSV que genera track_height.py. La escala px->metros sale del propio
disco: un disco olimpico estandar mide 45 cm de diametro, asi que el diametro
medio de la mascara dentro de cada fase da los metros por pixel de esa fase,
sin calibrar la camara. Calcularlo por fase y no una vez por video absorbe los
cambios de escala aparente (el atleta que se acerca o se aleja entre series).

La segmentacion es una maquina de 4 estados sobre la altura normalizada:
arriba -> bajando -> abajo -> subiendo. Los umbrales llevan histeresis, asi que
el ruido alrededor de un umbral no genera fases falsas.

Uso:
    python rep_phases.py runs/tracking/red2_altura.csv
    python rep_phases.py runs/tracking/inftest_circ/black2_altura.csv --start 25 --end 50
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

DISK_DIAMETER_M = 0.45  # disco olimpico estandar
ARRIBA, ABAJO, BAJANDO, SUBIENDO = "arriba", "abajo", "bajando", "subiendo"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fases y metricas VBT del disco.")
    parser.add_argument("csv", help="CSV *_altura.csv generado por track_height.py.")
    parser.add_argument("--start", type=float, default=None,
                        help="Segundo inicial a analizar (recorta el unrack).")
    parser.add_argument("--end", type=float, default=None,
                        help="Segundo final a analizar (recorta el rerack).")
    parser.add_argument("--no-auto-window", action="store_true",
                        help="No detectar la serie de trabajo; analizar todo el vídeo.")
    parser.add_argument("--min-amp", type=float, default=0.30,
                        help="Amplitud mínima de una excursión, en fracción del recorrido total.")
    parser.add_argument("--max-gap-s", type=float, default=4.0,
                        help="Separación máxima (s) entre excursiones de la misma serie.")
    parser.add_argument("--pad", type=float, default=1.0,
                        help="Margen (s) que se añade a cada lado de la ventana detectada.")
    parser.add_argument("--top", type=float, default=0.85,
                        help="Altura normalizada por encima de la cual se considera 'arriba'.")
    parser.add_argument("--bottom", type=float, default=0.15,
                        help="Altura normalizada por debajo de la cual se considera 'abajo'.")
    parser.add_argument("--vel-frac", type=float, default=0.10,
                        help="Fraccion del pico de velocidad por debajo de la cual la fase termina.")
    parser.add_argument("--min-phase", type=float, default=0.15,
                        help="Duracion minima (s) de una fase para no descartarla.")
    parser.add_argument("--smooth", type=int, default=5,
                        help="Ventana de media movil sobre el centro del disco (frames).")
    parser.add_argument("--max-gap", type=int, default=5,
                        help="Huecos de hasta N frames se interpolan; mas largos cortan.")
    parser.add_argument("--disk-diameter", type=float, default=DISK_DIAMETER_M,
                        help="Diametro real del disco en metros.")
    parser.add_argument("--outdir", default=None,
                        help="Carpeta de salida (por defecto, la del CSV).")
    return parser.parse_args()


def detect_window(t: np.ndarray, cy: np.ndarray, min_amp: float,
                  max_gap: float, pad: float) -> tuple[tuple[float, float], int] | None:
    """Encuentra la serie de trabajo buscando las repeticiones, no la ventana.

    Una repeticion deja dos excursiones grandes y seguidas en el recorrido del
    disco; el paseo con la barra deja una sola, larga y aislada. Filtrar por
    amplitud y luego agrupar por cercania temporal separa una cosa de la otra.
    Devuelve ((t0, t1), n_series) o None si no hay nada periodico.
    """
    fps = 1.0 / float(np.median(np.diff(t)))
    y = smooth(cy, max(3, int(0.25 * fps)))

    # puntos de retorno = cambios de signo de la velocidad
    v = np.gradient(y)
    sign = np.sign(v)
    sign[sign == 0] = 1
    tp = np.flatnonzero(np.diff(sign) != 0) + 1
    if len(tp) < 3:
        return None

    rango = float(np.nanpercentile(y, 97) - np.nanpercentile(y, 3))
    exc = [(a, b) for a, b in zip(tp, tp[1:]) if abs(y[b] - y[a]) >= min_amp * rango]
    if not exc:
        return None

    # cada grupo de excursiones proximas en el tiempo es una serie
    grupos = [[exc[0]]]
    for e in exc[1:]:
        if t[e[0]] - t[grupos[-1][-1][1]] <= max_gap:
            grupos[-1].append(e)
        else:
            grupos.append([e])
    mejor = max(grupos, key=lambda g: t[g[-1][1]] - t[g[0][0]])
    t0 = max(float(t[0]), float(t[mejor[0][0]]) - pad)
    t1 = min(float(t[-1]), float(t[mejor[-1][1]]) + pad)
    return (t0, t1), len(grupos)


def load_track(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Lee el CSV y devuelve (tiempo_s, centro_y_px, diametro_px), NaN si no hubo disco."""
    t, cy, diam = [], [], []
    with open(path) as fh:
        for row in csv.DictReader(fh):
            t.append(float(row["tiempo_s"]))
            cy.append(float(row["centro_y_px"]) if row["centro_y_px"] else np.nan)
            diam.append(float(row["diametro_area_px"]) if row["diametro_area_px"] else np.nan)
    return np.array(t), np.array(cy), np.array(diam)


def fill_gaps(y: np.ndarray, max_gap: int) -> np.ndarray:
    """Interpola los huecos de hasta max_gap frames; los mas largos siguen a NaN."""
    out = y.copy()
    ok = ~np.isnan(y)
    if ok.sum() < 2:
        return out
    idx = np.arange(len(y))
    out[~ok] = np.interp(idx[~ok], idx[ok], y[ok])
    i = 0
    while i < len(y):
        if ok[i]:
            i += 1
            continue
        j = i
        while j < len(y) and not ok[j]:
            j += 1
        if j - i > max_gap or i == 0 or j == len(y):
            out[i:j] = np.nan  # hueco largo o extremo: no inventamos datos
        i = j
    return out


def smooth(y: np.ndarray, win: int) -> np.ndarray:
    """Media movil centrada que ignora los NaN en lugar de propagarlos."""
    if win <= 1:
        return y.copy()
    k = np.ones(win) / win
    valid = ~np.isnan(y)
    num = np.convolve(np.where(valid, y, 0.0), k, mode="same")
    den = np.convolve(valid.astype(float), k, mode="same")
    out = np.full(len(y), np.nan)
    np.divide(num, den, out=out, where=den > 0)
    out[~valid] = np.nan
    return out


def state_series(h: np.ndarray, top: float, bottom: float) -> list[str | None]:
    """Maquina de 4 estados sobre la altura normalizada.

    En la banda intermedia el estado depende del ultimo extremo visitado, que es
    lo que da la histeresis: sin haber tocado 'arriba' no se puede estar bajando.
    """
    states: list[str | None] = []
    ultimo_extremo: str | None = None
    for v in h:
        if np.isnan(v):
            states.append(None)
            continue
        if v >= top:
            ultimo_extremo = ARRIBA
            states.append(ARRIBA)
        elif v <= bottom:
            ultimo_extremo = ABAJO
            states.append(ABAJO)
        elif ultimo_extremo == ARRIBA:
            states.append(BAJANDO)
        elif ultimo_extremo == ABAJO:
            states.append(SUBIENDO)
        else:
            states.append(None)  # aun no sabemos de donde venimos
    return states


def runs_of(states: list[str | None], kind: str) -> list[tuple[int, int]]:
    """Tramos contiguos [inicio, fin) en los que el estado es `kind`."""
    out, i = [], 0
    while i < len(states):
        if states[i] != kind:
            i += 1
            continue
        j = i
        while j < len(states) and states[j] == kind:
            j += 1
        out.append((i, j))
        i = j
    return out


def extend_phase(v: np.ndarray, i0: int, i1: int, direccion: float,
                 alpha: float) -> tuple[int, int]:
    """Extiende una fase hacia fuera mientras el disco siga moviendose en el mismo sentido.

    El nucleo de la fase (entre umbrales) cubre solo el 70% del recorrido. Extenderlo
    por velocidad recupera el resto sin meterse en la meseta: se para en cuanto el
    movimiento cae por debajo de alpha veces el pico de la fase o cambia de sentido,
    que es justo el punto de retorno. Buscar el maximo de la meseta no vale: si el
    atleta se mueve un poco de pie, el maximo cae lejos del inicio real del movimiento.
    """
    pico = float(np.nanmax(np.abs(v[i0:i1 + 1])))
    if not np.isfinite(pico) or pico <= 0:
        return i0, i1
    umbral = alpha * pico

    a = i0
    while a > 0 and np.isfinite(v[a - 1]) and v[a - 1] * direccion >= umbral:
        a -= 1
    b = i1
    while b < len(v) - 1 and np.isfinite(v[b + 1]) and v[b + 1] * direccion >= umbral:
        b += 1
    return a, b


def build_phases(h: np.ndarray, states: list[str | None], v: np.ndarray,
                 alpha: float) -> list[dict]:
    """Cada tramo contiguo de 'bajando' o 'subiendo' es una fase, extendida por velocidad."""
    fases = []
    for kind, direccion in ((BAJANDO, -1.0), (SUBIENDO, 1.0)):
        for i0, i1 in runs_of(states, kind):
            a, b = extend_phase(v, i0, i1 - 1, direccion, alpha)
            fases.append({"fase": kind, "i0": a, "i1": b})
    fases.sort(key=lambda f: f["i0"])
    return fases


def phase_metrics(f: dict, t: np.ndarray, cy: np.ndarray, cy_s: np.ndarray,
                  diam: np.ndarray, disk_m: float) -> dict:
    """Metricas VBT de una fase. La escala px->m sale del diametro medio del disco."""
    i0, i1 = f["i0"], f["i1"]
    sl = slice(i0, i1 + 1)
    dur = float(t[i1] - t[i0])
    diam_px = float(np.nanmean(diam[sl]))
    m_por_px = disk_m / diam_px
    desp_px = float(abs(cy_s[i1] - cy_s[i0]))

    dt = np.diff(t[sl])
    vel_px = np.abs(np.diff(cy_s[sl])) / np.where(dt > 0, dt, np.nan)
    v_pico = float(np.nanmax(vel_px) * m_por_px) if len(vel_px) else float("nan")

    return {**f,
            "t0": float(t[i0]), "t1": float(t[i1]), "dur": dur,
            "desp_px": desp_px, "diam_px": diam_px, "m_por_px": m_por_px,
            "desp_m": desp_px * m_por_px,
            "v_media": desp_px * m_por_px / dur if dur > 0 else float("nan"),
            "v_pico": v_pico}


def plot(path: Path, t: np.ndarray, h: np.ndarray, fases: list[dict],
         top: float, bottom: float) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(14, 5))
    for f in fases:
        ax.axvspan(f["t0"], f["t1"], color="#d62728" if f["fase"] == BAJANDO else "#2ca02c",
                   alpha=0.16, lw=0)
        if f["fase"] == SUBIENDO:
            ax.annotate(f"{f['rep']}", (0.5 * (f["t0"] + f["t1"]), 1.04),
                        ha="center", fontsize=9, color="#2ca02c")
    ax.axhline(top, ls=":", lw=0.9, color="gray")
    ax.axhline(bottom, ls=":", lw=0.9, color="gray")
    ax.plot(t, h, lw=1.4, color="#1f77b4")
    ax.set_xlabel("tiempo (s)")
    ax.set_ylabel("altura normalizada")
    ax.set_title(f"{path.stem} — rojo: bajada, verde: subida (numeradas por repeticion)")
    ax.set_ylim(-0.05, 1.12)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    src = Path(args.csv)
    outdir = Path(args.outdir) if args.outdir else src.parent
    outdir.mkdir(parents=True, exist_ok=True)
    stem = src.stem.replace("_altura", "")

    t, cy, diam = load_track(src)

    manual = args.start is not None or args.end is not None
    aviso = ""
    if not manual and not args.no_auto_window:
        det = detect_window(t, cy, args.min_amp, args.max_gap_s, args.pad)
        if det is None:
            aviso = "no se detectó ninguna serie periódica; se analiza el vídeo entero"
        else:
            (args.start, args.end), n_series = det
            aviso = (f"ventana detectada {args.start:.1f}-{args.end:.1f}s"
                     + (f" ({n_series} series en el vídeo, se analiza la más larga)"
                        if n_series > 1 else ""))

    win = np.ones(len(t), dtype=bool)
    if args.start is not None:
        win &= t >= args.start
    if args.end is not None:
        win &= t <= args.end
    t, cy, diam = t[win], cy[win], diam[win]
    if len(t) < 10:
        raise SystemExit("La ventana seleccionada no tiene datos suficientes.")

    # la escala px->m se apoya en el diámetro: si se mueve dentro de la serie, avisar
    cv = 100.0 * float(np.nanstd(diam)) / float(np.nanmean(diam))

    cy_f = fill_gaps(cy, args.max_gap)
    cy_s = smooth(cy_f, args.smooth)

    # normalizar dentro de la ventana: 1 = lo mas alto, 0 = lo mas bajo
    y_min, y_max = np.nanmin(cy_s), np.nanmax(cy_s)
    h = (y_max - cy_s) / ((y_max - y_min) or 1.0)

    states = state_series(h, args.top, args.bottom)
    vel_h = np.gradient(h, t)  # velocidad en altura normalizada por segundo
    fases = build_phases(h, states, vel_h, args.vel_frac)
    fases = [phase_metrics(f, t, cy, cy_s, diam, args.disk_diameter) for f in fases]
    fases = [f for f in fases if f["dur"] >= args.min_phase]

    # una repeticion = bajada + la subida que la sigue
    rep = 0
    for f in fases:
        if f["fase"] == BAJANDO:
            rep += 1
        f["rep"] = max(rep, 1)

    csv_path = outdir / f"{stem}_fases.csv"
    with open(csv_path, "w") as fh:
        fh.write("rep,fase,t_inicio_s,t_fin_s,duracion_s,desplaz_px,diametro_medio_px,"
                 "m_por_px,desplaz_m,vel_media_m_s,vel_pico_m_s\n")
        for f in fases:
            fh.write(f"{f['rep']},{f['fase']},{f['t0']:.3f},{f['t1']:.3f},{f['dur']:.3f},"
                     f"{f['desp_px']:.1f},{f['diam_px']:.1f},{f['m_por_px']:.6f},"
                     f"{f['desp_m']:.4f},{f['v_media']:.4f},{f['v_pico']:.4f}\n")

    png_path = outdir / f"{stem}_fases.png"
    plot(png_path, t, h, fases, args.top, args.bottom)

    print(f"\n=== {stem}: {sum(f['fase'] == SUBIENDO for f in fases)} repeticiones ===")
    if aviso:
        print(aviso)
    if cv > 5.0:
        print(f"AVISO: el diámetro varía un {cv:.1f}% dentro de la ventana "
              f"(cámara o atleta en movimiento); la escala px->m es menos fiable.")
    print("rep  fase      t (s)          dur    ROM      diam      vel media   vel pico")
    for f in fases:
        print(f"{f['rep']:3d}  {f['fase']:8s} {f['t0']:6.2f}-{f['t1']:6.2f}  {f['dur']:5.2f}s  "
              f"{f['desp_m']:.3f} m  {f['diam_px']:6.1f}px  {f['v_media']:6.3f} m/s  "
              f"{f['v_pico']:6.3f} m/s")

    conc = [f for f in fases if f["fase"] == SUBIENDO]
    if conc:
        v = np.array([f["v_media"] for f in conc])
        print(f"\nconcentrico: media {v.mean():.3f} m/s  |  mejor {v.max():.3f}  "
              f"|  ultima {v[-1]:.3f}  |  perdida de velocidad {100 * (1 - v[-1] / v.max()):.1f}%")
        rom = np.array([f["desp_m"] for f in conc])
        print(f"ROM concentrico: {rom.mean():.3f} +- {rom.std():.3f} m")
    print(f"\nfases: {csv_path}\ngrafica: {png_path}")


if __name__ == "__main__":
    main()
