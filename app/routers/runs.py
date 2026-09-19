import base64
import csv
import os
import uuid
from pathlib import Path

import numpy as np
from fastapi import APIRouter, Depends, Form, Request, Response, UploadFile
from fastapi.responses import HTMLResponse
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.classes.vbt_profile import ExerciseProfile, VBTProfile
from app.config import settings
from app.db import get_session
from app.models import Run
from app.tasks import process_video
from app.templating import templates
from app.vbt.phases import analyze_phases

router = APIRouter()


def _render_row(request: Request, run: Run) -> str:
    return templates.get_template("partials/run_list_item.html").render(request=request, run=run)


def _render_detail(request: Request, run: Run) -> str:
    return templates.get_template("partials/run_detail.html").render(request=request, run=run)


def _toast(
    request: Request,
    kind: str,
    title: str,
    message: str,
    exercise: ExerciseProfile | None = None,
    template: str = "partials/toast_oob.html",
):
    """Aviso que se inyecta en #toast con un swap out-of-band.

    Si se pasa `exercise` el aviso lleva un botón para abrir el modal de
    creación de perfil ya con ese ejercicio seleccionado.
    """
    return templates.TemplateResponse(request, template, {
        "kind": kind,
        "title": title,
        "message": message,
        "exercise": exercise,
        "exercises": list(ExerciseProfile),
    })


@router.get("/runs")
def list_runs(request: Request, session: Session = Depends(get_session)):
    runs = session.scalars(select(Run).order_by(Run.created_at.desc())).all()
    return templates.TemplateResponse(request, "partials/run_list.html", {"runs": runs})


@router.post("/runs", response_class=HTMLResponse)
def create_run(
    request: Request,
    response: Response,
    video: UploadFile,
    name: str = Form(...),
    exercise: ExerciseProfile | None = Form(None),
    weight: float = Form(...),
    use_profile: bool = Form(True),
    session: Session = Depends(get_session),
):
    if use_profile and exercise is None:
        return _toast(
            request, "error", "Pick an exercise",
            "“Use profile” needs an exercise so we know which profile to load.",
        )

    if use_profile and not exercise.filepath.exists():
        return _toast(
            request, "error", f"No profile for {exercise.value}",
            "Build a velocity profile for this exercise, or uncheck "
            "“Use profile” to analyze this run without it.",
            exercise=exercise,
        )

    run_id = uuid.uuid4()
    suffix = Path(video.filename or "video.mp4").suffix or ".mp4"
    video_filename = f"{run_id}{suffix}"
    video_path = settings.uploads_dir / video_filename
    with open(video_path, "wb") as fh:
        fh.write(video.file.read())

    run = Run(id=run_id, name=name, exercise=exercise.value if exercise else None, weight=weight,
               status="pending", video_filename=video_filename, profile_used=use_profile)
    session.add(run)
    session.commit()
    session.refresh(run)

    process_video.delay(str(run_id))

    # el formulario se vacía al oír este evento (y no en after-request), para
    # que un aviso de error no le borre al usuario el vídeo ya seleccionado
    response.headers["HX-Trigger"] = "vbt-run-created"

    row_html = _render_row(request, run)
    detail_html = _render_detail(request, run)
    return (
        row_html
        + f'<div id="run-detail-panel" hx-swap-oob="innerHTML">{detail_html}</div>'
    )


@router.get("/runs/{run_id}/row", response_class=HTMLResponse)
def run_row(request: Request, run_id: uuid.UUID, session: Session = Depends(get_session)):
    run = session.get(Run, run_id)
    return _render_row(request, run)


@router.get("/runs/{run_id}", response_class=HTMLResponse)
def run_detail(request: Request, run_id: uuid.UUID, session: Session = Depends(get_session)):
    run = session.get(Run, run_id)
    return _render_detail(request, run)


def _read_track_csv(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    t, cy, diam = [], [], []
    with open(path) as fh:
        for row in csv.DictReader(fh):
            t.append(float(row["time_s"]))
            cy.append(float(row["center_y_px"]) if row["center_y_px"] else np.nan)
            diam.append(float(row["diameter_px"]) if row["diameter_px"] else np.nan)
    return np.array(t), np.array(cy), np.array(diam)


@router.get("/runs/{run_id}/chart-data")
def run_chart_data(run_id: uuid.UUID, session: Session = Depends(get_session)):
    run = session.get(Run, run_id)
    csv_path = settings.runs_dir / str(run_id) / "track.csv"
    t, cy, diam = _read_track_csv(csv_path)
    height, phases = analyze_phases(t, cy, diam, settings.disk_diameter_m)

    return {
        "t": t.tolist(),
        "height": [None if np.isnan(v) else v for v in height],
        "phases": [
            {
                "rep": p.rep,
                "phase": p.kind,
                "t0": p.t0,
                "t1": p.t1,
                "duration": p.duration,
                "v_avg": p.v_avg,
                "v_peak": p.v_peak,
                "v_sticking": p.v_sticking,
                "rir": p.rir,
            }
            for p in phases
        ],
    }


def _exercise_or_none(value: str | None) -> ExerciseProfile | None:
    """El <select> manda exercise="" mientras no se elige nada: eso no es un 422."""
    if not value:
        return None
    try:
        return ExerciseProfile.from_key(value)
    except ValueError:
        return None


@router.get("/profile/create_profile", response_class=HTMLResponse)
def create_profile_modal(request: Request, exercise: str | None = None):
    """Modal del perfil: enseña el guardado y deja crear/reemplazarlo."""
    return templates.TemplateResponse(request, "partials/profile_modal.html", {
        "exercises": list(ExerciseProfile),
        "selected": _exercise_or_none(exercise),
    })


@router.get("/profile/chart", response_class=HTMLResponse)
def profile_chart(request: Request, exercise: str | None = None):
    """Fragmento con la gráfica del perfil guardado, para el modal."""
    selected = _exercise_or_none(exercise)
    chart_b64 = None

    if selected is not None and selected.filepath.exists():
        profile = VBTProfile.load_profile_from_file(selected.value)
        chart_b64 = base64.b64encode(profile.plot_profile()).decode()

    return templates.TemplateResponse(request, "partials/profile_chart.html", {
        "exercise": selected,
        "chart_b64": chart_b64,
    })


@router.get("/profile/video-row", response_class=HTMLResponse)
def profile_video_row(request: Request):
    """Una fila más (vídeo + peso) para el modal, así no hace falta JS."""
    return templates.TemplateResponse(request, "partials/profile_video_row.html")


@router.post("/profile/{exercise_name}", response_class=HTMLResponse)
def create_profile(
    request: Request,
    exercise_name: ExerciseProfile,
    videos: list[UploadFile],
    weights: list[float],
):
    """
    Cargamos los videos para el perfil, los guardamos, y llamamos al perfil con la funcion de creacion desde videos
    """

    if len(videos) < 2 or len(weights) != len(videos):
        # la recta velocidad-peso necesita al menos dos puntos
        return _toast(
            request, "error", "Not enough sets",
            "A profile needs at least two videos, each with its own weight.",
            template="partials/profile_result.html",
        )

    video_list = []

    profile_dir = Path(os.getenv("BASE_PROFILE_PATH")) / exercise_name.name.lower()
    profile_dir.mkdir(parents=True, exist_ok=True)

    for i, video in enumerate(videos):

        suffix = Path(video.filename or "video.mp4").suffix or ".mp4"
        video_filename = f"profile_video_{i}{suffix}"
        video_path = profile_dir / video_filename
        with open(video_path, "wb") as fh:
            fh.write(video.file.read())

        video_list.append((video_path, weights[i]))

    try:
        _ = VBTProfile.from_videos(exercise_name.value, video_list)
    except Exception:
        # el modal se vuelve a pintar (template profile_result) para reintentar
        return _toast(
            request, "error", "Couldn’t build the profile",
            "Something failed while processing the videos — check the API logs and try again.",
            template="partials/profile_result.html",
        )

    return _toast(
        request, "success", f"{exercise_name.value} profile ready",
        "New runs on this exercise can now estimate your daily 1RM and RIR.",
        template="partials/profile_result.html",
    )
