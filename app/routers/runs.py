import csv
import os
import uuid
from pathlib import Path

import numpy as np
from fastapi import APIRouter, Depends, Form, Request, UploadFile
from fastapi.responses import HTMLResponse
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.classes.vbt_profile import VBTProfile
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


@router.get("/runs")
def list_runs(request: Request, session: Session = Depends(get_session)):
    runs = session.scalars(select(Run).order_by(Run.created_at.desc())).all()
    return templates.TemplateResponse(request, "partials/run_list.html", {"runs": runs})


@router.post("/runs", response_class=HTMLResponse)
def create_run(
    request: Request,
    video: UploadFile,
    name: str = Form(...),
    exercise: str | None = Form(None),
    weight: float | None = Form(None),
    use_profile: bool = Form(True),
    session: Session = Depends(get_session),
):
    if use_profile and not VBTProfile.load_profile_from_file(exercise):
        return "MENSAJE DE ERROR DICIENDO QUE NO TIENE PERFIL, QUE LO CREE O QUITE EL CHECK DE USAR PERFIL"
    
    run_id = uuid.uuid4()
    suffix = Path(video.filename or "video.mp4").suffix or ".mp4"
    video_filename = f"{run_id}{suffix}"
    video_path = settings.uploads_dir / video_filename
    with open(video_path, "wb") as fh:
        fh.write(video.file.read())

    run = Run(id=run_id, name=name, exercise=exercise, weight=weight,
               status="pending", video_filename=video_filename, profile_used=use_profile)
    session.add(run)
    session.commit()
    session.refresh(run)

    process_video.delay(str(run_id))

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

@router.get("/profile/create_profile")
async def create_profile_modal():
    # TODO: Crear el model de crear perfil
    return "modal para crear perfil, donde se seleccione el ejercicio, y por cada video que a;adas tengas que a;adir un peso"


@router.post("/profile/{exercise_name}")
def create_profile(
    exercise_name: str,
    videos: list[UploadFile],
    weights: list[float]
):
    """
    Cargamos los videos para el perfil, los guardamos, y llamamos al perfil con la funcion de creacion desde videos
    """

    video_list = []

    for i, video in enumerate(videos):

        video_filename = f"profile_video_{i}"
        video_path = Path(os.getenv("BASE_PROFILE_PATH")) / exercise_name / video_filename
        with open(video_path, "wb") as fh:
            fh.write(video.file.read())

        video_list.append((video_path, weights[i]))

    
    _ = VBTProfile.from_videos(exercise_name, video_list)

    # TODO: crear el template de jinja para mostrar un pop up de perfil creado
    return "Profile created"


