import uuid
from pathlib import Path

from app.celery_app import celery_app
from app.config import settings
from app.db import SessionLocal
from app.models import Run
from app.pubsub import publish_event
from app.vbt.phases import CONCENTRIC, analyze_phases
from app.vbt.plotting import save_phases_plot
from app.vbt.track import track_disk, write_track_csv
from app.classes.vbt_profile import VBTProfile


def _set_status(run_id: uuid.UUID, status: str) -> None:
    with SessionLocal() as session:
        run = session.get(Run, run_id)
        if run is not None:
            run.status = status
            session.commit()


def _save_results(run_id: uuid.UUID, calculated_1rm: float | None, final_rep_rir: int | None) -> None:
    """Guarda las metricas del perfil y marca el run como done, en un commit.

    El frontend recarga el detalle al recibir el evento "done", asi que los
    datos tienen que estar ya en la base de datos antes de publicarlo.
    """
    with SessionLocal() as session:
        run = session.get(Run, run_id)
        if run is not None:
            run.calculated_1rm = calculated_1rm
            run.final_rep_rir = final_rep_rir
            run.status = "done"
            session.commit()


@celery_app.task(name="app.tasks.process_video", bind=True)
def process_video(self, run_id: str) -> None:
    rid = uuid.UUID(run_id)
    run_dir = settings.runs_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    with SessionLocal() as session:
        run = session.get(Run, rid)
        if run is None:
            return
        run.status = "processing"
        run.celery_task_id = self.request.id
        session.commit()
        video_path = settings.uploads_dir / run.video_filename
        run_name = run.name
        exercise_name = run.exercise
        profile_used = run.profile_used
        weight = run.weight

    try:
        publish_event(rid, {"stage": "tracking", "status": "processing"})
        result = track_disk(
            video_path, settings.model_weights, settings.conf_threshold, settings.device,
            batch_size=settings.yolo_batch_size,
        ) # result es un objeto TrackResult con fps, tiempo, center_y (array por frame) y diameter (array por frame)
        write_track_csv(run_dir / "track.csv", result)

        publish_event(rid, {"stage": "phases", "status": "processing"})
        height, phases = analyze_phases(result.t, result.center_y, result.diameter, settings.disk_diameter_m) # Height es un array con la altura normalizada de cada frame, phases es una lista de objetos Phase con info de cada fase (incluida velocidad media)
        save_phases_plot(run_dir / "phases.png", result.t, height, phases, run_name)

        calculated_1rm: float | None = None
        calculated_rir: int | None = None
        # la recta del perfil se construye con velocidades concentricas, asi que
        # las metricas se calculan solo con esas fases; sin ninguna (video sin
        # repeticiones detectadas) el run se guarda igual, pero sin metricas
        if profile_used:
            concentric = [p for p in phases if p.kind == CONCENTRIC]

            if concentric:
                profile = VBTProfile.load_profile_from_file(exercise_name)
                calculated_1rm = profile.calculate_daily_1rm(concentric, weight)
                calculated_rir = profile.calculate_rir(concentric)

        _save_results(rid, calculated_1rm, calculated_rir)
        publish_event(rid, {"stage": "done", "status": "done", "reps": max((p.rep for p in phases), default=0)})
    except Exception:
        _set_status(rid, "error")
        publish_event(rid, {"stage": "error", "status": "error"})
        raise
