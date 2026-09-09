import uuid
from pathlib import Path

from app.celery_app import celery_app
from app.config import settings
from app.db import SessionLocal
from app.models import Run
from app.pubsub import publish_event
from app.vbt.phases import analyze_phases
from app.vbt.plotting import save_phases_plot
from app.vbt.track import track_disk, write_track_csv
from app.classes.vbt_profile import VBTProfile


def _set_status(run_id: uuid.UUID, status: str) -> None:
    with SessionLocal() as session:
        run = session.get(Run, run_id)
        if run is not None:
            run.status = status
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

    try:
        publish_event(rid, {"stage": "tracking", "status": "processing"})
        result = track_disk(
            video_path, settings.model_weights, settings.conf_threshold, settings.device,
            batch_size=settings.yolo_batch_size,
        )
        write_track_csv(run_dir / "track.csv", result)

        publish_event(rid, {"stage": "phases", "status": "processing"})
        height, phases = analyze_phases(result.t, result.center_y, result.diameter, settings.disk_diameter_m)
        save_phases_plot(run_dir / "phases.png", result.t, height, phases, run_name)

        phases.sort(key=lambda x: x.v_avg, reverse=True)
        fastest_rep = {"speed": phases[0].v_avg,
                       "weight": phases[0].weight}

        if profile_used:
            calculated_1rm = VBTProfile.load_profile_from_file(exercise_name).calculate_daily_1rm(fastest_rep)
            # TODO: GUARDAR ESTE DATO EN LA BASE DE DATOS

            #Calculamos el RIR
            calculated_rir = VBTProfile.load_profile_from_file(exercise_name).calculate_rir(phases)
            # TODO: LO MISMO QUE EL 1RM


        _set_status(rid, "done")
        publish_event(rid, {"stage": "done", "status": "done", "reps": max((p.rep for p in phases), default=0)})
    except Exception:
        _set_status(rid, "error")
        publish_event(rid, {"stage": "error", "status": "error"})
        raise
