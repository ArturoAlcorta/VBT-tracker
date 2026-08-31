# vbt-tracker

Velocity-based training (VBT) from a phone video: point a camera at a weight
plate, upload the clip, get bar-path velocity per repetition — no wearable
sensor or linear encoder required.

## Index

- [Motivation](#motivation)
- [Quick start](#quick-start)
- [Velocity-based training](#velocity-based-training)
- [How it works](#how-it-works)
- [Dataset](#dataset)
- [Training metrics](#training-metrics)
- [Design choices](#design-choices)
- [Architecture](#architecture)
- [Standalone scripts](#standalone-scripts)
- [License](#license)

## Motivation

Linear position transducers ("encoders") are the standard tool for VBT, but a
decent one costs several hundred euros — out of reach for most lifters
training on their own. Every barbell setup already has a cheap, plentiful,
perfectly circular reference object sitting on it: the weight plates. If a
phone camera plus a small object-detection model can track a plate closely
enough, it becomes a semi-accurate, essentially free substitute for an
encoder.

"Semi-accurate" is the operative word — this project has **not** yet been
validated against a real linear encoder on the same reps. That comparison is
planned as the next step; until then, treat the velocity numbers as directionally
useful (spotting fatigue, comparing reps within a session) rather than
lab-grade measurements.

## Quick start

Requirements: Docker, and access to a **Redis** and a **RabbitMQ** instance
(broker + result backend / pub-sub for progress events). If you don't have
either yet, the bundled dev compose file starts throwaway ones for you.

```bash
git clone <this-repo> vbt-tracker
cd vbt-tracker
cp .env.example .env               # then edit CELERY_BROKER_URL / CELERY_RESULT_BACKEND / REDIS_URL

# with your own Redis/RabbitMQ:
docker compose up --build

# or, to also spin up throwaway Redis/RabbitMQ for local testing:
docker compose -f docker-compose.yml -f docker-compose.dev.yml up --build
```

Then open <http://localhost:8000>, upload a video, name the run, and watch the
status update live while it processes. By default the worker runs on CPU
(`DEVICE=cpu`); set `DEVICE=0` in `.env` (and uncomment the GPU block in
`docker-compose.yml`) if you have an NVIDIA GPU with the
`nvidia-container-toolkit` installed.

## Velocity-based training

VBT uses the speed of the barbell — not just the weight on it — to gauge how
hard a set actually is. The same load can be crushed at 0.9 m/s on a fresh set
and grind out at 0.3 m/s once fatigue sets in; %1RM alone can't see that,
bar-speed can. It's used both to autoregulate training load session-to-session
and to flag when a set has gone far enough into fatigue to hurt the next one.

Two references if you want the background beyond this README:

- González-Badillo, J.J. & Sánchez-Medina, L. (2010). *Movement Velocity as a
  Measure of Loading Intensity in Resistance Training.* International Journal
  of Sports Medicine.
- Weakley, J. et al. (2021). *Velocity-Based Training: From Theory to
  Application.* Strength & Conditioning Journal.

## How it works

1. **Detect** the plate in each video frame with a YOLO segmentation model
   (single class: `Disk`).
2. **Track** the plate's vertical center and mask diameter across frames
   (`app/vbt/track.py`); the diameter gives a pixel-to-meter scale, since a
   standard olympic plate is a known 45cm across.
3. **Segment into phases**: a 4-state hysteresis machine turns the normalized
   height curve into alternating eccentric/concentric phases
   (`app/vbt/phases.py`), and computes per-phase average velocity, peak
   velocity, and — for the concentric phase — the **sticking-point velocity**
   (the local velocity minimum after leaving the bottom, before the lift
   speeds back up).
4. **Show it**: an interactive chart (hover a phase to see its numbers) plus a
   saved PNG snapshot per run.

## Dataset

59 images (42 train / 20 val), hand-labeled with SAM in CVAT and exported as
Ultralytics YOLO segmentation format, single class (`Disk`). Images span
different backgrounds, plate colors, and plate brands to keep the model from
overfitting to one specific look.

## Training metrics

Model: `yolo26s-seg`, trained 400 epochs, batch 8, image size 640
(`disk-seg-s-v3` run). The shipped checkpoint (`models/best.pt`) is
Ultralytics' best-fitness snapshot, around epoch 118:

| Metric (mask)       | Value  |
|----------------------|--------|
| Precision             | 1.00   |
| Recall                 | 0.985  |
| mAP50                  | 0.995  |
| mAP50-95               | 0.995  |

Performance saturates early because, once the model has seen a handful of
plate colors/backgrounds, segmenting a single, high-contrast, perfectly
circular object is an easy task — which is also why a small dataset was
enough (see [Design choices](#design-choices)).

## Design choices

- **Only standard plates.** All olympic plates share the same 45cm diameter
  regardless of weight, so the pixel-to-meter scale needs no camera
  calibration — just the mask diameter. If your plates are visually very
  different from the ones in the training set (unusual color, bumper plates,
  heavy branding), you may need to fine-tune the model on a few labeled
  frames of your own (see `standalone_scripts/train.py`).
- **YOLO-s, not a bigger model.** Fast enough for near-real-time inference on
  a single GPU (or CPU, just slower), and segmenting one simple circular
  class doesn't need a larger model's capacity — which is also why so few
  training images were sufficient.

## Architecture

FastAPI + HTMX frontend, Celery worker for the actual video processing, and
Postgres to track runs (schema `vbt`, table `runs`). Progress is pushed to the
browser over Server-Sent Events, sourced from a Redis pub/sub channel per run
— no polling on either the worker or the browser side.

```
video upload -> FastAPI (POST /runs) -> Celery task -> YOLO tracking + phase analysis
                                              |
                                              v
                                   Redis pub/sub (progress) --> SSE --> browser
                                              |
                                              v
                                    Postgres (run status) + CSV/PNG on disk
```

Redis and RabbitMQ are **not** bundled in `docker-compose.yml` — bring your
own (or use `docker-compose.dev.yml` for local testing, see
[Quick start](#quick-start)).

## Standalone scripts

`standalone_scripts/` keeps the original CLI tools this project grew out of,
useful if you want to retrain the model or run the pipeline without the web
app:

- `train.py` — trains the YOLO-seg model on `disk_dataset/`.
- `predict.py` — runs inference on an image/video and saves the annotated output.
- `track_height.py` — tracks the plate in a video and plots its normalized height over time.
- `rep_phases.py` — splits a `track_height.py` CSV into reps/phases and computes VBT metrics.
- `fix_sleeve_labels.py` — dataset-cleanup utility (see its docstring).

Each has its own `--help`. `setup.sh` creates a local virtualenv for these
scripts (`ultralytics` and its dependencies only — no FastAPI/Celery needed
here).

## License

MIT — see [LICENSE](LICENSE).
