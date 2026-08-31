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
- [Tests](#tests)
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

Three references if you want the background beyond this README:

- González-Badillo, J.J. & Sánchez-Medina, L. (2010). *Movement Velocity as a
  Measure of Loading Intensity in Resistance Training.* International Journal
  of Sports Medicine.
- Weakley, J. et al. (2021). *Velocity-Based Training: From Theory to
  Application.* Strength & Conditioning Journal.
- González-Badillo, J.J., Yañez-García, J.M., Mora-Custodio, R. &
  Rodríguez-Rosell, D. (2017). *Velocity Loss as a Variable for Monitoring
  Resistance Exercise.* International Journal of Sports Medicine, 38(3),
  217-225. — basis for the RIR estimate below.

## How it works

1. **Detect** the plate in each video frame with a YOLO segmentation model
   (single class: `Disk`).
2. **Track** the plate's vertical center and mask diameter across frames
   (`app/vbt/track.py`); the diameter gives a pixel-to-meter scale, since a
   standard olympic plate is a known 45cm across.
3. **Segment into phases** (`app/vbt/phases.py`): find the local minima
   ("bottoms") and maxima ("tops") of the normalized height curve via
   `scipy.signal.find_peaks`, filtered by a minimum time between reps
   (`MIN_REP_SECONDS`) and a minimum depth relative to the surrounding
   extrema (`MIN_PROMINENCE`) so a small wobble at the top of a rep, or
   sensor noise, doesn't get mistaken for its own rep. Each eccentric phase
   runs top → bottom, each concentric phase bottom → top. This replaced a
   4-state hysteresis machine over *fixed* height thresholds (`TOP=0.85` /
   `BOTTOM=0.15`, see git history): a rep that never reached that specific
   band — a shallower squat from fatigue, a camera that's drifted closer or
   further between sets, a lifter starting mid-ROM — could get dropped or
   misread, since the thresholds encoded no actual biomechanical event.
   Anchoring on real extrema fixes that: whatever depth a rep actually
   reaches, its own local min/max is still findable.

   Per-phase average velocity, peak velocity, and — for the concentric
   phase — **sticking-point velocity** are computed the same way as before:
   the latter now via the same peak-finder applied to velocity instead of
   height, so it's a genuine local minimum (a real dip the lift recovers
   from, not just wherever the phase happens to end). Not every rep has one
   — a rep that just accelerates once and coasts into lockout, with no
   mid-lift slowdown-then-push, has no sticking point, and this correctly
   returns `None` for it instead of forcing a value onto the nearest low
   point (which is what a fixed-window trim used to do; see git history).
4. **Estimate RIR**: each concentric rep's mean velocity maps directly to an
   RIR estimate via fixed thresholds — no comparison to other reps in the
   set. (An earlier version compared each rep to the *fastest* rep in the set
   instead; scrapped because it reports a high RIR for whichever rep happens
   to be fastest even when every rep in the set is objectively slow — it
   measures relative fatigue within the set, not actual proximity to
   failure.) The thresholds chain two tables from González-Badillo et al.
   (2017) (see [Velocity-based training](#velocity-based-training)): velocity
   → effective %1RM (their %1RM-velocity classification table, since velocity
   reflects current effort regardless of the prescribed load), then %1RM →
   mean reps to failure (their Table 1), interpolated between anchors and
   RIR = reps to failure − 1.

   The velocity fed in is `v_avg` (displacement/duration over the whole
   concentric phase) — the closest thing computed here to the paper's *mean
   propulsive velocity* (MPV: average velocity of just the accelerating
   portion of the rep, acceleration ≥ -9.81 m/s²). `v_avg` also includes the
   deceleration into lockout, since nothing here segments out a propulsive
   sub-phase, so it runs a bit low relative to true MPV — sticking-point
   velocity was tried first and dropped, since the literature behind these
   thresholds is built on MPV, not on a rep's local velocity minimum (that's
   a different line of research — locating the weakest point in the strength
   curve, e.g. Kompf & Arandjelović's sticking-point review — not fatigue/RIR
   estimation). Bench-press-specific either way (the equivalent squat/deadlift
   study, Rodríguez-Rosell et al. 2020, is paywalled) and applied to every
   exercise regardless. Treat the RIR figure as a rough, directional
   estimate, not a precise one.
5. **Show it**: an interactive chart (hover a phase to see its numbers) plus a
   per-rep table (duration, velocities, estimated RIR) and a saved PNG
   snapshot per run.

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

## Tests

```bash
uv pip install -e . --group dev   # main deps + pytest
pytest
```

Coverage is limited to `app/vbt/phases.py` (`tests/vbt/test_phases.py`) —
segmentation, sticking-point detection, and RIR estimation, since those are
the parts with actual numerical logic to get subtly wrong (and did, twice,
during development: sticking-point velocity picking up the coast into
lockout instead of a real dip, and a leading concentric fragment colliding
with rep 1's number). Runs against both synthetic signals (exact, deterministic
expectations) and a real recorded `track.csv` fixture (`tests/vbt/fixtures/`,
checked in) — the two real bugs above only showed up on actual noisy tracking
data, not clean synthetic ones, so both kinds of test earn their place.

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
