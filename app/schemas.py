import uuid
from datetime import datetime

from pydantic import BaseModel, ConfigDict


class RunOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    name: str
    exercise: str | None
    weight: float | None
    status: str
    video_filename: str
    created_at: datetime


class PhaseOut(BaseModel):
    rep: int
    phase: str  # "bajando" (eccentric) | "subiendo" (concentric)
    t0: float
    t1: float
    duration: float
    v_avg: float
    v_peak: float
    v_sticking: float | None = None  # solo para el concentrico
    rir: float | None = None  # RIR estimado, solo para el concentrico


class ChartData(BaseModel):
    t: list[float]
    height: list[float | None]
    phases: list[PhaseOut]
