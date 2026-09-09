import uuid
from datetime import datetime

from sqlalchemy import DateTime, Float, String, func, Boolean, Integer
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column

from app.db import SCHEMA, Base


class Run(Base):
    __tablename__ = "runs"
    __table_args__ = {"schema": SCHEMA}

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name: Mapped[str] = mapped_column(String, nullable=False)
    exercise: Mapped[str | None] = mapped_column(String, nullable=True)
    weight: Mapped[float | None] = mapped_column(Float, nullable=True)
    status: Mapped[str] = mapped_column(String, nullable=False, default="pending")
    celery_task_id: Mapped[str | None] = mapped_column(String, nullable=True)
    video_filename: Mapped[str] = mapped_column(String, nullable=False)
    profile_used: Mapped[bool] = mapped_column(Boolean, nullable=True)
    calculated_1rm: Mapped[float] = mapped_column(Boolean, nullable=True)
    final_rep_rir: Mapped[int] = mapped_column(Integer, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
