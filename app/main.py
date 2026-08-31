from pathlib import Path

from fastapi import Depends, FastAPI, Request
from fastapi.staticfiles import StaticFiles
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db import get_session, init_db
from app.models import Run
from app.routers import events, runs
from app.templating import templates

app = FastAPI(title="vbt-tracker")

app.mount("/static", StaticFiles(directory=Path(__file__).resolve().parent / "static"), name="static")
app.include_router(runs.router)
app.include_router(events.router)


@app.on_event("startup")
def on_startup() -> None:
    init_db()


@app.get("/")
def index(request: Request, session: Session = Depends(get_session)):
    all_runs = session.scalars(select(Run).order_by(Run.created_at.desc())).all()
    return templates.TemplateResponse(request, "index.html", {"runs": all_runs})
