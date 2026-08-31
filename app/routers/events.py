import json
import uuid

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session

from app.db import get_session
from app.models import Run
from app.pubsub import subscribe

router = APIRouter()


@router.get("/runs/{run_id}/events")
async def run_events(run_id: uuid.UUID, session: Session = Depends(get_session)):
    run = session.get(Run, run_id)
    initial_status = run.status if run else "error"

    async def event_stream():
        if initial_status in ("done", "error"):
            yield f"data: {json.dumps({'stage': initial_status, 'status': initial_status})}\n\n"
            return
        async for payload in subscribe(run_id):
            yield f"data: {json.dumps(payload)}\n\n"
            if payload.get("status") in ("done", "error"):
                break

    return StreamingResponse(event_stream(), media_type="text/event-stream")
