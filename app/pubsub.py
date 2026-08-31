"""Redis pub/sub for run-progress events.

The Celery worker publishes each progress update to a channel dedicated to
that run; the SSE endpoint subscribes to it and forwards messages to the
browser as they arrive. No polling on either side.
"""
from __future__ import annotations

import json
import uuid
from collections.abc import AsyncIterator

import redis
import redis.asyncio as aioredis

from app.config import settings

_sync_client: redis.Redis | None = None


def _channel(run_id: uuid.UUID) -> str:
    return f"run:{run_id}:events"


def publish_event(run_id: uuid.UUID, payload: dict) -> None:
    global _sync_client
    if _sync_client is None:
        _sync_client = redis.Redis.from_url(settings.redis_url)
    _sync_client.publish(_channel(run_id), json.dumps(payload))


async def subscribe(run_id: uuid.UUID) -> AsyncIterator[dict]:
    client = aioredis.Redis.from_url(settings.redis_url)
    pubsub = client.pubsub()
    await pubsub.subscribe(_channel(run_id))
    try:
        async for message in pubsub.listen():
            if message["type"] != "message":
                continue
            yield json.loads(message["data"])
    finally:
        await pubsub.unsubscribe(_channel(run_id))
        await pubsub.aclose()
        await client.aclose()
