from celery import Celery

from app.config import settings

celery_app = Celery(
    "vbt_tracker",
    broker=settings.celery_broker_url,
    backend=settings.celery_result_backend,
)
celery_app.conf.task_routes = {"app.tasks.process_video": {"queue": "vbt"}}
celery_app.autodiscover_tasks(["app"])
