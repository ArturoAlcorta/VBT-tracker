from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    database_url: str = "postgresql+psycopg2://vbt:vbt@localhost:5432/vbt"
    celery_broker_url: str = "amqp://guest:guest@localhost:5672//"
    celery_result_backend: str = "redis://localhost:6379/0"
    redis_url: str = "redis://localhost:6379/0"

    model_weights: Path = PROJECT_DIR / "models" / "best.pt"
    device: str = "cpu"
    conf_threshold: float = 0.4
    disk_diameter_m: float = 0.45
    # frames per forward pass during YOLO inference; 1 (the ultralytics
    # default for a video `source` under stream=True) leaves throughput on
    # the table for no accuracy cost — verified bit-identical detections at
    # batch=1 vs 8 on a real video, ~17% faster on GPU, a smaller but real
    # gain on CPU too. 8 was the sweet spot for this model; 16 didn't help
    # further on the one GPU (RTX 3070) this was benchmarked on.
    yolo_batch_size: int = 8

    data_dir: Path = PROJECT_DIR / "data"

    @property
    def uploads_dir(self) -> Path:
        return self.data_dir / "uploads"

    @property
    def runs_dir(self) -> Path:
        return self.data_dir / "runs"


settings = Settings()
settings.uploads_dir.mkdir(parents=True, exist_ok=True)
settings.runs_dir.mkdir(parents=True, exist_ok=True)
