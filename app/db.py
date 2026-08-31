from collections.abc import Generator

from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from app.config import settings

SCHEMA = "vbt"

engine = create_engine(settings.database_url)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)


class Base(DeclarativeBase):
    pass


def get_session() -> Generator[Session, None, None]:
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()


def init_db() -> None:
    from sqlalchemy.schema import CreateSchema

    from app import models  # noqa: F401  (registra los modelos en Base.metadata)

    with engine.begin() as conn:
        if not conn.dialect.has_schema(conn, SCHEMA):
            conn.execute(CreateSchema(SCHEMA))
        Base.metadata.create_all(conn)
