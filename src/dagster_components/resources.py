import dagster as dg
import sqlalchemy
from pydantic import PrivateAttr
from typing import Generator
from contextlib import contextmanager


class PostGISResource(dg.ConfigurableResource):
    host: str
    port: str
    user: str
    password: str
    db: str

    _engine: sqlalchemy.engine.Engine = PrivateAttr()

    def setup_for_execution(self, context: dg.InitResourceContext) -> None:  # noqa: ARG002
        self._engine = sqlalchemy.create_engine(
            f"postgresql+psycopg2://{self.user}:{self.password}@{self.host}:{self.port}/{self.db}",
        )

    @contextmanager
    def connect(self) -> Generator[sqlalchemy.engine.Connection, None, None]:
        conn = None
        try:
            conn = self._engine.connect()
            yield conn
        finally:
            if conn is not None:
                conn.close()
