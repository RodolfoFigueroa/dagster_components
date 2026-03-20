import dagster as dg
import sqlalchemy
from contextlib import contextmanager
from typing import Generator

class PostGISResource(dg.ConfigurableResource):
    host: str
    port: str
    user: str
    password: str
    db: str
    def setup_for_execution(self, context: dg.InitResourceContext) -> None: ...
    @contextmanager
    def connect(self) -> Generator[sqlalchemy.engine.Connection, None, None]: ...
