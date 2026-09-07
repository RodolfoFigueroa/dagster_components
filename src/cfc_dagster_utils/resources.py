from collections.abc import Generator
from contextlib import contextmanager

import sqlalchemy

from cfc_dagster_utils._optional import raise_optional_dependency_error

try:
    import dagster as dg
except ModuleNotFoundError as error:
    raise_optional_dependency_error(
        error,
        import_name="dagster",
        dependency_name="dagster",
        extra="postgres",
    )

try:
    import psycopg  # noqa: F401
except ModuleNotFoundError as error:
    raise_optional_dependency_error(
        error,
        import_name="psycopg",
        dependency_name="psycopg",
        extra="postgres",
    )

from pydantic import PrivateAttr


class PostgresResource(dg.ConfigurableResource):
    """Provide SQLAlchemy connections for Dagster execution."""

    host: str
    port: int
    user: str
    password: str
    database: str

    _engine: sqlalchemy.engine.Engine = PrivateAttr()

    def setup_for_execution(self, context: dg.InitResourceContext) -> None:  # noqa: ARG002
        """Create the SQLAlchemy engine used during Dagster execution."""
        url = sqlalchemy.URL.create(
            "postgresql+psycopg",
            username=self.user,
            password=self.password,
            host=self.host,
            port=self.port,
            database=self.database,
        )
        self._engine = sqlalchemy.create_engine(url)

    def teardown_after_execution(self, context: dg.InitResourceContext) -> None:  # noqa: ARG002
        """Dispose pooled database connections after Dagster execution."""
        self._engine.dispose()

    @contextmanager
    def connect(self) -> Generator[sqlalchemy.engine.Connection, None, None]:
        """Yield a connection without automatically committing a transaction."""
        with self._engine.connect() as conn:
            yield conn

    @contextmanager
    def begin(self) -> Generator[sqlalchemy.engine.Connection, None, None]:
        """Yield a connection in a commit-or-rollback transaction."""
        with self._engine.begin() as conn:
            yield conn
