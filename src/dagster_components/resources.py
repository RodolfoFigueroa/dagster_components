import dagster as dg
import sqlalchemy
from pydantic import PrivateAttr
from typing import Generator
from contextlib import contextmanager


class PostGISResource(dg.ConfigurableResource):
    """PostGIS database resource for Dagster.
    This resource provides a configured connection to a PostGIS-enabled PostgreSQL database.
    It manages SQLAlchemy engine creation and connection lifecycle.
    Attributes:
        host (str): The hostname or IP address of the PostgreSQL server.
        port (str): The port number on which PostgreSQL is listening.
        user (str): The username for database authentication.
        password (str): The password for database authentication.
        db (str): The name of the database to connect to.
    """
    host: str
    port: str
    user: str
    password: str
    db: str

    _engine: sqlalchemy.engine.Engine = PrivateAttr()

    def setup_for_execution(self, context: dg.InitResourceContext) -> None:  # noqa: ARG002
        """
        Initialize the database engine for execution.

        This method is called by Dagster during resource initialization to set up
        the PostgreSQL database connection using SQLAlchemy.

        Args:
            context (dg.InitResourceContext): The Dagster resource initialization context.
                Unused in this implementation but required by the Dagster resource interface.

        Returns:
            None

        Raises:
            sqlalchemy.exc.ArgumentError: If the connection string format is invalid.
            sqlalchemy.exc.OperationalError: If the database connection cannot be established.

        Note:
            The database engine is stored in the `_engine` instance variable for use
            during resource execution. The connection uses psycopg2 as the PostgreSQL driver.
        """
        self._engine = sqlalchemy.create_engine(
            f"postgresql+psycopg2://{self.user}:{self.password}@{self.host}:{self.port}/{self.db}",
        )

    @contextmanager
    def connect(self) -> Generator[sqlalchemy.engine.Connection, None, None]:
        """
        Context manager that provides a SQLAlchemy database connection.

        Yields:
            sqlalchemy.engine.Connection: An active database connection from the engine's connection pool.

        Raises:
            Any exceptions raised by the engine's connect() method.

        Example:
            with resource.connect() as conn:
                result = conn.execute("SELECT * FROM table")
        """
        conn = None
        try:
            conn = self._engine.connect()
            yield conn
        finally:
            if conn is not None:
                conn.close()
